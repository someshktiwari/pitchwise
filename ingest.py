"""
ingest.py

Loads the Pitchwise knowledge base, splits it into chunks along markdown
header boundaries, embeds those chunks using a free local embedding model,
and stores the result in a Chroma vector store.

This module implements the decisions recorded in DECISIONS.md:
  - D-001: MarkdownHeaderTextSplitter, not RecursiveCharacterTextSplitter
  - D-002: all-MiniLM-L6-v2 (local, free, no API dependency)

  - D-014: the knowledge base has two parts: 17 curated documents and
    about 130 Wikipedia articles built by scripts/build_kb.py
  - D-015: the built index is saved to disk with a fingerprint of everything
    that shapes it, and reused until the knowledge base, the chunking or the
    embedding model changes (replaces D-005's rebuild-on-every-start)
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KNOWLEDGE_BASE_PATH = "knowledge-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # D-002: free, local, no API key needed
COLLECTION_NAME = "pitchwise"
INDEX_DIR = os.getenv("PITCHWISE_INDEX_DIR", ".index")  # D-015: saved index, gitignored

# D-014: all-MiniLM-L6-v2 reads at most 256 word pieces (~1,000 characters)
# of a chunk and ignores the rest, so sections longer than this are split
# further. Every curated section is shorter (longest: 708 characters), so
# the curated chunks are unchanged.
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
CHUNKING_VERSION = "3"  # bump when chunking changes, so a saved index is rebuilt
# A Wikipedia chunk with less real text than this (after its heading, bold
# sub-heading labels and "Last updated" notes) is dropped: these are sections
# whose content was a table, which the plain-text extract leaves out (D-014).
MIN_CHUNK_TEXT_CHARS = 40
_LABEL_LINE = re.compile(r"^\*\*.*\*\*$|^last updated\b.*$|^all records correct as of\b.*$|^\(as (on|of) .*\)$",
                         re.IGNORECASE)

# D-001: split on header boundaries, not raw character count
HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
]


def split_front_matter(text):
    """Separate a leading '---' block of 'key: value' lines (written by
    scripts/build_kb.py: title, source revision, licence) from the body."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, text[end + 5:].lstrip("\n")


def knowledge_base_files(root=None):
    """Every .md file under the knowledge base, in a fixed (sorted) order so
    the index and its fingerprint are reproducible. root defaults to
    KNOWLEDGE_BASE_PATH, read at call time."""
    root = root or KNOWLEDGE_BASE_PATH
    return sorted(p for p in Path(root).rglob("*.md") if p.is_file())


def load_documents(root=None):
    """Load every markdown file in the knowledge base. doc_type is the folder
    a file sits in (players, formats, tournaments, icc, teams, grounds,
    game); origin is "wikipedia" for files under knowledge-base/wikipedia/
    and "curated" for the rest (D-014). Front matter becomes metadata.

    Reads files directly with plain Python rather than LangChain's
    DirectoryLoader/TextLoader — those live in langchain-community, which
    is being sunset, and since we only ever need to read plain .md files,
    the loader abstraction isn't buying us anything here."""
    root = root or KNOWLEDGE_BASE_PATH
    documents = []
    for path in knowledge_base_files(root):
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root)
        metadata = {
            "source": str(Path(root) / rel),
            "doc_type": path.parent.name,
            "origin": "wikipedia" if rel.parts[0] == "wikipedia" else "curated",
        }
        if meta.get("source"):
            metadata["source_url"] = meta["source"]
        if meta.get("licence"):
            metadata["licence"] = meta["licence"]
        documents.append(Document(page_content=body, metadata=metadata))
    return documents


def chunk_documents(documents):
    """Split documents on markdown header boundaries (D-001), so each
    chunk corresponds to one coherent section rather than an arbitrary
    character-count cut. Re-attaches source/doc_type metadata to every
    resulting chunk, since MarkdownHeaderTextSplitter otherwise drops it.

    D-014, for the Wikipedia documents: each chunk starts with its section
    heading ("Brian Lara: Early life"), because the splitter removes
    headings from the text and the article body rarely repeats the name;
    and sections longer than MAX_CHUNK_CHARS are split again, with overlap,
    each piece keeping the heading. Curated chunks are left exactly as v1/v2
    built them."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    sizer = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS)

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc.page_content)
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)
        if doc.metadata.get("origin") != "wikipedia":
            chunks.extend(doc_chunks)
            continue
        for chunk in doc_chunks:
            heading = chunk.metadata.get("header_2") or chunk.metadata.get("header_1") or ""
            pieces = sizer.split_text(chunk.page_content) if len(chunk.page_content) > MAX_CHUNK_CHARS \
                else [chunk.page_content]
            for piece in pieces:
                if not has_real_text(piece):
                    continue
                text = f"{heading}\n{piece}" if heading else piece
                chunks.append(Document(page_content=text, metadata=dict(chunk.metadata)))

    return chunks


def has_real_text(text):
    """False for a chunk that is only labels and dates, e.g. a heading over
    a table the extract dropped: "**Finals.**" or "Last updated 25 October
    2025." Such chunks add retrieval noise and no answerable facts."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    real = " ".join(l for l in lines if not _LABEL_LINE.match(l))
    return len(real) >= MIN_CHUNK_TEXT_CHARS


def get_embeddings():
    """The embedding model (D-002). A function so tests can swap in a fake."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def fingerprint(root=None):
    """A hash of everything that decides what the index contains: every
    knowledge-base file (path and bytes), the embedding model and the
    chunking settings (D-015). A saved index is reused only if this matches."""
    root = root or KNOWLEDGE_BASE_PATH
    h = hashlib.sha256()
    h.update(f"{EMBEDDING_MODEL}|{CHUNKING_VERSION}|{MAX_CHUNK_CHARS}|{CHUNK_OVERLAP_CHARS}|"
             f"{MIN_CHUNK_TEXT_CHARS}|{_LABEL_LINE.pattern}".encode())
    for path in knowledge_base_files(root):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build_vector_store(chunks, persist_directory=None, embeddings=None):
    """Embed chunks with all-MiniLM-L6-v2 (D-002) and store them in Chroma,
    in memory or saved to persist_directory (D-015).

    Starts from an empty collection either way. In memory, it deletes any
    existing collection with the same name first: without this, re-running
    ingest() in the same process silently appends a duplicate set of chunks
    (C-001). On disk, it clears the target directory, which is a fresh,
    fingerprint-named one (see index_path)."""
    embeddings = embeddings or get_embeddings()

    if persist_directory:
        shutil.rmtree(persist_directory, ignore_errors=True)  # a half-built index from a crash
    else:
        try:
            existing = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings)
            existing.delete_collection()
        except Exception:
            pass  # no existing collection to delete — fine, nothing to clean up

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )


def index_path(persist_directory, fp):
    """Each index lives in a folder named after its fingerprint. A changed
    knowledge base builds into a new folder instead of deleting one that a
    running process (or Chroma's per-path client cache) still has open."""
    return Path(persist_directory) / fp[:16]


def _manifest_path(index_dir):
    return Path(index_dir) / "pitchwise-manifest.json"


def _remove_old_indexes(persist_directory, keep):
    for child in Path(persist_directory).iterdir():
        if child.is_dir() and child != keep:
            shutil.rmtree(child, ignore_errors=True)


def load_saved_index(index_dir, expected_fingerprint, embeddings=None):
    """The saved index in index_dir if it was built from exactly the current
    knowledge base and settings, otherwise None."""
    manifest = _manifest_path(index_dir)
    if not manifest.exists():
        return None
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if saved.get("fingerprint") != expected_fingerprint:
        return None
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings or get_embeddings(),
                  persist_directory=str(index_dir))


def ingest(persist_directory=INDEX_DIR, rebuild=False, embeddings=None):
    """Full pipeline: load, chunk, embed, store. Returns a ready-to-query
    vector store.

    With persist_directory set (the default, D-015), a saved index is reused
    when its fingerprint matches; otherwise the index is rebuilt and saved
    with a manifest. persist_directory=None builds in memory every time (the
    v1/v2 behaviour, D-005)."""
    fp = fingerprint()
    target = index_path(persist_directory, fp) if persist_directory else None
    if target and not rebuild:
        store = load_saved_index(target, fp, embeddings)
        if store is not None:
            print(f"Loaded saved index from {target}: {store._collection.count()} chunks")
            return store

    documents = load_documents()
    origins = {}
    for d in documents:
        origins[d.metadata["origin"]] = origins.get(d.metadata["origin"], 0) + 1
    print(f"Loaded {len(documents)} documents from {KNOWLEDGE_BASE_PATH} {origins}")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = build_vector_store(chunks, persist_directory=str(target) if target else None,
                                     embeddings=embeddings)
    count = vectorstore._collection.count()
    sample_vec = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    print(f"Vector store built: {count} chunks embedded, {len(sample_vec)} dimensions")

    if target:
        _manifest_path(target).write_text(json.dumps({
            "fingerprint": fp,
            "documents": len(documents),
            "chunks": count,
            "embedding_model": EMBEDDING_MODEL,
            "chunking_version": CHUNKING_VERSION,
        }, indent=2), encoding="utf-8")
        _remove_old_indexes(persist_directory, keep=target)
        print(f"Saved index to {target}")

    return vectorstore


if __name__ == "__main__":
    # Quick manual test: run `python ingest.py` to sanity-check ingestion
    # without starting the full app.
    store = ingest()

    print("\nSample retrieval test:")
    results = store.similarity_search("What is Don Bradman's exact career Test batting average?", k=1)
    for r in results:
        print(f"- [{r.metadata.get('doc_type')}] {r.page_content[:]}...")