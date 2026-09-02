"""
ingest.py

Loads the Pitchwise knowledge base, splits it into chunks along markdown
header boundaries, embeds those chunks using a free local embedding model,
and stores the result in a Chroma vector store.

This module implements the decisions recorded in DECISIONS.md:
  - D-001: MarkdownHeaderTextSplitter, not RecursiveCharacterTextSplitter
  - D-002: all-MiniLM-L6-v2 (local, free, no API dependency)

Design note: this is built to run fresh on every app startup rather than
persisting the vector store to disk. The knowledge base is small (17
documents, 81 chunks) and static, so re-embedding on boot takes only a few
seconds and avoids disk-persistence issues on free hosting tiers (e.g.
Hugging Face Spaces).
"""

import os
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KNOWLEDGE_BASE_PATH = "knowledge-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # D-002: free, local, no API key needed
COLLECTION_NAME = "pitchwise"

# D-001: split on header boundaries, not raw character count
HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
]


def load_documents():
    """Load every markdown file in the knowledge base, tagging each with
    its doc_type (players / formats / tournaments / icc) based on the
    subfolder it lives in.

    Reads files directly with plain Python rather than LangChain's
    DirectoryLoader/TextLoader — those live in langchain-community, which
    is being sunset, and since we only ever need to read plain .md files,
    the loader abstraction isn't buying us anything here."""
    documents = []
    folders = [
        f.path for f in os.scandir(KNOWLEDGE_BASE_PATH) if f.is_dir()
    ]

    for folder in folders:
        doc_type = os.path.basename(folder)
        for filename in os.listdir(folder):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": filepath, "doc_type": doc_type},
                )
            )

    return documents


def chunk_documents(documents):
    """Split documents on markdown header boundaries (D-001), so each
    chunk corresponds to one coherent section rather than an arbitrary
    character-count cut. Re-attaches source/doc_type metadata to every
    resulting chunk, since MarkdownHeaderTextSplitter otherwise drops it."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc.page_content)
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)
        chunks.extend(doc_chunks)

    return chunks


def build_vector_store(chunks):
    """Embed chunks with all-MiniLM-L6-v2 (D-002) and store them in an
    in-memory Chroma collection.

    Explicitly deletes any existing collection with the same name first —
    without this, re-running ingest() in the same process (e.g. calling it
    twice in one notebook session) silently appends a duplicate set of
    chunks into the same collection rather than replacing it."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Clear any existing collection with this name before rebuilding
    try:
        existing = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings)
        existing.delete_collection()
    except Exception:
        pass  # no existing collection to delete — fine, nothing to clean up

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        # no persist_directory — in-memory only, rebuilt fresh on each boot
    )
    return vectorstore


def ingest():
    """Full pipeline: load, chunk, embed, store. Returns a ready-to-query
    vector store."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {KNOWLEDGE_BASE_PATH}")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = build_vector_store(chunks)
    count = vectorstore._collection.count()
    sample_vec = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    print(f"Vector store built: {count} chunks embedded, {len(sample_vec)} dimensions")

    return vectorstore


if __name__ == "__main__":
    # Quick manual test: run `python ingest.py` to sanity-check ingestion
    # without starting the full app.
    store = ingest()

    print("\nSample retrieval test:")
    results = store.similarity_search("What is Don Bradman's exact career Test batting average?", k=1)
    for r in results:
        print(f"- [{r.metadata.get('doc_type')}] {r.page_content[:]}...")