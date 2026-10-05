"""
Offline tests for the knowledge-base pipeline: the Wikipedia converter
(scripts/build_kb.py, D-014) and loading, chunking and saved-index reuse in
ingest.py (D-015). No network and no embedding model: articles are scripted
and embeddings are LangChain's deterministic fake.

Run from the project root:  uv run pytest tests/ -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter

import build_kb
import ingest

EXTRACT = """Brian Lara is a former West Indies batter.
He holds the record for the highest Test score.

== Early life ==
Lara was born in Trinidad.

=== Schooling ===
He attended Fatima College.

==== Junior cricket ====
He played for Trinidad under-19s.

== Empty section ==

== Records ==
Lara scored 400 not out in 2004.
{\\displaystyle x^{2}}

== References ==
Some citation.

=== Books ===
A book.

== External links ==
A link.
"""


def test_headings_name_their_subject_and_reference_sections_are_dropped():
    md = build_kb.to_markdown("Brian Lara", EXTRACT)
    headings = [l for l in md.splitlines() if l.startswith("#")]
    assert headings == [
        "# Brian Lara",
        "## Brian Lara: Overview",
        "## Brian Lara: Early life",
        "## Brian Lara: Early life - Schooling",
        "## Brian Lara: Records",
    ]
    assert "**Junior cricket.**" in md        # level 4 folds into its parent
    assert "citation" not in md and "A book" not in md and "A link" not in md
    assert "displaystyle" not in md


def test_articles_are_capped_at_a_section_boundary():
    long_text = "Lead.\n\n" + "\n\n".join(f"== Part {i} ==\n" + "x" * 900 for i in range(40))
    md = build_kb.to_markdown("Long", long_text, max_chars=5000)
    assert len(md) <= 5000
    assert md.rstrip().endswith("x")          # never cut inside a section


def test_sources_file_is_valid_and_avoids_curated_subjects():
    sources = build_kb.read_sources()
    titles = [t for _, t in sources]
    assert len(titles) == len(set(titles)), "duplicate titles"
    assert not {t.lower() for t in titles} & build_kb.CURATED_SUBJECTS
    assert {c for c, _ in sources} == {"players", "teams", "tournaments", "grounds", "game"}
    assert 120 <= len(sources) <= 140


def fake_fetch(article_title="Brian Lara", text=EXTRACT):
    def fetch(title):
        if title == "Missing page":
            return None
        return {"title": article_title, "text": text, "revid": 123,
                "timestamp": "2026-10-01T00:00:00Z", "url": "https://en.wikipedia.org/wiki/Brian_Lara"}
    return fetch


def test_build_writes_attributed_files_and_skips_missing_pages(tmp_path):
    written, skipped = build_kb.build([("players", "Brian Lara"), ("players", "Missing page")],
                                      output_dir=tmp_path, fetch=fake_fetch(), pause=0,
                                      retrieved="2026-10-06")
    assert [p.name for p in written] == ["brian-lara.md"] and len(skipped) == 1
    text = written[0].read_text()
    assert "oldid=123" in text and "CC BY-SA 4.0" in text and "retrieved: 2026-10-06" in text


def test_build_refuses_a_curated_subject_even_through_a_redirect(tmp_path):
    with pytest.raises(ValueError):
        build_kb.build([("players", "Bradman")], output_dir=tmp_path,
                       fetch=fake_fetch(article_title="Don Bradman"), pause=0)


def test_curated_chunks_are_split_on_headings_only():
    """The size split and heading prefix (D-014) apply to Wikipedia
    documents only: curated chunks are exactly the heading split, as in
    v1/v2. (Four curated sections were re-headed in v3, C-006.)"""
    docs = [d for d in ingest.load_documents() if d.metadata["origin"] == "curated"]
    assert len(docs) == 17
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=ingest.HEADERS_TO_SPLIT_ON)
    expected = sorted(c.page_content for d in docs for c in splitter.split_text(d.page_content))
    got = sorted(c.page_content for c in ingest.chunk_documents(docs))
    assert got == expected and len(got) == 81


def make_kb(tmp_path, body_chars=3000):
    kb = tmp_path / "kb"
    (kb / "players").mkdir(parents=True)
    (kb / "players" / "curated.md").write_text("# Curated\n\n## Curated: Stats\n\nShort fact.\n")
    wiki = kb / "wikipedia" / "players"
    wiki.mkdir(parents=True)
    body = build_kb.to_markdown("Brian Lara", "Lead text.\n\n== Career ==\n" + "word " * (body_chars // 5))
    (wiki / "brian-lara.md").write_text(
        build_kb.front_matter({"title": "Brian Lara", "revid": 9, "timestamp": "t",
                               "url": "u"}, "2026-10-06") + body)
    return kb


def test_wikipedia_chunks_are_anchored_sized_and_attributed(tmp_path):
    docs = ingest.load_documents(make_kb(tmp_path))
    wiki = [d for d in docs if d.metadata["origin"] == "wikipedia"][0]
    assert wiki.metadata["doc_type"] == "players" and "oldid=9" in wiki.metadata["source_url"]
    assert not wiki.page_content.startswith("---")  # front matter is metadata, not text
    chunks = [c for c in ingest.chunk_documents(docs) if c.metadata["origin"] == "wikipedia"]
    career = [c for c in chunks if c.metadata.get("header_2") == "Brian Lara: Career"]
    assert len(career) >= 3                       # 3,000 characters split into pieces
    for c in chunks:
        heading = c.metadata["header_2"]
        assert c.page_content.startswith(heading + "\n")   # every piece names its subject
        assert len(c.page_content) <= ingest.MAX_CHUNK_CHARS + len(heading) + 1


def test_saved_index_is_reused_until_the_knowledge_base_changes(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    monkeypatch.setattr(ingest, "KNOWLEDGE_BASE_PATH", str(kb))
    fake = DeterministicFakeEmbedding(size=16)
    index = tmp_path / "index"

    built = ingest.ingest(persist_directory=str(index), embeddings=fake)
    n = built._collection.count()
    first = ingest.fingerprint(str(kb))
    first_dir = ingest.index_path(index, first)
    assert ingest.load_saved_index(first_dir, first, fake) is not None
    again = ingest.ingest(persist_directory=str(index), embeddings=fake)   # reused, not rebuilt
    assert again._collection.count() == n

    (kb / "players" / "curated.md").write_text("# Curated\n\n## Curated: Stats\n\nA changed fact.\n")
    second = ingest.fingerprint(str(kb))
    assert second != first
    assert ingest.load_saved_index(ingest.index_path(index, second), second, fake) is None  # stale
    rebuilt = ingest.ingest(persist_directory=str(index), embeddings=fake)  # same process, no clash
    assert rebuilt._collection.count() == n
    assert [p.name for p in index.iterdir()] == [ingest.index_path(index, second).name]  # old one removed


def test_table_only_chunks_are_dropped_but_short_facts_are_kept():
    assert not ingest.has_real_text("**Finals.**")
    assert not ingest.has_real_text("Last updated 25 October 2025.")
    assert not ingest.has_real_text("All records correct as of 11 January 2026.")
    assert ingest.has_real_text("Some informal baseball games use variations of the follow-on.")
    assert ingest.has_real_text("The six teams are all owned by existing IPL franchise owners.")


def test_markup_debris_is_cleaned_but_years_in_brackets_are_kept():
    assert build_kb.clean_line(
        "%5B%5BWikipedia%3ATemplates%5D%5D{{{CRITERION}}} The Pakistan Super League") == "The Pakistan Super League"
    assert build_kb.clean_line("he made 426 runs.[2] Later") == "he made 426 runs. Later"
    assert build_kb.clean_line("during the [2003] World Cup") == "during the [2003] World Cup"
    assert build_kb.clean_line("^ The win percentage excludes no-results") == ""


def test_v3_test_set_is_grounded_in_the_knowledge_base():
    """Every keyword of every answerable v3 question appears in the
    knowledge base (the v3 builder checks each against its source file)."""
    import json
    from evaluation.test import TEST_FILE_V3
    text = "\n".join(p.read_text(encoding="utf-8") for p in ingest.knowledge_base_files()).lower()
    tests = [json.loads(l) for l in Path(TEST_FILE_V3).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(tests) == 177
    for t in tests:
        if t["category"] == "out_of_scope":
            assert t["keywords"] == []
        for k in t["keywords"]:
            assert k.lower() in text, (t["question"], k)
