"""
scripts/build_kb.py

Builds the Wikipedia part of the knowledge base (DECISIONS.md D-014): fetches
every article listed in knowledge-base/wikipedia-sources.txt from the
Wikipedia API, converts it to markdown in the same shape as the curated
documents, and writes it under knowledge-base/wikipedia/<category>/.

Run from the project root (needs internet access):

    uv run python scripts/build_kb.py            # fetch everything
    uv run python scripts/build_kb.py --only "Brian Lara" --only "Lord's"
    uv run python scripts/build_kb.py --dry-run   # parse sources, fetch nothing

What the conversion does, and why:
- Every heading names its subject ("## Brian Lara: Early life"), so a chunk
  still says whose early life it is once it's cut out of the article. This
  is the D-003 fix (the Bradman "Career Statistics" bug), applied by code.
- Reference sections (References, External links, See also, ...) and empty
  sections are dropped; they add tokens but no answerable facts.
- Each article is capped at MAX_DOC_CHARS, cutting only at section
  boundaries, so one huge article can't dominate the index.
- Each file starts with front matter recording the exact revision used and
  the licence: Wikipedia text is CC BY-SA 4.0, so it is attributed and the
  generated files carry the same licence.

Uses only the standard library, so it adds no dependencies.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = ROOT / "knowledge-base" / "wikipedia-sources.txt"
OUTPUT_DIR = ROOT / "knowledge-base" / "wikipedia"

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves with a contact point.
USER_AGENT = "PitchwiseKB/1.0 (https://github.com/someshktiwari/pitchwise)"
REQUEST_PAUSE_SECONDS = 1.0
MAX_DOC_CHARS = 20_000
LICENCE = "CC BY-SA 4.0, Wikipedia contributors. Adapted: reformatted into markdown and trimmed."

SKIP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources", "citations", "footnotes", "gallery",
    "explanatory notes", "notes and references", "references and notes",
    "works cited", "general references", "cited sources",
}

# Subjects the 17 curated documents already cover (D-014: never two versions
# of the same facts). Compared case-insensitively with the resolved title.
CURATED_SUBJECTS = {
    "don bradman", "virat kohli", "jasprit bumrah", "sachin tendulkar",
    "ben stokes", "kane williamson", "cricket world cup", "indian premier league",
    "icc men's t20 world cup", "the ashes", "international cricket council",
    "laws of cricket", "test cricket", "one day international",
    "twenty20 international", "first-class cricket", "history of cricket",
}

HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def read_sources(path=SOURCES_FILE):
    """Parse 'category: Title' lines; blank lines and # comments ignored."""
    sources = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, sep, title = line.partition(":")
        if not sep or not title.strip():
            raise ValueError(f"{path}:{n}: expected 'category: Title', got {line!r}")
        sources.append((category.strip(), title.strip()))
    return sources


def slugify(title):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "untitled"


def fetch_article(title, retries=3):
    """One article as plain text with == wiki == section markers, plus its
    resolved title, revision id and canonical URL. Returns None if the page
    is missing or a disambiguation page."""
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "extracts|info|revisions|pageprops", "explaintext": "1",
        "exsectionformat": "wiki", "inprop": "url", "rvprop": "ids|timestamp",
        "ppprop": "disambiguation", "titles": title,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(f"fetching {title!r} failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or "disambiguation" in pages[0].get("pageprops", {}):
        return None
    page = pages[0]
    revision = (page.get("revisions") or [{}])[0]
    return {
        "title": page["title"],
        "text": page.get("extract", ""),
        "revid": revision.get("revid"),
        "timestamp": revision.get("timestamp"),
        "url": page.get("fullurl"),
    }


def to_markdown(title, text, max_chars=MAX_DOC_CHARS):
    """Convert a plain-text extract into markdown sections that name their
    subject. Level-2 headings become '## Title: Heading'; level-3 become
    '## Title: Parent - Heading' (their own chunk, still anchored); deeper
    levels fold into their parent's text. The lead becomes '## Title:
    Overview'. Returns the markdown body (no front matter)."""
    sections = []  # [heading_text, [lines]]
    current = [f"{title}: Overview", []]
    parent = None
    skipping = False
    for raw in text.splitlines():
        m = HEADING.match(raw.strip())
        if m:
            level, name = len(m.group(1)), m.group(2).strip()
            if level == 2:
                skipping = name.lower() in SKIP_SECTIONS
                parent = name
                if not skipping:
                    sections.append(current)
                    current = [f"{title}: {name}", []]
            elif level == 3 and not skipping:
                sections.append(current)
                current = [f"{title}: {parent} - {name}" if parent else f"{title}: {name}", []]
            elif not skipping:
                current[1].append(f"**{name}.**")
            continue
        if skipping:
            continue
        line = raw.strip()
        if "{\\displaystyle" in line:  # leftover maths markup, not readable text
            continue
        current[1].append(line)
    sections.append(current)

    parts, used = [f"# {title}\n"], len(title) + 3
    for heading, lines in sections:
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if not body:
            continue  # drop empty sections
        block = f"## {heading}\n\n{body}\n"
        if used + len(block) > max_chars and len(parts) > 1:
            break  # cap at a section boundary
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def front_matter(article, retrieved):
    source = article["url"]
    if article.get("revid"):
        source = (f"https://en.wikipedia.org/w/index.php?title="
                  f"{urllib.parse.quote(article['title'].replace(' ', '_'))}&oldid={article['revid']}")
    lines = ["---",
             f"title: {article['title']}",
             f"source: {source}",
             f"revision_timestamp: {article.get('timestamp') or 'unknown'}",
             f"retrieved: {retrieved}",
             f"licence: {LICENCE}",
             "---", ""]
    return "\n".join(lines)


def build(sources, output_dir=OUTPUT_DIR, fetch=fetch_article, pause=REQUEST_PAUSE_SECONDS,
          retrieved=None):
    retrieved = retrieved or date.today().isoformat()
    written, skipped = [], []
    for i, (category, title) in enumerate(sources, 1):
        if title.lower() in CURATED_SUBJECTS:
            raise ValueError(f"{title!r} is already a curated document; remove it from the sources")
        article = fetch(title)
        if article is None:
            skipped.append((title, "missing or disambiguation page"))
            print(f"[{i}/{len(sources)}] SKIP {title}: missing or disambiguation page")
            continue
        if article["title"].lower() in CURATED_SUBJECTS:
            raise ValueError(f"{title!r} resolves to curated subject {article['title']!r}")
        body = to_markdown(article["title"], article["text"])
        if body.count("\n## ") < 1:
            skipped.append((title, "no usable text"))
            print(f"[{i}/{len(sources)}] SKIP {title}: no usable text")
            continue
        path = Path(output_dir) / category / f"{slugify(article['title'])}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(front_matter(article, retrieved) + body, encoding="utf-8")
        written.append(path)
        print(f"[{i}/{len(sources)}] {category}/{path.name}: {len(body):,} chars")
        if pause:
            time.sleep(pause)
    return written, skipped


def main():
    parser = argparse.ArgumentParser(description="Build the Wikipedia part of the knowledge base.")
    parser.add_argument("--only", action="append", help="fetch only this title (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="check the sources file, fetch nothing")
    args = parser.parse_args()

    sources = read_sources()
    if args.only:
        wanted = {t.lower() for t in args.only}
        sources = [(c, t) for c, t in sources if t.lower() in wanted]
    overlap = [t for _, t in sources if t.lower() in CURATED_SUBJECTS]
    if overlap:
        sys.exit(f"Already curated, remove from {SOURCES_FILE.name}: {overlap}")
    if args.dry_run:
        by_cat = {}
        for c, _ in sources:
            by_cat[c] = by_cat.get(c, 0) + 1
        print(f"{len(sources)} articles: {by_cat}")
        return

    written, skipped = build(sources)
    print(f"\nWrote {len(written)} files under {OUTPUT_DIR.relative_to(ROOT)}; skipped {len(skipped)}.")
    for title, reason in skipped:
        print(f"  skipped {title}: {reason}")


if __name__ == "__main__":
    main()
