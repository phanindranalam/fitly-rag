#!/usr/bin/env python3
"""Build the RAG corpus: fetch job postings, clean them, write JSONL.

    python build_corpus.py --fitly-path ../fitly --out data/corpus.jsonl

WHY A SEPARATE CORPUS STEP
--------------------------
Fitly fetches postings live on every search and never stores them, which is
right for a job search (a stale posting is worse than a missing one) and
wrong for RAG (you cannot chunk, embed and evaluate against a corpus that
changes under you). So this script snapshots the corpus once, and everything
downstream -- both chunking strategies, both indexes, the eval set -- runs
against the same frozen file. Reproducibility is the whole point: a chunking
comparison against a moving corpus measures nothing.

THE BOILERPLATE PROBLEM, AND WHY IT IS THE INTERESTING PART
-----------------------------------------------------------
Job postings are maybe 40% boilerplate. Every one ends with an EEO
statement, a benefits blurb, an "about us" paragraph and a pay-transparency
notice, and those blocks are near-identical across thousands of documents.
Chunk naively and they become the densest region of your embedding space:
ask "what does this role require?" and you retrieve five copies of "we are
an equal opportunity employer" because that text appears in every document
and therefore sits close to everything.

Rather than maintain a regex list of banned phrases -- brittle, and it
encodes my guesses about what boilerplate looks like -- this detects it from
the data: hash every paragraph across the whole corpus, and drop the ones
that appear in more than BOILERPLATE_THRESHOLD of documents. A paragraph
repeated in 30% of postings by different employers is definitionally not
about any particular job.

That is measurable, which matters for the writeup: the script reports how
many paragraphs it dropped and shows you the top offenders, so the effect is
a number rather than an assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import os
import re
import sys
from collections import Counter

import vendor_geo as geo

# Paragraphs appearing in more than this share of documents are treated as
# boilerplate. 0.15 is deliberately aggressive: with a few thousand postings
# from ~140 different employers, genuine role content essentially never
# repeats across 15% of documents, while EEO and benefits language does.
BOILERPLATE_THRESHOLD = 0.15

# A paragraph shorter than this is a heading or a fragment. Frequency
# filtering would flag "Requirements:" as boilerplate, which is true but
# useless -- we want the headings for section-aware chunking later.
MIN_BOILERPLATE_LENGTH = 120

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    """Greenhouse returns HTML-escaped HTML; Ashby and Lever return either
    plain text or HTML depending on the customer. Normalize all of it to
    plain text with paragraph breaks preserved, because paragraph structure
    is what both the boilerplate detector and the section chunker rely on.
    """
    if not raw:
        return ""
    text = html_lib.unescape(raw)
    # Block-level tags become paragraph breaks before tags are stripped,
    # otherwise the whole posting collapses into one run-on line and every
    # downstream boundary is lost.
    text = re.sub(r"(?i)<\s*(br|/p|/div|/li|/h[1-6]|/tr)\s*/?>", "\n", text)
    text = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", text)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)  # entities can survive one pass
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _fingerprint(paragraph: str) -> str:
    """Hash of the normalized paragraph. Normalized because the same EEO
    statement appears with different company names and whitespace; comparing
    raw text would treat each variant as unique and catch none of them."""
    norm = re.sub(r"[^a-z0-9 ]", "", paragraph.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def find_boilerplate(docs: list[dict], threshold: float = BOILERPLATE_THRESHOLD):
    """Find repeated, non-informative paragraphs. Returns a BoilerplateReport.

    WHY THIS COUNTS PER COMPANY AND NOT ACROSS THE WHOLE CORPUS
    -----------------------------------------------------------
    The first version of this function counted paragraph frequency globally
    and dropped anything appearing in more than 15% of all documents. Run
    against a real corpus it found exactly zero paragraphs, and the reason is
    worth keeping in the code because it is a nice lesson in checking your
    unit of analysis.

    Boilerplate in this corpus is written PER EMPLOYER. Stripe's EEO
    paragraph says "Stripe is an equal opportunity employer"; Airbnb's says
    Airbnb. Different text, different fingerprint. So across 93 employers,
    the most any single fingerprint can reach is the number of postings that
    ONE company has open -- which is nowhere near 15% of the whole corpus.
    The global cutoff was 312 documents when the ceiling was about 25. Zero
    was arithmetically guaranteed, and the "aggressive threshold" comment
    above it was confidently wrong.

    The right unit is the company. A paragraph appearing in most of ONE
    employer's postings is that employer's boilerplate, and that is true
    whether they have 8 openings or 800.

    Both passes are kept, because both kinds exist:
      per-company  the EEO / benefits / about-us blocks, the vast majority
      global       text genuinely identical across employers (E-Verify
                   notices, identical third-party benefits vendor copy)
    """
    from collections import defaultdict

    global_counts: Counter[str] = Counter()
    company_counts: dict[str, Counter[str]] = defaultdict(Counter)
    company_docs: Counter[str] = Counter()
    fp_companies: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, str] = {}

    for doc in docs:
        company = doc.get("company") or "unknown"
        company_docs[company] += 1
        seen_in_this_doc = set()
        for para in paragraphs(doc["text"]):
            if len(para) < MIN_BOILERPLATE_LENGTH:
                continue
            fp = _fingerprint(para)
            if fp in seen_in_this_doc:
                continue  # count documents, not occurrences
            seen_in_this_doc.add(fp)
            global_counts[fp] += 1
            company_counts[company][fp] += 1
            fp_companies[fp].add(company)
            samples.setdefault(fp, para)

    # Pass 1: per company.
    per_company: dict[str, set[str]] = {}
    company_hits: list[tuple[str, str, int, int]] = []
    for company, counts in company_counts.items():
        n_docs = company_docs[company]
        if n_docs < 3:
            # With one or two postings, "appears in most of them" carries no
            # information -- a single job's own requirements would qualify.
            continue
        cutoff = max(2, int(round(n_docs * threshold)))
        hits = {fp for fp, n in counts.items() if n >= cutoff}
        if hits:
            per_company[company] = hits
            for fp in hits:
                company_hits.append((company, samples[fp], counts[fp], n_docs))

    # Pass 2: across companies -- counted in EMPLOYERS, not documents.
    #
    # Counting documents here is the same mistake as pass 1 in reverse: one
    # company with 800 openings would push its own boilerplate over any
    # document-based threshold and get it classified as cross-employer text,
    # and on a small corpus a document floor drops genuine content that two
    # or three postings happen to share. The question this pass asks is "do
    # unrelated employers publish this identical paragraph", so the unit is
    # the employer.
    n_companies = max(len(company_docs), 1)
    company_cutoff = max(3, int(round(n_companies * threshold)))
    global_drop = {fp for fp, cos in fp_companies.items()
                   if len(cos) >= company_cutoff}

    company_hits.sort(key=lambda x: -(x[2] / max(x[3], 1)))
    top_global = sorted(((samples[fp], len(fp_companies[fp])) for fp in global_drop),
                        key=lambda x: -x[1])

    # Diagnostics: the most-repeated paragraphs REGARDLESS of whether they
    # cleared a cutoff. If a future corpus finds nothing again, this shows
    # immediately whether the thresholds are wrong or the text simply never
    # repeats (which would mean paragraph splitting is broken upstream).
    most_repeated = [(samples[fp], n) for fp, n in global_counts.most_common(5)]

    return {
        "per_company": per_company,
        "global": global_drop,
        "company_hits": company_hits,
        "top_global": top_global,
        "most_repeated": most_repeated,
        "n_company_blocks": sum(len(v) for v in per_company.values()),
    }


def strip_boilerplate(text: str, company: str, report: dict) -> str:
    """Drop this company's boilerplate plus anything global."""
    drop = report["global"] | report["per_company"].get(company or "unknown", set())
    kept = [p for p in paragraphs(text) if _fingerprint(p) not in drop]
    return "\n\n".join(kept)


def load_postings(fitly_path: str, limit_per_board: int | None = None) -> list[dict]:
    """Reuse Fitly's ATS fetchers rather than reimplementing them.

    This is the only coupling between the two projects, and it is one-way:
    fitly-rag reads from fitly, never the reverse, so nothing here can break
    the deployed app.
    """
    sys.path.insert(0, os.path.abspath(fitly_path))
    import ats  # noqa: E402  (path must be set first)

    boards = ats.load_company_boards(os.path.join(fitly_path, "data", "company_boards.json"))
    if not boards:
        raise SystemExit(
            f"No boards found under {fitly_path}/data/company_boards.json. "
            "Run scripts/find_company_boards.py in the Fitly repo first."
        )

    print(f"Fetching {len(boards)} company boards...")
    results = ats.fetch_boards(boards)

    docs = []
    for board, records in zip(boards, results):
        if limit_per_board:
            records = records[:limit_per_board]
        for rec in records:
            text = html_to_text(rec.get("description") or "")
            if len(text) < 200:
                continue  # a posting with no body can't answer anything
            # Location is parsed ONCE, here, into clean filterable fields.
            # Doing it at index time instead would mean re-parsing on every
            # rebuild; doing it at query time would mean running a text
            # classifier inside the retrieval hot path. The raw string is
            # kept too, so a human can always check the parse.
            loc = geo.parse_location(rec.get("location") or "", description=text[:4000])
            docs.append({
                "id": rec.get("source_id"),
                "company": rec.get("company"),
                "title": rec.get("title"),
                "location": rec.get("location"),
                "posted_date": rec.get("posted_date"),
                "url": rec.get("redirect_url"),
                "compensation": rec.get("compensation") or "",
                # Normalized geography, used as a PRE-filter at retrieval
                # time (see retrieve.py). Carrying Fitly's rule forward:
                # geography is a hard constraint, applied before ranking,
                # never a tiebreak after it.
                "geo_country": loc.country or "",
                "geo_state": loc.state or "",
                "geo_city": (loc.city or "").lower(),
                "work_mode": loc.work_mode,
                "remote_scope": loc.remote_scope,
                "geo_confidence": loc.location_confidence,
                "text": text,
            })
        print(f"  {board.company:28} {len(records):4} posting(s)")
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fitly-path", default="../fitly", help="Path to the Fitly repo (default: ../fitly)")
    parser.add_argument("--out", default="data/corpus.jsonl")
    parser.add_argument("--limit-per-board", type=int, default=None,
                        help="Cap postings per company. Useful for a fast first run.")
    parser.add_argument("--threshold", type=float, default=BOILERPLATE_THRESHOLD)
    parser.add_argument("--keep-boilerplate", action="store_true",
                        help="Skip boilerplate removal. Use to produce the 'before' corpus for the report.")
    args = parser.parse_args()

    docs = load_postings(args.fitly_path, args.limit_per_board)
    if not docs:
        raise SystemExit("No postings with usable text. Check the Fitly board config.")

    print(f"\n{len(docs)} postings with usable text.")

    dropped_paragraphs = 0
    if not args.keep_boilerplate:
        report = find_boilerplate(docs, args.threshold)
        chars_before = sum(len(d["text"]) for d in docs)
        for doc in docs:
            before = len(paragraphs(doc["text"]))
            doc["text"] = strip_boilerplate(doc["text"], doc.get("company"), report)
            dropped_paragraphs += before - len(paragraphs(doc["text"]))
        chars_after = sum(len(d["text"]) for d in docs)

        print(f"\nBoilerplate detected in two passes (threshold {args.threshold:.0%}):")
        print(f"  per-company: {report['n_company_blocks']} block(s) across "
              f"{len(report['per_company'])} employer(s)")
        print(f"  cross-company: {len(report['global'])} block(s)")
        print(f"Dropped {dropped_paragraphs} paragraph instances, "
              f"{chars_before - chars_after:,} characters "
              f"({100 * (chars_before - chars_after) / max(chars_before, 1):.1f}% of the corpus).")

        if report["company_hits"]:
            print("\nTop offenders (these would otherwise dominate retrieval):")
            for company, sample, n, of in report["company_hits"][:6]:
                print(f"  [{company}: {n}/{of} postings] {sample[:100]}...")
        if report["top_global"]:
            print("\nIdentical across different employers:")
            for sample, n in report["top_global"][:3]:
                print(f"  [{n} employers] {sample[:100]}...")

        if not dropped_paragraphs:
            print("\nNothing was dropped. Most-repeated paragraphs in the corpus, "
                  "for diagnosis:")
            for sample, n in report["most_repeated"]:
                print(f"  [{n} docs] {sample[:100]}...")

        # Postings that were nothing BUT boilerplate are now empty.
        before_count = len(docs)
        docs = [d for d in docs if len(d["text"]) >= 200]
        if before_count - len(docs):
            print(f"\nDropped {before_count - len(docs)} posting(s) that were entirely boilerplate.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    from collections import Counter as _C
    modes = _C(d["work_mode"] for d in docs)
    countries = _C(d["geo_country"] or "unknown" for d in docs)
    print("\nGeography parsed from each posting (used as a retrieval pre-filter):")
    print("  work mode:", dict(modes))
    print("  country:  ", dict(countries.most_common(6)))

    total_chars = sum(len(d["text"]) for d in docs)
    companies = len({d["company"] for d in docs})
    print(f"\nWrote {len(docs)} documents from {companies} companies to {args.out}")
    print(f"Corpus size: {total_chars:,} characters, "
          f"mean {total_chars // max(len(docs), 1):,} chars per posting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
