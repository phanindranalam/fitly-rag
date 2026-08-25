"""Cut a small, committable sample out of the full corpus.

WHY THIS EXISTS
---------------
`data/corpus.jsonl` is gitignored and `data/chroma/` is ~107 MB, so anyone who
clones this repo has a working codebase and nothing to run it against. Building
the real corpus needs a clone of the Fitly repo beside this one, which is a lot
to ask of someone who just wants to see whether the thing works.

So: one posting per employer, up to N employers. Round-robin rather than the
first N lines, because the corpus is written employer-by-employer and the first
40 lines are four companies. A sample that is four companies wide would make
the boilerplate statistics and the geography filter look broken in a way the
full corpus is not.

Small enough to commit, wide enough that retrieval, refusal, hybrid fusion and
the location filter all behave qualitatively like they do on the full 874.

    python make_sample.py                    # 40 postings, one per company
    python make_sample.py --n 60
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/corpus.jsonl")
    ap.add_argument("--out", default="data/corpus.sample.jsonl")
    ap.add_argument("--n", type=int, default=40, help="how many postings to keep")
    ap.add_argument("--per-company", type=int, default=1)
    args = ap.parse_args()

    by_company: dict[str, list[dict]] = defaultdict(list)
    total = 0
    with open(args.src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            by_company[doc.get("company") or "unknown"].append(doc)
            total += 1

    # Widest employers first: they are the ones whose repeated EEO and benefits
    # text makes the boilerplate detector do anything at all.
    order = sorted(by_company, key=lambda c: -len(by_company[c]))

    picked: list[dict] = []
    for slot in range(args.per_company):
        for company in order:
            if len(picked) >= args.n:
                break
            docs = by_company[company]
            if slot < len(docs):
                picked.append(docs[slot])
        if len(picked) >= args.n:
            break

    with open(args.out, "w", encoding="utf-8") as fh:
        for doc in picked:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    companies = len({d.get("company") for d in picked})
    print(f"source : {total} postings, {len(by_company)} companies")
    print(f"sample : {len(picked)} postings, {companies} companies -> {args.out}")


if __name__ == "__main__":
    main()
