#!/usr/bin/env python3
"""Does the corpus actually contain answers to the eval's questions?

    python eval/check_coverage.py

WHY RUN THIS BEFORE THE EVAL
----------------------------
The eval labels twelve questions "answerable". That label is a claim about
the CORPUS, not about the retriever -- and it was written before the corpus
existed, against an imagined set of postings.

If the corpus turns out to contain no security-clearance roles, then q05
scores as a retrieval failure when nothing is wrong with retrieval at all.
The pipeline would be marked down for correctly failing to find something
that isn't there, and the reported term-hit rate would be measuring corpus
coverage while claiming to measure recall.

So: check first. This greps the raw corpus for each question's expected
terms and reports how many postings contain them. Then either fix the
question or keep it and say why.

WHAT TO DO WITH THE OUTPUT
--------------------------
  0 postings        The question is unanswerable against THIS corpus. Either
                    swap the question for one the corpus covers, or move it
                    to should_refuse -- a question the corpus cannot answer
                    is, by this app's definition, one it should refuse.
  1-3 postings      Thin. It stays answerable, but a miss tells you about
                    that one posting, not about retrieval in general.
  many postings     Good. A genuine test of retrieval.

Moving a question to should_refuse is not gaming the eval, PROVIDED you say
so in the writeup: "q06 asked about HIPAA; the corpus contains no healthcare
postings, so it was relabelled as a refusal test." That is a documented
property of the corpus. Silently deleting the question would be the dishonest
version.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from run_eval import load_questions, term_in  # noqa: E402


def load_corpus() -> list[dict]:
    if not os.path.exists(config.CORPUS_PATH):
        raise SystemExit(f"{config.CORPUS_PATH} not found. Run build_corpus.py first.")
    with open(config.CORPUS_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    docs = load_corpus()
    blobs = [(d.get("company", ""), d["text"].lower()) for d in docs]
    print(f"{len(docs)} postings from {len({b[0] for b in blobs})} companies\n")

    questions = load_questions()
    thin, missing = [], []

    for q in questions:
        terms = q.get("expect_terms") or []
        if not terms:
            continue  # unanswerable by design; nothing to check
        per_term = {}
        hit_docs = set()
        for t in terms:
            n = 0
            for i, (_, blob) in enumerate(blobs):
                if term_in(t, blob):
                    n += 1
                    hit_docs.add(i)
            per_term[t] = n

        total = len(hit_docs)
        companies = len({blobs[i][0] for i in hit_docs})
        if total == 0:
            flag, bucket = "NOT IN CORPUS", missing
            bucket.append(q)
        elif total <= 3:
            flag, bucket = "THIN", thin
            bucket.append(q)
        else:
            flag = "ok"

        print(f"[{flag:13}] {q['id']} {q['category']:11} "
              f"{total:4} posting(s), {companies:3} compan(y/ies)")
        print(f"                {q['question'][:74]}")
        detail = ", ".join(f"{t}={n}" for t, n in sorted(per_term.items(), key=lambda x: -x[1]))
        print(f"                {detail}\n")

    print("-" * 72)
    if missing:
        print(f"\n{len(missing)} question(s) have NO supporting postings. "
              f"These will score as retrieval failures for a reason that has "
              f"nothing to do with retrieval:")
        for q in missing:
            print(f"  {q['id']}: {q['question']}")
        print("\n  Fix: edit eval/questions.yaml -- either replace the question with "
              "one\n  this corpus covers, or set should_refuse: true and add a `why:` "
              "line\n  explaining that the corpus has no such postings. Then say so in "
              "the\n  writeup. Do not delete them silently.")
    if thin:
        print(f"\n{len(thin)} question(s) are supported by 3 or fewer postings. "
              f"Keep them,\nbut a miss on these says more about one posting than "
              f"about retrieval:")
        for q in thin:
            print(f"  {q['id']}: {q['question']}")
    if not missing and not thin:
        print("\nEvery answerable question has real support in the corpus. "
              "Run the eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
