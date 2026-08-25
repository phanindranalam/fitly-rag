#!/usr/bin/env python3
"""Measure the pipeline instead of asserting things about it.

    python eval/run_eval.py --retrieval            # matrix, no LLM calls, fast
    python eval/run_eval.py --sweep-threshold      # where to set MIN_SIM
    python eval/run_eval.py --generate             # full run with the judge
    python eval/run_eval.py --generate --config section/hybrid/rerank

WHAT THIS MEASURES AND WHY EACH NUMBER IS HERE
----------------------------------------------
Every design choice in this project was argued for in a docstring. Arguments
are free. This file is where they get checked, and it is built so that a
result contradicting the argument is visible rather than buried.

Three separable stages, three separable measurements:

RETRIEVAL (no LLM, so it is cheap enough to run over the whole matrix)
  term_hit@k   Did the retrieved context contain the vocabulary the question
               needs? A proxy for recall -- see the note in questions.yaml
               about why this set is not labelled with chunk ids.
  first_hit    The rank of the first useful chunk. term_hit says the answer
               was in the context somewhere; first_hit says whether it was
               at position 1 or position 5. This is the number the reranker
               is supposed to move, and if it does not move it, the
               reranker gets deleted.
  top_sim      Similarity of the best chunk, split by answerable vs
               unanswerable. If those two distributions do not separate,
               no threshold can work and the refusal guard is theatre.

THRESHOLD (the sweep)
  MIN_SIM is a number someone has to pick. Picking it by eye is guessing.
  The sweep walks it across the range and reports precision and recall of
  the refusal decision at each step, so the config value in the writeup has
  a reason attached.

GENERATION (costs an API call per question, so it runs on one config)
  faithful     An LLM judge reads ONLY the context and the answer and counts
               claims the context does not support. This is the headline
               number: it is the direct measurement of whether the app makes
               things up.
  citations    Fraction of answers that cite at least one source, plus the
               count of dangling [n] indices that point at nothing.
  refusal      Did it refuse exactly the questions it should have? Both
               error directions are reported, because they are not equally
               bad: answering an unanswerable question is a hallucination,
               refusing an answerable one is merely unhelpful.

ON USING AN LLM AS THE JUDGE
----------------------------
The judge is the same model family as the generator, which is a real
weakness: a model is a soft grader of its own output, so the faithfulness
number is optimistic. It is used anyway because the alternative at this
scale is hand-grading 18 answers per config, and a slightly optimistic
number computed the same way across every config still ranks the configs
correctly. The judge sees only the context and the answer, never the
question's label, so it cannot cheat by knowing which questions were traps.
This limitation belongs in the writeup, not hidden in a footnote.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from generate import count_bad_citations, extract_citations  # noqa: E402
from graph import ask  # noqa: E402
from retrieve import get_retriever  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(HERE, "questions.yaml")

STRATEGIES = ("fixed", "section")
MODES = ("dense", "hybrid")
RERANKS = (False, True)


def load_questions(path: str = QUESTIONS_PATH) -> list[dict]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["questions"]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Term matching
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT A SUBSTRING TEST
# -------------------------------
# The first version asked `term in text`. On a real corpus that reported the
# HIPAA question as supported by 99 postings, because "phi" is inside
# "sophisticated", "graphics" and "Philadelphia". The real number was 2. It
# also scored "go" against "going", "goal" and "algorithm", and "secret"
# against "secrets management" -- a DevOps phrase with no relation to security
# clearances.
#
# A metric that cannot fail is not a metric. Short terms need word boundaries,
# and the terms that carry the most signal in this domain are exactly the
# short ones.
#
# Terms containing punctuation ("ts/sci", "node.js", "$", "c++") are matched
# as plain substrings, because \b does not do what you want next to a symbol:
# \bc++\b never matches. The tradeoff is deliberate and the punctuated terms
# are specific enough that false positives are unlikely.

import re as _re

_WORDY = _re.compile(r"^[a-z0-9]+(?: [a-z0-9]+)*$")
_pattern_cache: dict = {}


def term_matcher(term: str):
    """Compiled matcher for one expected term, or None to use substring."""
    t = term.lower().strip()
    if t in _pattern_cache:
        return _pattern_cache[t]
    pat = None
    if _WORDY.match(t):
        pat = _re.compile(r"\b" + _re.escape(t) + r"\b")
    _pattern_cache[t] = pat
    return pat


def term_in(term: str, blob: str) -> bool:
    pat = term_matcher(term)
    return bool(pat.search(blob)) if pat else term.lower() in blob


def _term_hit_rank(hits, terms: list[str]) -> int | None:
    """1-based rank of the first chunk containing any expected term."""
    if not terms:
        return None
    for i, h in enumerate(hits, start=1):
        blob = h.text.lower()
        if any(term_in(t, blob) for t in terms):
            return i
    return None


def eval_retrieval(questions: list[dict], strategy: str, mode: str,
                   use_rerank: bool) -> dict:
    retriever = get_retriever(strategy)
    answerable = [q for q in questions if not q.get("should_refuse")]
    unanswerable = [q for q in questions if q.get("should_refuse")]

    hits_found, first_ranks, latencies = 0, [], []
    sims_answerable, sims_unanswerable = [], []
    # Split the refusal cases by WHY they are unanswerable. The first eval run
    # lumped them together and concluded "the threshold separates nothing" --
    # true on average, and it hid the question that matters: does similarity
    # distinguish off-DOMAIN questions (sourdough, brake pads) even though it
    # cannot distinguish off-CONTENT ones (revenue, hiring manager)? Those are
    # different failure modes and averaging them answers neither.
    sims_off_domain = []
    per_question = []

    for q in questions:
        t0 = time.time()
        res = retriever.retrieve(q["question"], use_rerank=use_rerank, mode=mode)
        latencies.append(time.time() - t0)

        rank = _term_hit_rank(res.hits, q.get("expect_terms") or [])
        if q.get("should_refuse"):
            sims_unanswerable.append(res.top_sim)
            if q["category"] == "off_domain":
                sims_off_domain.append(res.top_sim)
        else:
            sims_answerable.append(res.top_sim)
            if rank:
                hits_found += 1
                first_ranks.append(rank)

        per_question.append({
            "id": q["id"], "category": q["category"],
            "should_refuse": bool(q.get("should_refuse")),
            "top_sim": round(res.top_sim, 4),
            "first_hit_rank": rank,
            "n_hits": len(res.hits),
        })

    n_ans = max(len(answerable), 1)
    return {
        "config": f"{strategy}/{mode}/{'rerank' if use_rerank else 'norerank'}",
        "strategy": strategy, "mode": mode, "rerank": use_rerank,
        "term_hit_rate": hits_found / n_ans,
        "mean_first_hit": round(statistics.mean(first_ranks), 2) if first_ranks else None,
        "mrr": round(sum(1 / r for r in first_ranks) / n_ans, 4),
        "mean_sim_answerable": round(statistics.mean(sims_answerable), 4) if sims_answerable else 0,
        "mean_sim_unanswerable": round(statistics.mean(sims_unanswerable), 4) if sims_unanswerable else 0,
        "separation": round((statistics.mean(sims_answerable) if sims_answerable else 0)
                            - (statistics.mean(sims_unanswerable) if sims_unanswerable else 0), 4),
        "mean_sim_off_domain": round(statistics.mean(sims_off_domain), 4) if sims_off_domain else None,
        "separation_off_domain": (
            round((statistics.mean(sims_answerable) if sims_answerable else 0)
                  - statistics.mean(sims_off_domain), 4) if sims_off_domain else None),
        "p50_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
        "n_unanswerable": len(unanswerable),
        "per_question": per_question,
    }


def retrieval_matrix(questions: list[dict]) -> list[dict]:
    rows = []
    for strategy in STRATEGIES:
        for mode in MODES:
            for rr in RERANKS:
                label = f"{strategy}/{mode}/{'rerank' if rr else 'norerank'}"
                print(f"  {label:34}", end="", flush=True)
                row = eval_retrieval(questions, strategy, mode, rr)
                print(f" term_hit={row['term_hit_rate']:.0%} "
                      f"mrr={row['mrr']:.3f} "
                      f"sep={row['separation']:+.3f} "
                      f"p95={row['p95_latency_s']}s")
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_threshold(row: dict) -> list[dict]:
    """Walk MIN_SIM and report the refusal decision quality at each value.

    Only the retrieval-side guard is being tuned here. The model-side guard
    catches what this one lets through, which is why a threshold that lets a
    few unanswerable questions past is acceptable and a threshold that
    blocks answerable ones is not: there is a second net downstream, but
    nothing recovers a question the retriever refused.
    """
    out = []
    for step in range(0, 21):
        thr = step / 20.0
        tp = fp = tn = fn = 0
        for q in row["per_question"]:
            refuses = q["top_sim"] < thr
            if q["should_refuse"]:
                tp += refuses          # correctly refused
                fn += not refuses      # HALLUCINATION RISK: answered a trap
            else:
                fp += refuses          # unhelpful: refused a real question
                tn += not refuses
        out.append({
            "threshold": thr,
            "correct_refusals": tp,
            "missed_refusals": fn,
            "wrong_refusals": fp,
            "correct_answers": tn,
            "accuracy": round((tp + tn) / max(len(row["per_question"]), 1), 3),
        })
    return out


def print_sweep(rows: list[dict]) -> None:
    print(f"\n{'MIN_SIM':>8} {'refused ok':>11} {'MISSED':>7} {'over-refused':>13} "
          f"{'answered ok':>12} {'acc':>6}")
    best = max(rows, key=lambda r: (r["accuracy"], -r["missed_refusals"]))
    baseline = max(rows, key=lambda r: r["correct_answers"] + r["correct_refusals"] * 0)
    for r in rows:
        star = " <-- best" if r is best else ""
        print(f"{r['threshold']:>8.2f} {r['correct_refusals']:>11} {r['missed_refusals']:>7} "
              f"{r['wrong_refusals']:>13} {r['correct_answers']:>12} {r['accuracy']:>6.2f}{star}")
    # A "best" threshold of 0.00 means the sweep found nothing: refusing
    # nothing scored as well as any cutoff. Recommending it as a tuned value
    # would dress up a null result as a decision.
    if best["threshold"] <= 0.0 or best["correct_refusals"] == 0:
        print("\nNO THRESHOLD WORKS on this set. The best score comes from refusing")
        print("nothing at all, which means the similarity signal carries no usable")
        print("information about answerability here.")
        print("\nDo NOT read this as 'set MIN_SIM=0'. It means guard 1 (retrieval-side")
        print("refusal) cannot catch these questions, and guard 2 (the model returning")
        print("INSUFFICIENT_CONTEXT) is doing all the work. Keep MIN_SIM low enough not")
        print("to over-refuse, and report this result rather than hiding it.")
        print("\nCheck separation_off_domain in the matrix before concluding guard 1 is")
        print("useless: it may still catch genuinely off-domain queries even when it")
        print("cannot catch on-topic ones that the corpus happens not to answer.")
    else:
        print(f"\nSet MIN_SIM={best['threshold']:.2f} in .env "
              f"(accuracy {best['accuracy']:.0%}, {best['missed_refusals']} trap(s) still "
              f"reaching the model -- guard 2 has to catch those).")


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading a generated answer for FAITHFULNESS to its source material.

You will see CONTEXT (numbered source passages) and an ANSWER. Judge only whether every factual claim in the ANSWER is supported by the CONTEXT. You are not judging whether the answer is good, well written, or complete.

Rules:
- A claim is unsupported if the context does not state it, even if it is true in the real world.
- Reasonable paraphrase of the context is supported. Added specifics (numbers, names, requirements not in the context) are not.
- General framing that makes no factual claim ("here are a few options") is not a claim.
- If the ANSWER is a refusal, there are zero claims and zero unsupported claims.

Reply with ONLY a JSON object, no other text:
{"total_claims": <int>, "unsupported_claims": <int>, "worst_example": "<the single least supported sentence, or empty string>"}"""


def judge_faithfulness(context: str, answer_text: str) -> dict:
    from generate import call_llm

    user = f"CONTEXT:\n\n{context}\n\n---\n\nANSWER:\n\n{answer_text}"
    try:
        # Deliberately a different model family from the generator. See the
        # note on config.JUDGE_MODEL: a model grading its own output is not a
        # measurement, it is an opinion with a percentage sign.
        raw, _, _ = call_llm(JUDGE_PROMPT, user, model=config.JUDGE_MODEL)
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1])
    except Exception as exc:
        # A judge failure must not be scored as a pass. Marked unknown and
        # excluded from the mean, so a flaky API cannot inflate the result.
        return {"total_claims": 0, "unsupported_claims": 0,
                "worst_example": "", "judge_error": str(exc)}


def eval_generation(questions: list[dict], strategy: str, mode: str,
                    use_rerank: bool, judge: bool = True) -> dict:
    from generate import format_context

    rows = []
    for q in questions:
        st = ask(q["question"], strategy=strategy, use_rerank=use_rerank, mode=mode)
        ans = st["answer"]
        should = bool(q.get("should_refuse"))

        row = {
            "id": q["id"], "category": q["category"],
            "question": q["question"],
            "should_refuse": should,
            "refused": ans.refused,
            "correct": ans.refused == should,
            "widened": bool(st.get("widened")),
            "n_citations": len(ans.citations),
            "bad_citations": count_bad_citations(ans.text, ans.hits),
            "latency_s": round(st["elapsed_s"], 2),
            # Guard 3 bookkeeping, so the verification node's effect is a
            # measured number and not a claim that it helps.
            "verified": bool(ans.verified),
            "claims_total": ans.claims_total,
            "claims_unsupported": ans.claims_unsupported,
            "overturned": ans.refused and "guard 3" in (ans.reason or ""),
            "prompt_tokens": ans.prompt_tokens,
            "completion_tokens": ans.completion_tokens,
            "answer": ans.text,
            "trace": st.get("trace", []),
        }

        if judge and not ans.refused and ans.hits:
            verdict = judge_faithfulness(format_context(ans.hits), ans.text)
            row["total_claims"] = verdict.get("total_claims", 0)
            row["unsupported_claims"] = verdict.get("unsupported_claims", 0)
            row["worst_example"] = verdict.get("worst_example", "")
            if "judge_error" in verdict:
                row["judge_error"] = verdict["judge_error"]

        mark = "ok " if row["correct"] else "BAD"
        state = "refused" if ans.refused else "answered"
        if row["overturned"]:
            state = "G3-stop"
        print(f"  [{mark}] {q['id']} {q['category']:13} "
              f"{state:8} "
              f"cites={row['n_citations']} "
              f"{row['latency_s']:>5.2f}s  {q['question'][:52]}")
        rows.append(row)

    answered = [r for r in rows if not r["refused"]]
    judged = [r for r in answered if "total_claims" in r and "judge_error" not in r]
    total_claims = sum(r["total_claims"] for r in judged)
    unsupported = sum(r["unsupported_claims"] for r in judged)

    should_refuse = [r for r in rows if r["should_refuse"]]
    should_answer = [r for r in rows if not r["should_refuse"]]

    return {
        "config": f"{strategy}/{mode}/{'rerank' if use_rerank else 'norerank'}",
        "n_questions": len(rows),
        "refusal_accuracy": round(sum(r["correct"] for r in rows) / max(len(rows), 1), 3),
        "missed_refusals": sum(1 for r in should_refuse if not r["refused"]),
        "over_refusals": sum(1 for r in should_answer if r["refused"]),
        "faithfulness": round(1 - unsupported / total_claims, 3) if total_claims else None,
        "total_claims_judged": total_claims,
        "unsupported_claims": unsupported,
        "judge_failures": sum(1 for r in answered if "judge_error" in r),
        "answers_with_citations": sum(1 for r in answered if r["n_citations"] > 0),
        "dangling_citations": sum(r["bad_citations"] for r in rows),
        "widen_rescues": sum(1 for r in rows if r["widened"] and not r["refused"]),
        "guard3_ran": sum(1 for r in rows if r.get("verified")),
        "guard3_overturned": sum(1 for r in rows if r.get("overturned")),
        "guard3_correct": sum(1 for r in rows if r.get("overturned") and r["should_refuse"]),
        "guard3_wrong": sum(1 for r in rows if r.get("overturned") and not r["should_refuse"]),
        "p50_latency_s": round(statistics.median([r["latency_s"] for r in rows]), 2),
        "p95_latency_s": round(sorted(r["latency_s"] for r in rows)[int(len(rows) * 0.95) - 1], 2),
        "mean_prompt_tokens": int(statistics.mean([r["prompt_tokens"] for r in answered])) if answered else 0,
        "per_question": rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def markdown_retrieval_table(rows: list[dict]) -> str:
    """Two separation columns, not one.

    "off-content" questions are about jobs but ask for something the postings
    do not contain (revenue, hiring manager). "off-domain" questions are not
    about jobs at all. Similarity behaves completely differently on the two,
    and a single averaged separation number hides that -- which is exactly
    what the first eval run did.
    """
    def fmt(v, spec=".3f"):
        return "-" if v is None else format(v, spec)

    header = ("| config | term hit@5 | MRR | mean first hit | sim answerable | "
              "sim off-content | sim off-domain | sep off-content | sep off-domain | p95 |")
    out = [header, "|---" * 10 + "|"]
    for r in sorted(rows, key=lambda r: -r["mrr"]):
        out.append(
            f"| `{r['config']}` "
            f"| {r['term_hit_rate']:.0%} "
            f"| {r['mrr']:.3f} "
            f"| {fmt(r['mean_first_hit'], '.2f')} "
            f"| {fmt(r['mean_sim_answerable'])} "
            f"| {fmt(r['mean_sim_unanswerable'])} "
            f"| {fmt(r.get('mean_sim_off_domain'))} "
            f"| {fmt(r['separation'], '+.3f')} "
            f"| {fmt(r.get('separation_off_domain'), '+.3f')} "
            f"| {r['p95_latency_s']}s |")
    return "\n".join(out)


def markdown_generation_table(r: dict) -> str:
    faith = f"{r['faithfulness']:.1%}" if r["faithfulness"] is not None else "n/a"
    return "\n".join([
        f"| metric | value |", "|---|---|",
        f"| config | `{r['config']}` |",
        f"| refusal accuracy | {r['refusal_accuracy']:.0%} ({r['n_questions']} questions) |",
        f"| missed refusals (hallucination risk) | {r['missed_refusals']} |",
        f"| over-refusals (unhelpful) | {r['over_refusals']} |",
        f"| faithfulness | {faith} ({r['unsupported_claims']}/{r['total_claims_judged']} claims unsupported) |",
        f"| answers carrying a citation | {r['answers_with_citations']} |",
        f"| dangling citations | {r['dangling_citations']} |",
        f"| questions rescued by the widen node | {r['widen_rescues']} |",
        f"| guard 3 ran on | {r['guard3_ran']} answer(s) |",
        f"| guard 3 overturned | {r['guard3_overturned']} "
        f"({r['guard3_correct']} correctly, {r['guard3_wrong']} wrongly) |",
        f"| p50 / p95 latency | {r['p50_latency_s']}s / {r['p95_latency_s']}s (budget {config.LATENCY_BUDGET_S}s) |",
        f"| mean prompt tokens | {r['mean_prompt_tokens']:,} |",
        f"| judge model | `{config.JUDGE_MODEL}` "
        f"{'(independent of generator)' if config.JUDGE_MODEL != config.NEBIUS_MODEL else '**(SAME AS GENERATOR — self-graded)**'} |",
    ])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrieval", action="store_true", help="Retrieval matrix only. No API calls.")
    p.add_argument("--sweep-threshold", action="store_true", help="Tune MIN_SIM against the labels.")
    p.add_argument("--generate", action="store_true", help="Full run including generation and the judge.")
    p.add_argument("--no-judge", action="store_true", help="Skip the faithfulness judge (halves the API calls).")
    p.add_argument("--config", default=None,
                   help="strategy/mode/rerank for --generate, e.g. section/hybrid/rerank. "
                        "Default: the best config from the retrieval matrix.")
    p.add_argument("--out", default="eval/results")
    args = p.parse_args()

    if not any([args.retrieval, args.sweep_threshold, args.generate]):
        args.retrieval = args.sweep_threshold = True

    questions = load_questions()
    print(config.describe())
    print(f"{len(questions)} questions "
          f"({sum(1 for q in questions if q.get('should_refuse'))} unanswerable by design)\n")

    os.makedirs(args.out, exist_ok=True)
    report: dict = {}

    matrix = None
    if args.retrieval or args.sweep_threshold or (args.generate and not args.config):
        print("RETRIEVAL MATRIX (2 chunking strategies x dense/hybrid x rerank on/off)")
        matrix = retrieval_matrix(questions)
        report["retrieval"] = matrix
        print("\n" + markdown_retrieval_table(matrix))

    if args.sweep_threshold and matrix:
        best_row = max(matrix, key=lambda r: r["separation"])
        print(f"\nTHRESHOLD SWEEP on the config with the cleanest separation "
              f"(`{best_row['config']}`)")
        sweep = sweep_threshold(best_row)
        report["threshold_sweep"] = {"config": best_row["config"], "rows": sweep}
        print_sweep(sweep)

    if args.generate:
        if args.config:
            strategy, mode, rr = args.config.split("/")
            use_rerank = rr == "rerank"
        else:
            best = max(matrix, key=lambda r: (r["mrr"], r["separation"]))
            strategy, mode, use_rerank = best["strategy"], best["mode"], best["rerank"]
            print(f"\nBest retrieval config: {best['config']}. Generating with it.")
        print(f"\nGENERATION ({strategy}/{mode}/{'rerank' if use_rerank else 'norerank'})")
        gen = eval_generation(questions, strategy, mode, use_rerank, judge=not args.no_judge)
        report["generation"] = gen
        print("\n" + markdown_generation_table(gen))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.out, f"eval-{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    md_path = os.path.join(args.out, f"eval-{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Evaluation run {stamp}\n\n`{config.describe()}`\n\n")
        if "retrieval" in report:
            fh.write("## Retrieval matrix\n\n" + markdown_retrieval_table(report["retrieval"]) + "\n\n")
        if "generation" in report:
            fh.write("## Generation\n\n" + markdown_generation_table(report["generation"]) + "\n\n")
            fh.write("## Every answer\n\n")
            for r in report["generation"]["per_question"]:
                fh.write(f"### {r['id']} ({r['category']})\n\n"
                         f"**{r['question']}**\n\n"
                         f"- expected: {'refuse' if r['should_refuse'] else 'answer'}; "
                         f"got: {'refused' if r['refused'] else 'answered'} "
                         f"{'OK' if r['correct'] else '**WRONG**'}\n"
                         f"- citations: {r['n_citations']}, latency {r['latency_s']}s\n\n"
                         f"> {r['answer'][:800]}\n\n"
                         "```\n" + "\n".join(r["trace"]) + "\n```\n\n")

    print(f"\nWrote {json_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
