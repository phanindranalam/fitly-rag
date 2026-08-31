"""End-to-end test of the Streamlit app, headless.

WHY THIS EXISTS
---------------
`smoke_test.py` checks the pipeline. It does not check the UI, and the UI is
the demo. Every refactor in this project changed a field the app renders --
`Hit.score` split into `score` + `sim`, metadata gained the `geo_*` carry
fields, `Answer` gained the verification fields -- and none of those changes
would raise an error at import time. They would show up as a dash in a table,
or a location filter that silently returns nothing, live, on camera.

So this does three things, in increasing cost:

  PHASE 1  Boot app.py headlessly with Streamlit's own AppTest harness.
           Catches import errors, missing config, widget construction bugs.
           No API calls, a few seconds.

  PHASE 2  Run the exact demo questions through `graph.ask()` with the exact
           keyword arguments app.py passes, then assert on the fields
           `render_answer` reads. This is the part that catches "the panel
           shows dashes" and "guard 1 stopped firing".
           One API call per answered question.

  PHASE 3  The filter path, which has no visible failure mode. A location
           filter that matches nothing returns an empty result set that looks
           exactly like a corpus with no jobs in that country.

Run it before recording:

    python ui_test.py                      # phases 1-3
    python ui_test.py --fast               # phase 1 + 3 only, no API calls
    python ui_test.py --resume path.pdf    # also exercise the resume parser
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
    print(f"[{mark}] {name}" + (f"\n           {detail}" if detail else ""))


def check(name: str, condition: bool, detail_ok: str = "", detail_bad: str = "",
          soft: bool = False) -> bool:
    if condition:
        record(name, PASS, detail_ok)
        return True
    record(name, WARN if soft else FAIL, detail_bad)
    return False


# ---------------------------------------------------------------------------
# Phase 1 -- does the app boot at all
# ---------------------------------------------------------------------------

def phase_boot() -> None:
    print("\n--- PHASE 1: app boots ---")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        record("streamlit AppTest available", FAIL,
               "needs streamlit>=1.28 -- pip install -U streamlit")
        return

    try:
        at = AppTest.from_file("app.py", default_timeout=180)
        at.run()
    except Exception:
        record("app.py runs headlessly", FAIL, traceback.format_exc(limit=3))
        return

    if at.exception:
        record("app.py runs headlessly", FAIL,
               "\n".join(str(e.message) for e in at.exception))
        return
    record("app.py runs headlessly", PASS)

    # config.check() writes st.error and calls st.stop() when a key is missing.
    errors = [e.value for e in at.error]
    check("configuration is complete",
          not errors,
          detail_bad="; ".join(errors) or "st.error was written")

    check("title rendered", any("Fitly" in t.value for t in at.title),
          detail_bad="no st.title containing 'Fitly'")

    n_side = len(at.sidebar.selectbox) + len(at.sidebar.radio) + \
             len(at.sidebar.multiselect) + len(at.sidebar.toggle) + \
             len(at.sidebar.checkbox) + len(at.sidebar.text_input)
    check("sidebar controls rendered", n_side >= 4,
          detail_ok=f"{n_side} controls",
          detail_bad=f"only {n_side} sidebar controls -- the demo needs "
                     "strategy / mode / rerank / location")

    check("resume uploader present", len(at.file_uploader) >= 1,
          detail_bad="no st.file_uploader -- the match tab cannot work")


# ---------------------------------------------------------------------------
# Phase 2 -- the demo questions, through the app's own call path
# ---------------------------------------------------------------------------

DEMO = [
    # (question, expect_refusal, why it is in the video)
    ("Which roles require running Kubernetes in production?", False,
     "the working answer -- citations and the retrieval panel"),
    # NOTE: a shortened variant of eval q15 ("...applied to this job SO FAR?").
    # Kept short because it is what gets typed on camera, and because this exact
    # wording has been measured twice at top_sim 0.618 -- ABOVE MIN_SIM 0.60.
    # Guard 1 lets it through; guard 2 refuses it. That is the project's thesis
    # in one query. Do not silently swap in the eval wording without re-measuring:
    # 0.618 sits 0.018 above the threshold, and a different phrasing could fall
    # below it and change which guard fires.
    ("How many people applied to this job?", True,
     "GUARD 2: clears the similarity threshold, model refuses on reading it."),
    ("What is a good recipe for sourdough bread?", True,
     "GUARD 1: off-domain."),
    # VERBATIM from eval/questions.yaml q14. The email clause matters: without
    # it the model can answer truthfully ("a hiring manager is mentioned but not
    # named"), which is grounded and correct -- just not a refusal. Shortening a
    # measured question and expecting the measured outcome is not a test.
    ("Who is the hiring manager for this role and what is their email address?", True,
     "GUARD 2: retrieval is confident, the text does not contain it."),
]


def phase_pipeline() -> None:
    print("\n--- PHASE 2: demo questions through graph.ask() ---")
    import config
    from graph import ask
    from retrieve import Filters

    # Exactly what app.py's sidebar produces with demo settings.
    ctrl = dict(strategy="section", mode="hybrid", use_rerank=True,
                filters=Filters())

    print(f"      config: {config.describe()}")
    if config.VERIFY_ANSWERS:
        record("guard 3 is ON", WARN,
               "VERIFY_ANSWERS=true. Answered questions cost one extra API call; "
               "refusals are unaffected. Record this way unless guard 3 overturns "
               "the answered question below -- the UI shows no trace of it either "
               "way, so narrate it rather than pointing at it.")

    for question, expect_refusal, why in DEMO:
        label = question if len(question) < 52 else question[:49] + "..."
        t0 = time.time()
        try:
            state = ask(question, **ctrl)
        except Exception:
            record(f'ask: "{label}"', FAIL, traceback.format_exc(limit=3))
            continue
        dt = time.time() - t0

        ans = state.get("answer")
        res = state.get("retrieval")
        if ans is None or res is None:
            record(f'ask: "{label}"', FAIL, "graph returned no answer/retrieval")
            continue

        got = "refused" if ans.refused else "answered"
        want = "refused" if expect_refusal else "answered"
        check(f'"{label}" -> {want}',
              ans.refused == expect_refusal,
              detail_ok=f"{dt:.1f}s · top_sim={res.top_sim:.3f} · {why}",
              detail_bad=f"got {got}, wanted {want} · top_sim={res.top_sim:.3f}"
                         f"\n           {why}"
                         f"\n           {(ans.reason or ans.text)[:160]}")

        # --- the fields render_answer() actually reads -------------------
        if not res.hits:
            record(f"   panel data for \"{label}\"", WARN,
                   "no hits at all -- the retrieval table will be empty")
            continue

        missing_sim = [h for h in res.hits if h.sim is None]
        check(f'   similarity column populated',
              not missing_sim,
              detail_ok=f"{len(res.hits)} hits, all scored",
              detail_bad=f"{len(missing_sim)}/{len(res.hits)} hits have sim=None "
                         "-- those rows render as blank in the panel")

        if ctrl["use_rerank"]:
            reranked = [h for h in res.hits if h.rerank_score is not None]
            check("   rerank column populated", bool(reranked),
                  detail_ok=f"{len(reranked)}/{len(res.hits)} hits",
                  detail_bad="rerank is on but no hit carries a rerank_score "
                             "-- the column will be all dashes",
                  soft=True)

        # Hybrid should mean two retrievers actually contributed.
        dense = {h.chunk_id for h in res.hits if h.dense_rank is not None}
        sparse = {h.chunk_id for h in res.hits if h.sparse_rank is not None}
        if dense or sparse:
            overlap = len(dense & sparse)
            check("   both retrievers contributed", bool(dense) and bool(sparse),
                  detail_ok=f"dense {len(dense)} · bm25 {len(sparse)} · "
                            f"overlap {overlap}"
                            + ("  <- disjoint, good demo query" if overlap == 0 else ""),
                  detail_bad=f"dense {len(dense)} · bm25 {len(sparse)} -- only one "
                             "retriever returned anything; hybrid is not visible "
                             "on this query",
                  soft=True)

        if not ans.refused:
            check("   answer carries citations", bool(ans.citations),
                  detail_ok=f"{len(ans.citations)} sources will render",
                  detail_bad="no citations -- the Sources block will be empty",
                  soft=True)
            cited_ids = {h.chunk_id for h in ans.citations}
            hit_ids = {h.chunk_id for h in res.hits}
            check("   no dangling citations", cited_ids <= hit_ids,
                  detail_bad="a citation points at a chunk that was not retrieved")

        if ans.refused:
            check("   refusal explains itself", bool(ans.reason),
                  detail_ok=ans.reason[:90],
                  detail_bad="ans.reason is empty -- the caption under the "
                             "warning will be blank, and that caption is the "
                             "whole point of the refusal beat")

        check("   trace populated", bool(state.get("trace")),
              detail_ok=f"{len(state.get('trace', []))} lines",
              detail_bad="the trace code block in the panel will be empty",
              soft=True)


# ---------------------------------------------------------------------------
# Phase 3 -- the filter path, which fails silently
# ---------------------------------------------------------------------------

def phase_filters() -> None:
    print("\n--- PHASE 3: metadata filters (silent-failure path) ---")
    from retrieve import Filters, get_retriever

    r = get_retriever("section")
    q = "senior software engineer"

    base = r.retrieve(q, filters=None)
    check("unfiltered retrieval returns hits", bool(base.hits),
          detail_ok=f"{len(base.hits)} hits",
          detail_bad="the index is empty -- run index.py")
    if not base.hits:
        return

    us = r.retrieve(q, filters=Filters(country="us"))
    check("country filter returns hits", bool(us.hits),
          detail_ok=f"{len(us.hits)} hits for country=us",
          detail_bad="country=us matched NOTHING. Either the corpus has no US "
                     "roles (unlikely) or geo_country is missing from the "
                     "indexed metadata -- re-run index.py after any chunking "
                     "change. This fails silently in the UI.")

    remote = r.retrieve(q, filters=Filters(work_modes=("remote",)))
    check("work-mode filter returns hits", bool(remote.hits),
          detail_ok=f"{len(remote.hits)} remote hits",
          detail_bad="work_mode=remote matched nothing -- check index metadata",
          soft=True)

    # Both halves of a hybrid retriever must apply the same predicate. If the
    # dense side filters and the sparse side does not, filtered results quietly
    # include ineligible chunks.
    if us.hits:
        def meta_of(hit):
            return r.metas[r._pos[hit.chunk_id]] or {}
        wrong = [h for h in us.hits
                 if (meta_of(h).get("geo_country") or "").lower() not in ("", "us")]
        check("filter applied to BOTH retrievers", not wrong,
              detail_ok="every hit satisfies the predicate",
              detail_bad=f"{len(wrong)} hits are not US despite the filter -- "
                         "the BM25 half is not filtering. This is the bug the "
                         "Filters docstring warns about.")

    nonsense = r.retrieve(q, filters=Filters(city="atlantis"))
    check("impossible filter returns nothing", not nonsense.hits,
          detail_ok="pre-filter is actually constraining the search",
          detail_bad=f"city=atlantis returned {len(nonsense.hits)} hits -- the "
                     "filter is being ignored entirely",
          soft=True)


# ---------------------------------------------------------------------------
# Phase 4 -- resumes: parsing, and whether different people get different answers
# ---------------------------------------------------------------------------

# What each synthetic resume is supposed to prove. `must` / `must_not` are
# competency GROUP names from vendor_matching.COMPETENCY_GROUPS.
EXPECTED = {
    "platform_sre.pdf": dict(
        seniority="senior", years=(9, 13), min_skills=20,
        must=["Cloud & platform"], terms=["kubernetes", "terraform"],
        note="dense overlap with the corpus -- the demo resume"),
    "data_ml.docx": dict(
        seniority="senior", years=(6, 10), min_skills=15,
        must=["Data & ML"], terms=["airflow"], parser="python-docx",
        note="DOCX takes a separate parser branch"),
    "junior_frontend.pdf": dict(
        seniority="entry", years=(1, 2), min_skills=8,
        must=["Web & app"], terms=["react"],
        note="1.5 years -- must not round up to mid"),
    "nurse.pdf": dict(
        seniority="senior", years=(8, 11), min_skills=12,
        must=["Clinical & patient care"], must_not=["Cloud & platform"],
        note="taxonomy works outside tech; the CORPUS has nothing for her"),
    "two_column.pdf": dict(
        min_skills=10, must=["Cloud & platform"], terms=["kubernetes"],
        note="scrambled layout -- keyword extraction is order-insensitive"),
}

# Regression cases for the three bugs this test set found. Cheap, no I/O.
SENIORITY_CASES = [
    (["Director of Engineering"], 14.0, "director"),   # "director" contains "cto"
    (["Chief Technology Officer"], 20.0, "executive"),
    (["VP of Platform"], 16.0, "executive"),
    (["Engineering Manager"], 9.0, "manager"),
    (["Staff Engineer"], 11.0, "senior"),
    (["Sr. Analyst"], 6.0, "senior"),
    (["Junior Developer"], 1.0, "entry"),              # "developer" beat "junior"
    (["Frontend Developer"], 1.5, "entry"),            # weak title, low years
    (["Frontend Developer"], 8.0, "mid"),
    (["Team Lead"], 7.0, "manager"),
    (["Leadership Coach"], 5.0, "mid"),                # "lead " must not match
]

YEARS_CASES = [
    ("1.5 years of professional experience", 1.5),     # used to report 5.0
    ("11 years of experience", 11.0),
    ("12+ years of progressive experience", 12.0),
    ("with 7 years of relevant experience", 7.0),
]


def phase_seniority_regression() -> None:
    print("\n--- PHASE 4a: seniority + years regression ---")
    from resume_loader import extract_years, infer_seniority

    bad = [(t_, g, w) for t_, y, w in SENIORITY_CASES
           if (g := infer_seniority(t_, y)) != w]
    check("seniority classification", not bad,
          detail_ok=f"{len(SENIORITY_CASES)}/{len(SENIORITY_CASES)} cases",
          detail_bad="; ".join(f"{t_} -> {g}, wanted {w}" for t_, g, w in bad))

    ybad = [(s, got, want) for s, want in YEARS_CASES
            if (got := extract_years(s)) != want]
    check("years extraction", not ybad,
          detail_ok=f"{len(YEARS_CASES)}/{len(YEARS_CASES)} cases",
          detail_bad="; ".join(f"{s!r} -> {g}, wanted {w}" for s, g, w in ybad))


def phase_resumes(paths: list[str]) -> None:
    print("\n--- PHASE 4b: resume parsing, per person ---")
    from resume_loader import load as load_resume, to_query

    parsed_by_name: dict[str, object] = {}

    for path in paths:
        name = os.path.basename(path)
        exp = EXPECTED.get(name, {})
        print(f"\n  {name}" + (f"   ({exp['note']})" if exp.get("note") else ""))
        try:
            r = load_resume(path, prefer_llamaparse=False)
        except Exception:
            record(f"{name}: parses", FAIL, traceback.format_exc(limit=3))
            continue
        parsed_by_name[name] = r

        check(f"{name}: parses", r.usable,
              detail_ok=f"{r.parser} · {len(r.text)} chars · {len(r.skills)} skills · "
                        f"{r.years_experience} yrs · {r.seniority or 'unlabelled'}",
              detail_bad="parsed but unusable")
        for w in r.warnings:
            record(f"   {name}: parser warning", WARN, w[:120])

        if exp.get("parser"):
            check(f"{name}: parser branch", r.parser == exp["parser"],
                  detail_ok=r.parser,
                  detail_bad=f"used {r.parser}, expected {exp['parser']}")

        if exp.get("min_skills"):
            check(f"{name}: skills extracted", len(r.skills) >= exp["min_skills"],
                  detail_ok=f"{len(r.skills)} terms",
                  detail_bad=f"only {len(r.skills)}, expected >= {exp['min_skills']}")

        for group in exp.get("must", []):
            check(f"{name}: has '{group}'", group in r.competencies,
                  detail_ok=", ".join(r.competencies.get(group, [])[:6]),
                  detail_bad=f"missing -- found {list(r.competencies)}")

        for group in exp.get("must_not", []):
            check(f"{name}: no '{group}'", group not in r.competencies,
                  detail_ok="correctly absent",
                  detail_bad=f"unexpectedly present: {r.competencies.get(group)}",
                  soft=True)

        for term in exp.get("terms", []):
            check(f"{name}: found '{term}'", term in r.skills,
                  detail_bad=f"'{term}' is in the document but not in the skill list")

        if exp.get("seniority"):
            check(f"{name}: seniority", r.seniority == exp["seniority"],
                  detail_ok=r.seniority,
                  detail_bad=f"got {r.seniority!r}, expected {exp['seniority']!r}")

        if exp.get("years"):
            lo, hi = exp["years"]
            y = r.years_experience
            check(f"{name}: years in range", y is not None and lo <= y <= hi,
                  detail_ok=f"{y}",
                  detail_bad=f"got {y}, expected {lo}-{hi}")

        q = to_query(r, "")
        check(f"{name}: query built", bool(q.strip()),
              detail_ok=q[:100],
              detail_bad="to_query returned nothing to search with")

    _cross_domain(parsed_by_name)


def _cross_domain(parsed: dict) -> None:
    """Does a resume from outside the corpus's industry retrieve worse?

    No API calls -- this is retrieval only. If a nurse's resume comes back
    with the same confidence as a platform engineer's, the confidence number
    is not measuring fit and the refusal guard cannot protect anyone.
    """
    from resume_loader import to_query
    from retrieve import get_retriever

    have = [n for n in ("platform_sre.pdf", "nurse.pdf") if n in parsed]
    if len(have) < 2:
        return

    print("\n  cross-domain confidence")
    try:
        r = get_retriever("section")
    except BaseException as e:      # SystemExit when the index is missing
        record("cross-domain check", WARN, f"{type(e).__name__}: {e}"[:150])
        return

    sims = {}
    for name in have:
        res = r.retrieve(to_query(parsed[name], ""), use_rerank=False)
        sims[name] = res.top_sim
        print(f"      {name:22s} top_sim={res.top_sim:.3f}  hits={len(res.hits)}")

    gap = sims["platform_sre.pdf"] - sims["nurse.pdf"]
    check("in-domain resume beats out-of-domain", gap > 0.05,
          detail_ok=f"gap {gap:+.3f} -- similarity tracks fit across industries",
          detail_bad=f"gap {gap:+.3f}. A nurse's resume retrieves tech postings at "
                     "nearly the same confidence as a platform engineer's. Same "
                     "shape as finding 03: similarity is measuring topic, not fit.",
          soft=True)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip phase 2 (no API calls)")
    ap.add_argument("--resume", help="a single resume file to parse")
    ap.add_argument("--resumes", metavar="DIR",
                    default=os.path.join("data", "test_resumes"),
                    help="directory of resumes (default: data/test_resumes)")
    ap.add_argument("--no-resumes", action="store_true", help="skip phase 4")
    args = ap.parse_args()

    print("=" * 72)
    print("FITLY RAG -- end-to-end UI test")
    print("=" * 72)

    phase_boot()
    phase_filters()
    if not args.fast:
        phase_pipeline()
    else:
        print("\n--- PHASE 2 skipped (--fast) ---")
    if not args.no_resumes:
        phase_seniority_regression()
        paths = []
        if args.resume:
            paths.append(args.resume)
        if args.resumes and os.path.isdir(args.resumes):
            paths += [os.path.join(args.resumes, f)
                      for f in sorted(os.listdir(args.resumes))
                      if os.path.splitext(f)[1].lower() in (".pdf", ".docx")]
        if paths:
            phase_resumes(paths)
        else:
            record("resume fixtures present", WARN,
                   f"no resumes in {args.resumes} -- run: python make_test_resumes.py")

    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]

    print("\n" + "=" * 72)
    print(f"{len(results) - len(fails) - len(warns)} passed · "
          f"{len(warns)} warnings · {len(fails)} failed")
    print("=" * 72)

    if fails:
        print("\nDO NOT RECORD UNTIL THESE ARE FIXED:")
        for name, _, detail in fails:
            print(f"  x {name}")
            if detail:
                print(f"      {detail.splitlines()[0]}")
        return 1

    if warns:
        print("\nWorth a look, not blocking:")
        for name, _, detail in warns:
            print(f"  ! {name}" + (f" -- {detail.splitlines()[0]}" if detail else ""))

    print("\nSafe to record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
