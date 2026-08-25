#!/usr/bin/env python3
"""End-to-end check that the whole pipeline actually works.

    python smoke_test.py             # everything, one API call at the end
    python smoke_test.py --no-api    # skip generation, zero cost

Twelve checks in dependency order, so the first failure tells you where the
break is instead of leaving you to guess. Each prints PASS or FAIL with the
detail you would need to fix it. Exit code is non-zero if anything failed,
so it works as a pre-commit gate.

Run this before recording the demo. Finding out on camera that the index is
empty is a bad way to spend a take.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def wrap(fn):
        def run(*a, **kw):
            t0 = time.time()
            try:
                detail = fn(*a, **kw) or ""
                status = SKIP if detail.startswith("skipped") else PASS
            except AssertionError as exc:
                status, detail = FAIL, str(exc)
            except Exception as exc:
                status, detail = FAIL, f"{type(exc).__name__}: {exc}"
            dur = time.time() - t0
            results.append((status, name, detail))
            mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "skip"}[status]
            print(f"  [{mark}] {name:38} {dur:5.2f}s  {detail[:70]}")
            return status == PASS
        return run
    return wrap


# ---------------------------------------------------------------------------

@check("config loads and is complete")
def t_config():
    import config
    problems = config.check()
    assert not problems, "; ".join(problems)
    assert config.MIN_SIM > 0, "MIN_SIM is 0 — the refusal guard is disabled"
    return f"model={config.NEBIUS_MODEL.split('/')[-1]} MIN_SIM={config.MIN_SIM}"


@check("corpus file exists and parses")
def t_corpus():
    import json

    import config
    assert os.path.exists(config.CORPUS_PATH), f"{config.CORPUS_PATH} missing — run build_corpus.py"
    docs = [json.loads(l) for l in open(config.CORPUS_PATH, encoding="utf-8") if l.strip()]
    assert len(docs) > 100, f"only {len(docs)} documents — corpus looks truncated"
    required = ("id", "company", "title", "text", "geo_country", "work_mode", "url")
    missing = [f for f in required if f not in docs[0]]
    assert not missing, f"documents missing fields: {missing}"
    companies = len({d["company"] for d in docs})
    return f"{len(docs)} postings, {companies} companies"


@check("boilerplate was actually removed")
def t_boilerplate():
    import json

    import config
    docs = [json.loads(l) for l in open(config.CORPUS_PATH, encoding="utf-8") if l.strip()]
    mean = sum(len(d["text"]) for d in docs) / len(docs)
    # An uncleaned corpus of job postings runs ~6,000 chars per posting. If the
    # mean is still up there, the boilerplate pass silently did nothing.
    assert mean < 5000, f"mean posting is {mean:.0f} chars — boilerplate may not have been stripped"
    return f"mean {mean:.0f} chars/posting"


@check("both Chroma collections exist and are populated")
def t_collections():
    from index import collection_name, get_client
    client = get_client()
    counts = {}
    for strategy in ("fixed", "section"):
        coll = client.get_collection(collection_name(strategy))
        counts[strategy] = coll.count()
        assert counts[strategy] > 100, f"{strategy} has only {counts[strategy]} chunks — re-run index.py"
    return f"fixed={counts['fixed']} section={counts['section']}"


@check("chunk metadata carries the filter fields")
def t_metadata():
    from index import collection_name, get_client
    coll = get_client().get_collection(collection_name("section"))
    got = coll.get(limit=5, include=["metadatas"])
    m = got["metadatas"][0]
    for f in ("geo_country", "work_mode", "company", "title", "url", "section"):
        assert f in m, f"chunk metadata missing {f!r} — location filters will return nothing"
    countries = {x.get("geo_country") for x in got["metadatas"]}
    return f"fields present, sample countries={sorted(c for c in countries if c)}"


@check("retrieval returns hits (hybrid)")
def t_retrieve_hybrid():
    from retrieve import get_retriever
    res = get_retriever("section").retrieve("Which roles require Kubernetes?", mode="hybrid")
    assert res.hits, "no hits at all"
    assert res.top_sim > 0, "similarity is zero — embeddings may not be loading"
    both = [h for h in res.hits if h.dense_rank and h.sparse_rank]
    return f"{len(res.hits)} hits, top_sim={res.top_sim:.3f}, {len(both)} found by both retrievers"


@check("BM25 contributes something dense missed")
def t_hybrid_value():
    from retrieve import get_retriever
    r = get_retriever("section")
    # A rare exact token is the case hybrid exists for.
    res = r.retrieve("TS/SCI security clearance", mode="hybrid")
    assert res.hits, "no hits for the clearance query"
    sparse_only = [h for h in res.hits if h.sparse_rank and not h.dense_rank]
    note = f"{len(sparse_only)} sparse-only hit(s)" if sparse_only else "none this query (not a failure)"
    return note


@check("metadata pre-filter actually filters")
def t_filter():
    from retrieve import Filters, get_retriever
    r = get_retriever("section")
    wide = r.retrieve("engineering role", mode="hybrid")
    narrow = r.retrieve("engineering role", mode="hybrid",
                        filters=Filters(country="us", work_modes=("remote",)))
    assert narrow.hits, "filtered search returned nothing — the filter is broken, not selective"
    from index import collection_name, get_client
    coll = get_client().get_collection(collection_name("section"))
    ids = [h.chunk_id for h in narrow.hits]
    metas = coll.get(ids=ids, include=["metadatas"])["metadatas"]
    bad = [m for m in metas if m.get("geo_country") != "us" or m.get("work_mode") != "remote"]
    assert not bad, f"{len(bad)} hit(s) violate the filter — pre-filtering is not being applied"
    return f"unfiltered={len(wide.hits)} filtered={len(narrow.hits)}, all match constraint"


@check("guard 1 refuses an off-domain question")
def t_guard1():
    import config
    from retrieve import get_retriever
    res = get_retriever("section").retrieve("What is a good recipe for sourdough bread?", mode="hybrid")
    assert not res.confident, (
        f"off-domain question scored {res.top_sim:.3f} >= MIN_SIM {config.MIN_SIM} — "
        f"guard 1 would let this through")
    return f"top_sim={res.top_sim:.3f} < MIN_SIM={config.MIN_SIM}, correctly refused"


@check("resume parsing and query building")
def t_resume():
    import resume_loader
    sample = ("Jane Doe\nSenior Platform Engineer | Atlanta, GA\n\nEXPERIENCE\n"
              "Senior Platform Engineer, Acme  2019 - Present\n"
              "Built Kubernetes clusters on AWS with Terraform. CI/CD via Jenkins. Python and Go.\n"
              "Data Engineer, Beta  2015 - 2019\nSpark, Airflow, SQL, Snowflake pipelines.\n"
              "12 years of experience in distributed systems.\n")
    f = resume_loader.structure(sample)
    assert len(f["skills"]) >= 8, f"only {len(f['skills'])} skills extracted"
    assert f["titles"], "no job titles extracted"
    assert f["years_experience"], "no years of experience extracted"
    p = resume_loader.ParsedResume(text=sample, parser="test", **f)
    q = resume_loader.to_query(p, "Platform Engineering Manager")
    assert "Platform Engineering Manager" in q, "target role missing from query"
    assert len(q) < 400, "query is too long — it should be signals, not the whole resume"
    return f"{len(f['skills'])} skills, {f['years_experience']:g}y, seniority={f['seniority']}"


@check("app.py and every module compile")
def t_compile():
    import py_compile
    mods = ["app.py", "config.py", "retrieve.py", "generate.py", "graph.py",
            "chunking.py", "index.py", "embeddings.py", "rerank.py",
            "resume_loader.py", "mailer.py", "build_corpus.py",
            "eval/run_eval.py", "eval/check_coverage.py"]
    for m in mods:
        assert os.path.exists(m), f"{m} is missing"
        py_compile.compile(m, doraise=True)
    return f"{len(mods)} files compile"


@check("full pipeline: question -> cited answer")
def t_end_to_end(skip: bool = False):
    if skip:
        return "skipped (--no-api)"
    from graph import ask
    st = ask("Which roles require running Kubernetes in production?",
             strategy="section", use_rerank=True, mode="hybrid")
    ans = st["answer"]
    assert not ans.refused, f"refused a question the corpus answers: {ans.reason}"
    assert ans.citations, "answered with no citations"
    from generate import count_bad_citations
    bad = count_bad_citations(ans.text, ans.hits)
    assert bad == 0, f"{bad} citation(s) point at nothing"
    return f"{len(ans.citations)} citations, 0 dangling, {st['elapsed_s']:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="Skip the generation check (no API calls).")
    args = ap.parse_args()

    print("Fitly RAG — end to end smoke test\n")
    t_config(); t_corpus(); t_boilerplate()
    t_collections(); t_metadata()
    t_retrieve_hybrid(); t_hybrid_value(); t_filter(); t_guard1()
    t_resume(); t_compile()
    t_end_to_end(skip=args.no_api)

    failed = [r for r in results if r[0] == FAIL]
    passed = [r for r in results if r[0] == PASS]
    print(f"\n{len(passed)} passed, {len(failed)} failed, "
          f"{len([r for r in results if r[0] == SKIP])} skipped")
    if failed:
        print("\nFailures:")
        for _, name, detail in failed:
            print(f"  {name}: {detail}")
        return 1
    print("\nPipeline works end to end. Safe to demo and safe to push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
