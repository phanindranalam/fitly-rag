"""The pipeline as a LangGraph state machine.

WHY A GRAPH AND NOT A FUNCTION CALL
-----------------------------------
The naive version of this app is one function: retrieve, then generate. A
graph earns its place here because of one node, `widen`.

When the first retrieval pass comes back below threshold, there are two
possible causes and they need different responses. Either the corpus genuinely
does not cover the question, in which case refusing is correct. Or the
question was phrased in vocabulary the corpus does not use ("what do infra
leadership roles want" against postings that all say "platform engineering
manager"), in which case a second, widened pass finds it.

So retrieval becomes conditional and stateful: try, assess, maybe retry with
a broader k, then decide. That is a state machine, and expressing it as one
makes the control flow visible and testable rather than buried in ifs. It
also gives the demo something to show: the trace records which path each
question took, so you can point at a question that was rescued by the widen
node and one that was correctly refused.

    START -> retrieve -> [confident?] -> generate -> verify -> END
                  ^            |
                  |            v
                  +--------- widen (once)

The `verify` node is guard 3, added after the first full evaluation. It reads
the finished answer back against the context that produced it and can overturn
it. That is a second reason for the graph shape: a step that runs
conditionally, can reverse the previous node's output, and is switchable for
the ablation is a state machine whether or not you call it one.
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

import config
from generate import (Answer, answer as generate_answer,
                      apply_verdict, format_context, verify_answer)
from retrieve import RetrievalResult, get_retriever


class GraphState(TypedDict, total=False):
    question: str
    resume: str
    strategy: str
    filters: object          # retrieve.Filters, kept loose to avoid a cycle
    use_rerank: bool
    mode: str                # "hybrid" or "dense" (the eval's ablation switch)
    retrieval: RetrievalResult
    widened: bool
    answer: Answer
    verified: bool
    trace: list[str]
    elapsed_s: float


def _log(state: GraphState, msg: str) -> None:
    state.setdefault("trace", []).append(msg)


def node_retrieve(state: GraphState) -> GraphState:
    strategy = state.get("strategy", "section")
    retriever = get_retriever(strategy)
    filters = state.get("filters")
    res = retriever.retrieve(state["question"], filters=filters,
                             use_rerank=bool(state.get("use_rerank")),
                             mode=state.get("mode", "hybrid"))
    state["retrieval"] = res
    filt = filters.describe() if filters else "unfiltered"
    _log(state, f"retrieve[{strategy}/{res.mode}] k={config.TOP_K_RETRIEVE} filter=({filt}) "
                f"rerank={'on' if state.get('use_rerank') else 'off'} "
                f"-> {len(res.hits)} hits, top similarity {res.top_sim:.3f} "
                f"(threshold {config.MIN_SIM})")
    if res.rerank_note:
        _log(state, f"  {res.rerank_note}")
    if res.dense_only:
        _log(state, f"  dense contributed {len(res.dense_only)} chunk(s) BM25 missed")
    if res.sparse_only:
        _log(state, f"  BM25 contributed {len(res.sparse_only)} chunk(s) dense missed")
    return state


def node_widen(state: GraphState) -> GraphState:
    """Second pass with a much wider net, once only.

    Widening is not free: more candidates means more near-misses in the
    fusion, so this runs exactly once and only when the first pass failed.
    Retrying forever would eventually surface something for any question,
    which is precisely the hallucination-by-retrieval failure the refusal
    path exists to prevent.
    """
    strategy = state.get("strategy", "section")
    retriever = get_retriever(strategy)
    wide_k = config.TOP_K_RETRIEVE * 3
    res = retriever.retrieve(state["question"], k=wide_k, filters=state.get("filters"),
                             use_rerank=bool(state.get("use_rerank")),
                             mode=state.get("mode", "hybrid"))
    state["retrieval"] = res
    state["widened"] = True
    _log(state, f"widen k={wide_k} -> top similarity {res.top_sim:.3f} "
                f"({'recovered' if res.confident else 'still below threshold'})")
    return state


def node_generate(state: GraphState) -> GraphState:
    res: RetrievalResult = state["retrieval"]
    ans = generate_answer(state["question"], res, state.get("resume"))
    state["answer"] = ans
    if ans.refused:
        _log(state, f"refused: {ans.reason}")
    else:
        _log(state, f"generated {len(ans.text)} chars, {len(ans.citations)} citation(s), "
                    f"{ans.latency_s:.2f}s, {ans.prompt_tokens}+{ans.completion_tokens} tokens")
    return state


def node_verify(state: GraphState) -> GraphState:
    """Guard 3. Check the finished answer against the context that produced it.

    This node is why the graph shape earns its place better than it did with
    only `widen`. Verification is genuinely a separate step with its own
    branch: it runs only on answers (never on refusals, which have nothing to
    check), it can overturn the previous node's output, and it is switchable
    so the eval can measure the pipeline with and without it.
    """
    ans: Answer = state["answer"]
    if ans.refused or not ans.hits:
        return state  # nothing asserted, nothing to verify
    if not config.VERIFY_ANSWERS:
        _log(state, "verify skipped (VERIFY_ANSWERS=false)")
        return state

    verdict = verify_answer(format_context(ans.hits), ans.text)
    checked = apply_verdict(ans, verdict)
    state["answer"] = checked
    state["verified"] = verdict.ran

    if not verdict.ran:
        _log(state, f"verify FAILED to run ({verdict.error}) -- answer returned unchecked")
    elif checked.refused:
        _log(state, f"verify OVERTURNED the answer: {checked.reason}")
        if verdict.worst:
            _log(state, f'  worst claim: "{verdict.worst[:110]}"')
    else:
        _log(state, f"verify ok: {verdict.unsupported}/{verdict.total} claims unsupported")
    return state


def route_after_retrieve(state: GraphState) -> str:
    if state["retrieval"].confident:
        return "generate"
    if state.get("widened"):
        return "generate"  # already tried; let the generator emit the refusal
    return "widen"


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(GraphState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("widen", node_widen)
    g.add_node("generate", node_generate)
    g.add_node("verify", node_verify)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            {"widen": "widen", "generate": "generate"})
    g.add_conditional_edges("widen", route_after_retrieve,
                            {"widen": "generate", "generate": "generate"})
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def ask(question: str, resume: str | None = None, strategy: str = "section",
        filters=None, use_rerank: bool = False, mode: str = "hybrid") -> GraphState:
    """One question through the whole pipeline. Returns the full state so
    callers can show the trace, not just the answer."""
    t0 = time.time()
    state = get_graph().invoke({
        "question": question,
        "resume": resume or "",
        "strategy": strategy,
        "filters": filters,
        "use_rerank": use_rerank,
        "mode": mode,
        "verified": False,
        "trace": [],
    })
    state["elapsed_s"] = time.time() - t0
    return state


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Which roles require running Kubernetes at scale?"
    print(config.describe())
    st = ask(q)
    print("\n".join(st["trace"]))
    print(f"\ntotal {st['elapsed_s']:.2f}s "
          f"(budget {config.LATENCY_BUDGET_S}s)\n")
    print(st["answer"].text)
    if st["answer"].citations:
        print("\nSources:")
        for h in st["answer"].citations:
            print(f"  - {h.citation()} {h.url}")
