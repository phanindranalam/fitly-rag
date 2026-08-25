"""Cross-encoder reranking.

WHY A SECOND SCORING PASS AT ALL
--------------------------------
Retrieval and reranking answer different questions, and the difference is
the whole point.

A bi-encoder (the embedding model in embeddings.py) reads the query and the
document SEPARATELY and compresses each into a single vector. That is what
makes it fast enough to search 6,000 chunks: the documents were embedded
once, offline, and the query is compared to all of them with cheap vector
math. The cost of that speed is that the model never sees the query and the
document together, so it cannot notice that a chunk mentions Kubernetes in
the sentence "no Kubernetes experience required".

A cross-encoder reads the query and one document TOGETHER and scores the
pair. That is far more accurate and far too slow to run over the whole
corpus: it is a full forward pass per candidate.

So the standard shape, which this implements: retrieve widely and cheaply,
then rerank narrowly and accurately. Cast a wide net with a fast method,
then have the careful reader sort the catch.

MEASURE IT OR DROP IT
---------------------
Reranking adds latency, roughly 100-300ms for 20 candidates on CPU. That is
only worth paying if it actually improves the answers, which is exactly what
eval/run_eval.py measures with --rerank on and off. If the numbers come back
flat on this corpus, the honest move is to report that and turn it off, not
to keep it because it sounds sophisticated.
"""

from __future__ import annotations

import os

# ms-marco MiniLM: small, CPU-friendly, and trained specifically on the
# "given a query, is this passage relevant" task, which is precisely the job
# here. A larger reranker would score slightly better and blow the latency
# budget, and the budget was declared up front for a reason.
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_model = None


def get_reranker(model_name: str = DEFAULT_RERANK_MODEL):
    """Lazy singleton. The model is ~80MB and loading it costs a couple of
    seconds, so it must not happen per query."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(model_name)
    return _model


def rerank(query: str, hits: list, top_n: int) -> list:
    """Re-score hits against the query and return the best top_n.

    Returns hits unchanged if reranking is unavailable, rather than failing
    the query. A missing optional model should degrade the answer, never
    break the app.
    """
    if not hits:
        return hits
    try:
        model = get_reranker()
    except Exception:
        return hits[:top_n]

    pairs = [(query, h.text) for h in hits]
    scores = model.predict(pairs)

    for h, s in zip(hits, scores):
        # Kept alongside the fused RRF score rather than replacing it, so
        # the eval and the demo trace can show BOTH orderings and you can
        # point at a chunk the reranker promoted or buried.
        h.rerank_score = float(s)

    ordered = sorted(hits, key=lambda h: -h.rerank_score)
    return ordered[:top_n]


def movement(before: list, after: list) -> str:
    """Human-readable summary of what the reranker changed. Worth showing in
    the demo: 'it reordered things' is a claim, 'it promoted rank 7 to rank 1
    and dropped the previous top hit to 4' is evidence."""
    before_ids = [h.chunk_id for h in before]
    moves = []
    for new_rank, h in enumerate(after, start=1):
        old_rank = before_ids.index(h.chunk_id) + 1 if h.chunk_id in before_ids else None
        if old_rank and old_rank != new_rank:
            direction = "up" if new_rank < old_rank else "down"
            moves.append(f"{h.citation()}: {old_rank} -> {new_rank} ({direction})")
    if not moves:
        return "reranker kept the fusion order unchanged"
    return "reranker moved: " + "; ".join(moves[:4])
