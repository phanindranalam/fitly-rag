"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

WHY HYBRID, WITH THE SPECIFIC FAILURES IT FIXES
-----------------------------------------------
This corpus has two query types that break opposite retrievers, which is
what makes it a good hybrid case rather than a checkbox.

Dense alone fails on exact tokens. "TS/SCI", "ICD-10", "SOC 2", "GA4",
"Kubernetes 1.29" are near-arbitrary strings; an embedding places them near
their semantic neighbourhood, so a query for TS/SCI returns generic
"security clearance" text and misses the postings that actually say TS/SCI.

BM25 alone fails on vocabulary mismatch. "Site Reliability Engineer",
"Infrastructure Engineer" and "Platform Engineer" describe overlapping jobs
and share almost no tokens. A lexical retriever treats them as unrelated.

RRF rather than score interpolation, deliberately: cosine similarity and
BM25 scores are on incompatible scales, and normalizing them requires a
weighting constant that has to be tuned per corpus and silently rots.
RRF only uses RANK, so it needs no tuning and no calibration.

The cost is that it discards score magnitude, and magnitude is exactly what
a refusal threshold needs. So the two jobs are split: RRF decides ORDER, and
raw cosine similarity (carried on every Hit as `sim`) decides CONFIDENCE.
Thresholding the fused score instead is a subtle and common bug -- see the
long note on RetrievalResult.confident.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import config
from embeddings import get_embedder
from index import collection_name, get_client, load_corpus

# RRF's smoothing constant. 60 is the value from the original paper and is
# not worth tuning: it controls how sharply rank 1 beats rank 10, and the
# result is famously insensitive to it.
RRF_K = 60


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    company: str = ""
    title: str = ""
    location: str = ""
    url: str = ""
    section: str = ""
    compensation: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    # Raw cosine similarity between the query and this chunk. Kept separate
    # from `score` on purpose: `score` is a fused RANK, good for ordering and
    # meaningless in absolute terms; `sim` is a calibrated distance, useless
    # for ordering a hybrid list and the only honest basis for refusing.
    sim: float | None = None

    def citation(self) -> str:
        """What the generator is told to cite, and what the UI renders."""
        bits = [self.company or "Unknown company", self.title or "Untitled role"]
        label = " - ".join(bits)
        if self.section and self.section not in ("other", "unknown"):
            label += f", {self.section.replace('_', ' ')}"
        return label


@dataclass
class Filters:
    """Hard constraints, applied BEFORE the vector search rather than after.

    This is the single most important structural choice in retrieval here,
    and it is Fitly's geography lesson carried into the RAG layer.

    Post-filtering (retrieve 20 by meaning, then discard the ones in the
    wrong country) looks equivalent and is not: if all 20 nearest chunks
    happen to be in California, an Atlanta user gets zero results and the
    app looks broken. Pre-filtering asks the vector store for the 20 nearest
    chunks THAT ALREADY SATISFY the constraint, so the user always gets 20
    relevant, eligible results.

    The constraint is never a scoring signal. A role in the wrong country is
    not a worse match, it is not a match, and blending it into a similarity
    score would let a strong semantic hit outrank the user's actual
    requirement.
    """
    country: str | None = None          # "us", "gb", ...
    work_modes: tuple[str, ...] = ()    # remote / hybrid / onsite
    city: str | None = None
    companies: tuple[str, ...] = ()

    def to_chroma_where(self) -> dict | None:
        """Chroma's filter dialect. Returns None when unconstrained, because
        passing an empty {} makes Chroma match nothing."""
        clauses = []
        if self.country:
            clauses.append({"geo_country": self.country.lower()})
        if self.work_modes:
            clauses.append({"work_mode": {"$in": [m.lower() for m in self.work_modes]}})
        if self.city:
            clauses.append({"geo_city": self.city.lower()})
        if self.companies:
            clauses.append({"company": {"$in": list(self.companies)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def matches(self, meta: dict) -> bool:
        """The same predicate in Python, for the BM25 side.

        BM25 has no notion of metadata, so the sparse retriever is filtered
        by hand. Both halves of a hybrid retriever MUST apply identical
        constraints, otherwise fusion quietly reintroduces exactly the
        results the filter was meant to remove.
        """
        if self.country and (meta.get("geo_country") or "").lower() != self.country.lower():
            return False
        if self.work_modes and (meta.get("work_mode") or "").lower() not in [m.lower() for m in self.work_modes]:
            return False
        if self.city and self.city.lower() not in (meta.get("geo_city") or "").lower():
            return False
        if self.companies and meta.get("company") not in self.companies:
            return False
        return True

    def describe(self) -> str:
        bits = []
        if self.country:
            bits.append(f"country={self.country}")
        if self.work_modes:
            bits.append(f"mode={'/'.join(self.work_modes)}")
        if self.city:
            bits.append(f"city={self.city}")
        if self.companies:
            bits.append(f"companies={len(self.companies)}")
        return ", ".join(bits) or "unfiltered"


@dataclass
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    dense_only: list[str] = field(default_factory=list)
    sparse_only: list[str] = field(default_factory=list)
    strategy: str = "section"
    mode: str = "hybrid"
    reranked: bool = False
    rerank_note: str = ""
    candidates_before_filter: int = 0

    @property
    def top_sim(self) -> float:
        """Best cosine similarity among the returned chunks."""
        sims = [h.sim for h in self.hits if h.sim is not None]
        return max(sims) if sims else 0.0

    @property
    def confident(self) -> bool:
        """Whether anything cleared the refusal threshold. Designed first,
        per the brief: the app must be able to say it doesn't know.

        WHY THIS THRESHOLDS SIMILARITY AND NOT THE FUSED SCORE
        ------------------------------------------------------
        The obvious implementation -- refuse when the top RRF score is low --
        is broken, and the bug is instructive. RRF scores are built from
        ranks, not distances: the top result of any query scores about
        1/(60+1) per retriever that found it, whether that result is a
        perfect answer or the least-irrelevant chunk in a corpus that has
        nothing to say. Dense retrieval ALWAYS returns k results. So an RRF
        threshold measures how many retrievers agreed, not whether anything
        was actually relevant, and it moves when you toggle dense-only vs
        hybrid, which would make the eval's own ablation shift the refusal
        rate for reasons unrelated to quality.

        Cosine similarity is a real distance and is comparable across modes,
        chunking strategies and k. So: fuse for ORDER, threshold on
        SIMILARITY for refusal. MIN_SIM is swept in eval/run_eval.py rather
        than guessed.
        """
        return self.top_sim >= config.MIN_SIM


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*")


def tokenize(text: str) -> list[str]:
    """Kept deliberately permissive about internal punctuation so "ci/cd",
    "c++", "node.js", "icd-10" and "soc 2" survive as single tokens. A
    standard \\w+ tokenizer shreds exactly the terms BM25 is here to catch."""
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Dense over Chroma, sparse over an in-memory BM25 index.

    BM25 is rebuilt in memory at construction rather than persisted. At this
    corpus size that costs a couple of seconds and removes a whole class of
    index-drift bug where the sparse and dense sides disagree about what
    documents exist.
    """

    def __init__(self, strategy: str = "section"):
        self.strategy = strategy
        self.embedder = get_embedder()

        client = get_client()
        try:
            self.collection = client.get_collection(collection_name(strategy))
        except Exception as exc:
            raise SystemExit(
                f"No '{collection_name(strategy)}' collection. Run:  python index.py"
            ) from exc

        got = self.collection.get(include=["documents", "metadatas"])
        self.ids = got["ids"]
        self.docs = got["documents"]
        self.metas = got["metadatas"]
        if not self.ids:
            raise SystemExit(f"Collection '{collection_name(strategy)}' is empty. Re-run index.py")

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    def _similarities(self, qvec, ids: list[str]) -> dict[str, float]:
        """Cosine similarity between the query and each candidate chunk.

        Computed for the whole candidate set, not just the dense half, so a
        chunk BM25 found and dense missed still gets a comparable confidence
        number. Otherwise an exact-keyword hit ("TS/SCI clearance") would
        have no similarity at all and the refusal guard would throw away the
        one case hybrid retrieval exists to catch.
        """
        if not ids:
            return {}
        import numpy as np

        got = self.collection.get(ids=list(ids), include=["embeddings"])
        q = np.asarray(qvec, dtype="float32")
        q = q / (np.linalg.norm(q) or 1.0)
        out = {}
        for cid, emb in zip(got["ids"], got["embeddings"]):
            v = np.asarray(emb, dtype="float32")
            v = v / (np.linalg.norm(v) or 1.0)
            out[cid] = float(q @ v)
        return out

    def _dense(self, qvec, k: int, filters: "Filters | None") -> list[str]:
        vec = qvec
        where = filters.to_chroma_where() if filters else None
        res = self.collection.query(
            query_embeddings=[vec],
            n_results=min(k, len(self.ids)),
            # PRE-filter: Chroma restricts the search space before computing
            # nearest neighbours, so k results come back that already satisfy
            # the constraint. Post-filtering would return k neighbours and
            # then throw most of them away.
            where=where,
        )
        return res["ids"][0]

    def _sparse(self, query: str, k: int, filters: "Filters | None") -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0:
                break  # no query token present at all; everything after is noise
            if filters and not filters.matches(self.metas[i] or {}):
                continue
            out.append(self.ids[i])
            if len(out) >= k:
                break
        return out

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        top_n: int | None = None,
        filters: "Filters | None" = None,
        use_rerank: bool = False,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """mode is "hybrid" (dense + BM25 fused with RRF) or "dense".

        Dense-only exists so the eval can measure what BM25 actually buys.
        "We used hybrid retrieval" is an architecture claim; "hybrid beat
        dense-only by N points of recall on 18 questions" is a result. The
        ablation has to be runnable or the claim is decoration.
        """
        k = k or config.TOP_K_RETRIEVE
        top_n = top_n or config.TOP_K_CONTEXT

        # When reranking, retrieve MORE candidates than we'll keep: the
        # reranker's whole value is choosing well from a wider pool, and
        # handing it exactly top_n candidates gives it nothing to choose
        # between.
        fetch_k = k * 2 if use_rerank else k

        qvec = self.embedder.encode_query(query)
        dense_ids = self._dense(qvec, fetch_k, filters)
        sparse_ids = [] if mode == "dense" else self._sparse(query, fetch_k, filters)

        fused: dict[str, float] = {}
        dense_rank, sparse_rank = {}, {}
        for rank, cid in enumerate(dense_ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            dense_rank[cid] = rank
        for rank, cid in enumerate(sparse_ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            sparse_rank[cid] = rank

        order = sorted(fused, key=lambda c: -fused[c])
        sims = self._similarities(qvec, order[: fetch_k])
        candidates = []
        for cid in order[: fetch_k]:
            i = self._pos[cid]
            m = self.metas[i] or {}
            candidates.append(Hit(
                chunk_id=cid,
                text=self.docs[i],
                score=fused[cid],
                company=m.get("company", ""),
                title=m.get("title", ""),
                location=m.get("location", ""),
                url=m.get("url", ""),
                section=m.get("section", ""),
                compensation=m.get("compensation", ""),
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
                sim=sims.get(cid),
            ))

        note = ""
        if use_rerank and candidates:
            from rerank import movement, rerank as do_rerank

            before = list(candidates[:top_n])
            hits = do_rerank(query, candidates, top_n)
            note = movement(before, hits)
        else:
            hits = candidates[:top_n]

        return RetrievalResult(
            hits=hits,
            dense_only=[c for c in dense_ids[:top_n] if c not in sparse_rank],
            sparse_only=[c for c in sparse_ids[:top_n] if c not in dense_rank],
            strategy=self.strategy,
            mode=mode,
            reranked=bool(use_rerank and candidates),
            rerank_note=note,
            candidates_before_filter=len(self.ids),
        )

    def explain(self, query: str, k: int | None = None,
                filters: "Filters | None" = None, use_rerank: bool = False,
                mode: str = "hybrid") -> str:
        """Human-readable retrieval trace. Record this in the demo video --
        it is the difference between claiming hybrid retrieval and showing
        it."""
        res = self.retrieve(query, k, filters=filters, use_rerank=use_rerank, mode=mode)
        lines = [
            f"query: {query!r}",
            f"strategy={self.strategy}  mode={mode}"
            f"  filter=({filters.describe() if filters else 'unfiltered'})"
            f"  rerank={'on' if use_rerank else 'off'}",
            f"{'rank':>4} {'fused':>7} {'sim':>6} {'dense':>6} {'bm25':>5} {'rerank':>8}  source",
        ]
        for i, h in enumerate(res.hits, start=1):
            d = str(h.dense_rank) if h.dense_rank else "-"
            sp = str(h.sparse_rank) if h.sparse_rank else "-"
            rr = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-"
            sm = f"{h.sim:.3f}" if h.sim is not None else "-"
            lines.append(f"{i:>4} {h.score:>7.4f} {sm:>6} {d:>6} {sp:>5} {rr:>8}  {h.citation()}")
        if res.rerank_note:
            lines.append(f"  {res.rerank_note}")
        if res.dense_only:
            lines.append(f"  dense found {len(res.dense_only)} chunk(s) BM25 missed entirely")
        if res.sparse_only:
            lines.append(f"  BM25 found {len(res.sparse_only)} chunk(s) dense missed entirely")
        if not res.confident:
            lines.append(f"  top similarity {res.top_sim:.3f} < MIN_SIM {config.MIN_SIM}"
                         f" -> the app refuses this question")
        return "\n".join(lines)


_cache: dict[str, HybridRetriever] = {}


def get_retriever(strategy: str = "section") -> HybridRetriever:
    if strategy not in _cache:
        _cache[strategy] = HybridRetriever(strategy)
    return _cache[strategy]


if __name__ == "__main__":
    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "section"
    queries = sys.argv[2:] or [
        "Which roles require Kubernetes at scale?",
        "TS/SCI clearance",
        "site reliability engineering ownership of production",
        "what is the capital of France",
    ]
    r = get_retriever(strategy)
    for q in queries:
        print(r.explain(q))
        print()
