"""Embedding backends. Local by default, hosted if you ask for it.

The asymmetric-prefix detail matters and is easy to miss: bge models are
trained so that QUERIES carry an instruction prefix and DOCUMENTS do not.
Embedding both sides identically costs a few points of recall for nothing,
and it is the most common way people leave performance on the table with
these models. encode_query and encode_documents are separate functions here
precisely so that asymmetry can't be forgotten at a call site.
"""

from __future__ import annotations

import config


class LocalEmbedder:
    """sentence-transformers on CPU. No key, no network, no per-chunk cost.

    Loading the model takes a few seconds the first time (it downloads ~130MB
    from HuggingFace once, then caches), so it's constructed lazily and
    reused.
    """

    def __init__(self, model_name: str = config.LOCAL_EMBED_MODEL,
                 backend: str | None = None, max_seq_length: int | None = None):
        import os as _os

        # THREADS. torch on CPU sometimes starts with one thread, or with a
        # thread count that fights the OS scheduler, and the difference
        # between right and wrong here is not subtle -- it was the gap
        # between four minutes and two hours on the machine this was built
        # for. Set it explicitly rather than trusting the default.
        try:
            import torch

            n = config.EMBED_THREADS or (_os.cpu_count() or 4)
            torch.set_num_threads(max(1, int(n)))
            # Inter-op parallelism on top of intra-op oversubscribes cores and
            # makes things slower, not faster, for a single sequential encode.
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # already set; only settable once per process
        except Exception:
            pass

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        backend = backend or config.EMBED_BACKEND
        kwargs = {}
        if backend and backend != "torch":
            # onnxruntime is already a dependency (chromadb pulls it in), and
            # for a small encoder-only model on CPU it is typically several
            # times faster than eager torch. Falls back rather than failing:
            # a slow index still finishes, a crashed one does not.
            kwargs["backend"] = backend
        try:
            self.model = SentenceTransformer(model_name, **kwargs)
            self.backend = backend or "torch"
        except Exception:
            self.model = SentenceTransformer(model_name)
            self.backend = "torch"

        # bge-small's default window is 512 tokens. Chunks here run ~350
        # tokens, so the tail of every batch is padding that costs compute and
        # carries no signal. Attention is quadratic in sequence length, so
        # trimming the window is one of the few free speedups available.
        if max_seq_length or config.EMBED_MAX_SEQ:
            self.model.max_seq_length = int(max_seq_length or config.EMBED_MAX_SEQ)

        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()
        self._is_bge = "bge" in model_name.lower()

    def encode_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vecs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        # The prefix the model was trained with. Documents get none.
        query = config.BGE_QUERY_PREFIX + text if self._is_bge else text
        vec = self.model.encode([query], normalize_embeddings=True)[0]
        return vec.tolist()

    def describe(self) -> str:
        return f"local/{self.model_name} ({self.dim}d, {self.backend})"


class NebiusEmbedder:
    """Hosted embeddings over the OpenAI-compatible endpoint.

    Only worth it if you want a larger model than fits comfortably on CPU.
    Note it makes re-indexing a network operation, which slows the
    chunking-comparison loop considerably.
    """

    def __init__(self, model_name: str = config.NEBIUS_EMBED_MODEL):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY)
        self.dim = None  # discovered on first call

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        vecs = [d.embedding for d in resp.data]
        if self.dim is None and vecs:
            self.dim = len(vecs[0])
        return vecs

    def encode_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[i : i + batch_size]))
            if len(texts) > 500:
                print(f"    embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
        return out

    def encode_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def describe(self) -> str:
        return f"nebius/{self.model_name} ({self.dim or '?'}d)"


_cached = None


def get_embedder():
    """One instance per process. Loading the local model twice would double
    memory and startup for no reason."""
    global _cached
    if _cached is None:
        _cached = (NebiusEmbedder() if config.EMBED_PROVIDER == "nebius" else LocalEmbedder())
    return _cached
