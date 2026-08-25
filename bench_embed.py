#!/usr/bin/env python3
"""How fast does this machine actually embed? Measure before committing.

    python bench_embed.py

WHY THIS EXISTS
---------------
The first real index run on the target machine clocked about one chunk per
second -- roughly a hundred times slower than bge-small should manage on CPU.
At that rate the two indexes would take four hours instead of four minutes.

The wrong response is to guess at a fix and start another four-hour run. This
times a small sample under each configuration and projects the full run, so
the decision costs ninety seconds instead of an evening. It also prints the
machine facts (cores, torch build, thread count) that explain the number.

Whatever wins here goes in .env, and the number goes in the writeup: "chose
the ONNX backend because it was Nx faster on this hardware" is a measured
engineering decision. "Used ONNX because it's usually faster" is a rumor.
"""

from __future__ import annotations

import os
import platform
import sys
import time

import config

SAMPLE = 128          # enough to be representative, small enough to be quick
BATCH = 64


def machine_facts() -> None:
    print(f"python   {platform.python_version()} ({platform.machine()})")
    print(f"cores    {os.cpu_count()}")
    try:
        import torch

        print(f"torch    {torch.__version__}")
        print(f"threads  {torch.get_num_threads()} intra / "
              f"{torch.get_num_interop_threads()} inter")
    except Exception as exc:
        print(f"torch    unavailable ({exc})")
    try:
        import onnxruntime

        print(f"onnxrt   {onnxruntime.__version__} "
              f"providers={onnxruntime.get_available_providers()}")
    except Exception:
        print("onnxrt   not installed")


def load_sample(n: int = SAMPLE) -> list[str]:
    """Real chunks, not lorem ipsum. Encode time scales with token count, so
    benchmarking on short strings would report a speed the real run never
    reaches."""
    import json

    import chunking

    if not os.path.exists(config.CORPUS_PATH):
        raise SystemExit(f"{config.CORPUS_PATH} not found. Run build_corpus.py first.")
    docs = []
    with open(config.CORPUS_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(json.loads(line))
            if len(docs) >= 60:
                break
    chunks = chunking.chunk_corpus(docs, "section")
    return [c["text"] for c in chunks[:n]]


def time_config(texts: list[str], label: str, **kwargs) -> float | None:
    """Returns chunks/second, or None if the configuration failed to load."""
    from embeddings import LocalEmbedder

    try:
        t0 = time.time()
        emb = LocalEmbedder(**kwargs)
        load_s = time.time() - t0
    except Exception as exc:
        print(f"  {label:34} unavailable ({type(exc).__name__}: {exc})")
        return None

    # One tiny warm-up call: the first forward pass pays lazy allocation and
    # kernel selection costs that would otherwise be blamed on the backend.
    emb.encode_documents(texts[:8], batch_size=8)

    t0 = time.time()
    emb.encode_documents(texts, batch_size=BATCH)
    dur = time.time() - t0
    rate = len(texts) / dur
    print(f"  {label:34} {rate:7.1f} chunks/s  "
          f"({dur:.1f}s for {len(texts)}, {load_s:.1f}s load)")
    return rate


def main() -> int:
    print(config.describe())
    machine_facts()

    texts = load_sample()
    avg = sum(len(t) for t in texts) / max(len(texts), 1)
    print(f"\nsample   {len(texts)} real chunks, mean {avg:.0f} chars\n")

    results = {}
    # Baseline first, so the comparison is against what actually ran slowly.
    results["torch, 512 tokens"] = time_config(
        texts, "torch backend, full 512 window",
        backend="torch", max_seq_length=512)
    results["torch, 384 tokens"] = time_config(
        texts, "torch backend, 384 window",
        backend="torch", max_seq_length=384)
    results["onnx, 384 tokens"] = time_config(
        texts, "onnx backend, 384 window",
        backend="onnx", max_seq_length=384)

    live = {k: v for k, v in results.items() if v}
    if not live:
        print("\nEvery local configuration failed. Use hosted embeddings:")
        print("  set EMBED_PROVIDER=nebius in .env")
        return 1

    best = max(live, key=lambda k: live[k])
    rate = live[best]

    # Both strategies, both indexes -- that is what index.py actually does.
    try:
        import json

        n_docs = sum(1 for line in open(config.CORPUS_PATH, encoding="utf-8") if line.strip())
    except Exception:
        n_docs = 2450
    est_chunks = int(n_docs * 3.0) * 2   # ~3 chunks/posting, two strategies
    mins = est_chunks / rate / 60

    print(f"\nfastest: {best} at {rate:.0f} chunks/s")
    print(f"projected full index (~{est_chunks:,} chunks, both strategies): "
          f"{mins:.0f} min")

    if "onnx" in best:
        print("\nPut this in .env:\n  EMBED_BACKEND=onnx\n  EMBED_MAX_SEQ=384")
    elif "384" in best:
        print("\nPut this in .env:\n  EMBED_BACKEND=torch\n  EMBED_MAX_SEQ=384")

    if mins > 25:
        print("\nStill too slow for tonight. Two options, in order:")
        print("  1. Shrink the corpus:")
        print("     python build_corpus.py --fitly-path <path> --limit-per-board 10 "
              "--out data/corpus.jsonl")
        print("  2. Move embedding to the hosted API (uses Nebius credits, "
              "roughly cents for this corpus):")
        print("     set EMBED_PROVIDER=nebius in .env")
        print("\n  Option 2 contradicts the 'embeddings run locally' argument in "
              "config.py.\n  That argument assumed local embedding is fast. On this "
              "machine it is not,\n  and the writeup should say so -- an argument that "
              "loses to a measurement\n  was a bad argument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
