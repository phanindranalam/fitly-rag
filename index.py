#!/usr/bin/env python3
"""Build one vector index per chunking strategy.

    python index.py                    # builds both, prints the comparison
    python index.py --strategy section # rebuild just one

Both indexes live side by side in the same Chroma directory as separate
collections, so retrieve.py can query either one without a rebuild. That is
what makes the chunking comparison cheap enough to actually run: switch a
flag, re-ask the same 18 questions, diff the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import chunking
import config
from embeddings import get_embedder


def load_corpus(path: str = config.CORPUS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Run build_corpus.py first:\n"
            f"  python build_corpus.py --fitly-path ../fitly --out {path}"
        )
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def get_client():
    import chromadb

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def collection_name(strategy: str) -> str:
    return f"postings_{strategy}"


def build(strategy: str, docs: list[dict], rebuild: bool = True) -> dict:
    client = get_client()
    name = collection_name(strategy)

    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # first run, nothing to delete

    # cosine, not the L2 default: embeddings are normalized, and cosine
    # similarity on normalized vectors is what the model was trained for.
    coll = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    print(f"\n[{strategy}] chunking {len(docs)} documents...")
    chunks = chunking.chunk_corpus(docs, strategy)
    st = chunking.stats(chunks)
    print(f"[{strategy}] {st['chunks']} chunks, mean {st['mean_chars']} chars "
          f"(min {st['min_chars']}, max {st['max_chars']})")
    if strategy == "section":
        print(f"[{strategy}] sections found: {st['sections']}")

    embedder = get_embedder()
    print(f"[{strategy}] embedding with {embedder.describe()}...")
    t0 = time.time()
    vectors = embedder.encode_documents([c["text"] for c in chunks])
    embed_s = time.time() - t0
    print(f"[{strategy}] embedded {len(vectors)} chunks in {embed_s:.1f}s")

    # Chroma rejects None in metadata, so every value is coerced to a string.
    # Every field retrieval filters on has to be here, and has to be a
    # string: Chroma rejects None, and a missing key makes a $eq clause match
    # nothing rather than everything -- so a dropped field turns a location
    # filter into a silent "no results" instead of an error.
    metadatas = [{
        "doc_id": str(c["doc_id"]),
        "section": str(c.get("section") or ""),
        **{f: str(c.get(f) or "") for f in chunking.CARRY_FIELDS},
    } for c in chunks]

    BATCH = 2000  # Chroma caps add() size
    for i in range(0, len(chunks), BATCH):
        coll.add(
            ids=[c["id"] for c in chunks[i : i + BATCH]],
            documents=[c["text"] for c in chunks[i : i + BATCH]],
            embeddings=vectors[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )

    print(f"[{strategy}] stored in collection '{name}'")
    return {"strategy": strategy, "embed_seconds": round(embed_s, 1), **st}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["fixed", "section", "both"], default="both")
    parser.add_argument("--corpus", default=config.CORPUS_PATH)
    args = parser.parse_args()

    problems = config.check()
    # Embedding locally needs no key, so only complain about generation
    # config if it's actually broken in a way that blocks indexing.
    blocking = [p for p in problems if "EMBED_PROVIDER" in p]
    if blocking:
        raise SystemExit("\n".join(blocking))

    print(config.describe())
    docs = load_corpus(args.corpus)
    print(f"Loaded {len(docs)} documents from {args.corpus}")

    strategies = ["fixed", "section"] if args.strategy == "both" else [args.strategy]
    results = [build(s, docs) for s in strategies]

    if len(results) > 1:
        print("\n" + "=" * 66)
        print("CHUNKING COMPARISON (structure only; retrieval quality is eval/)")
        print("=" * 66)
        print(f"{'strategy':10} {'chunks':>8} {'mean chars':>11} {'embed s':>9}")
        for r in results:
            print(f"{r['strategy']:10} {r['chunks']:>8} {r['mean_chars']:>11} {r['embed_seconds']:>9}")
        fixed = next(r for r in results if r["strategy"] == "fixed")
        section = next(r for r in results if r["strategy"] == "section")
        ratio = section["chunks"] / max(fixed["chunks"], 1)
        print(f"\nSection-aware produced {ratio:.1f}x the chunks at "
              f"{section['mean_chars'] / max(fixed['mean_chars'], 1):.1f}x the mean size.")
        print("More, smaller, topically-pure chunks. Whether that helps is what "
              "the eval measures.")
        print("\nNext:  python eval/run_eval.py --strategy both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
