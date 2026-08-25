# Fitly RAG

Ask questions about real, currently-open job postings and get answers with the
posting attached — or an explicit refusal when the postings don't say.

The corpus is live job descriptions pulled from company ATS boards
(Greenhouse, Ashby, Lever). The point of the project is not that it retrieves;
it is that it **shows what it retrieved and refuses when retrieval fails.**

---

## What's actually in here

| Stage | File | The decision worth defending |
|---|---|---|
| Corpus | `build_corpus.py` | Boilerplate is detected **from the data** (paragraph fingerprints appearing in >15% of postings), not from a regex list of banned phrases |
| Chunking | `chunking.py` | Two strategies built side by side — fixed-window and section-aware — so the comparison is measured, not asserted |
| Embedding | `embeddings.py` | bge-small locally: 384 dims matched to ~400-token chunks, no API cost, no network dependency in the demo |
| Store | `index.py` | Chroma, cosine space, two collections (one per chunking strategy) |
| Retrieval | `retrieve.py` | Dense + BM25 fused with **Reciprocal Rank Fusion**; metadata **pre-filtered**, never post-filtered |
| Reranking | `rerank.py` | Cross-encoder over the fused candidates — kept only if the eval says it helps |
| Orchestration | `graph.py` | LangGraph state machine with a conditional `widen` retry |
| Generation | `generate.py` | Two independent refusal guards; citations are integers the model cannot invent |
| Resume | `resume_loader.py` | LlamaParse with a local fallback; skills extracted deterministically, never by an LLM |
| Evaluation | `eval/run_eval.py` | 18 labelled questions, 6 of them unanswerable by design |

---

## The three decisions that matter

**1. Geography is a hard pre-filter, not a ranking signal.**
Chroma restricts the search space *before* computing nearest neighbours, so
you get k results that already satisfy the constraint. Post-filtering — search
20, then discard the wrong ones — looks equivalent and returns zero results for
an Atlanta user when all 20 nearest chunks happen to be in California.
BM25 applies the identical predicate in Python, because a hybrid retriever
whose two halves filter differently quietly reintroduces exactly what the
filter was meant to remove.

**2. Fuse for order, threshold on similarity for refusal.**
RRF scores are built from ranks. The top result of *any* query scores about
`1/(60+1)` per retriever that found it, whether that result is a perfect answer
or the least-irrelevant chunk in a corpus that has nothing to say. So an RRF
threshold measures retriever agreement, not relevance — and it moves when you
toggle dense-only vs hybrid, which would make the ablation shift the refusal
rate for reasons unrelated to quality. Raw cosine similarity is a real distance
and is comparable across modes, strategies and k. `MIN_SIM` is set from
`--sweep-threshold`, not guessed.

**3. The refusal path was designed first.**
Two independent guards: retrieval-side (nothing clears `MIN_SIM`, the model is
never called) and prompt-side (the model returns `INSUFFICIENT_CONTEXT`). Guard
1 catches *no results*. Guard 2 catches *wrong results*. Six of the eighteen
eval questions exist only to try to break them.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # add NEBIUS_API_KEY
```

You need a clone of [Fitly](https://github.com/phanindranalam/fitly) beside
this folder — `build_corpus.py` reuses its ATS fetchers and its geography
parser rather than reimplementing them. The coupling is one-way: this project
reads from Fitly, never the reverse.

```bash
python build_corpus.py --fitly-path ../fitly --out data/corpus.jsonl
python index.py                                         # builds both collections
streamlit run app.py
```

## Evaluation

```bash
python eval/run_eval.py --retrieval --sweep-threshold   # no API calls, fast
python eval/run_eval.py --generate                      # adds the faithfulness judge
```

The retrieval matrix runs 2 chunking strategies × dense/hybrid × rerank on/off
and reports term-hit rate, MRR, and how far apart the similarity distributions
of answerable and unanswerable questions sit. Results land in `eval/results/`
as both JSON and markdown.

## Known limitations

- **Term-level retrieval labels, not chunk-level relevance judgements.** The
  eval measures whether retrieval surfaced the right *vocabulary*, which is a
  proxy for recall. Real relevance labels would need hand-annotation of the
  corpus and would break on every re-chunk.
- **The faithfulness judge is the same model family as the generator**, so the
  number is optimistic. It is computed identically across configs, so it still
  ranks them correctly.
- **The corpus is a snapshot.** Postings close. This is the right trade for
  reproducibility and the wrong one for actually applying to jobs — for that,
  Fitly fetches live.
- **Titles and years of experience come from regexes**, so creative job titles
  are missed and career gaps inflate the year count. Both are shown as
  approximate.
