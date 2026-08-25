# Fitly RAG — Week 2 Project Writeup

> **Before submitting:** every `<FILL: …>` below is a number that comes out of
> `python eval/run_eval.py`. Paste the real values in. A writeup with real
> numbers that are mediocre beats one with impressive numbers you didn't
> measure — and the whole argument of this project is that unmeasured claims
> are worthless.

**Repo:** <FILL: github url>
**Live app:** <FILL: streamlit url>
**Demo video:** <FILL: video url>

---

## 1. The problem, and why RAG is the right shape for it

Job seekers do not have a search problem. Job boards already return thousands
of matching postings. They have a **reading** problem: to decide whether a
posting is worth an evening of tailoring a resume, you have to read 800 words
of which maybe 200 are about the actual job, and you have to do that forty
times.

The obvious fix — ask a language model — fails in a specific and dangerous way.
Ask a model "which of these roles need Kubernetes at scale" and it will give
you a fluent, plausible, confidently-worded answer assembled from what job
postings *usually* say. Acting on that costs a real application to a role that
never wanted what you thought it wanted.

So the requirements are:

1. Every claim traces to a specific posting the user can open and check.
2. When the postings don't say, the app says the postings don't say.

That is exactly the contract RAG makes. Requirement 2 is the harder one and it
is the one this project is organized around.

### Why not just semantic search?

Because the useful questions are not lookups. "Which roles would suit someone
who likes turning messy data into things other teams can use" has no keyword
in it. "What do these postings say about on-call" needs five postings
synthesized into three sentences. Search returns documents; the user wanted an
answer with the documents attached.

---

## 2. Corpus

**Source:** live job postings from company applicant-tracking boards —
Greenhouse, Ashby and Lever — fetched through public, keyless board APIs across
~140 companies.

| | |
|---|---|
| Documents | <FILL: n postings> |
| Companies | <FILL: n> |
| Raw size | <FILL: chars> characters |
| Mean posting | <FILL: chars> characters |

**Why this corpus rather than a document set with cleaner structure:** it is
messy in an instructive way. Real HTML from three different vendors, wildly
inconsistent section headings, and — most importantly — heavy boilerplate.

### The boilerplate problem

Job postings are roughly 40% boilerplate. Every posting ends with an EEO
statement, a benefits blurb, an "about us" paragraph and a pay-transparency
notice. These blocks are near-identical across thousands of documents by
different employers.

Chunk naively and they become the **densest region of the embedding space**.
Ask "what does this role require?" and you retrieve five copies of "we are an
equal opportunity employer," because that text appears in every document and
therefore sits close to everything.

The naive fix is a regex list of banned phrases. That is brittle and it
encodes my guesses about what boilerplate looks like. So this detects it
**from the data**: fingerprint every paragraph, and drop the ones that repeat.

### The bug: I picked the wrong unit of analysis

The first implementation counted paragraph frequency **globally** and dropped
anything appearing in more than 15% of all documents. The comment above the
threshold called it "deliberately aggressive."

Run against the real corpus, it found **exactly zero paragraphs.**

The reason is worth the space. Boilerplate here is written *per employer*.
Stripe's EEO paragraph says "Stripe is an equal opportunity employer"; Airbnb's
says Airbnb. Different text, different fingerprint. Across 93 employers, the
most any single fingerprint can reach is the number of postings that **one**
company has open. The global cutoff was 312 documents when the ceiling was
about 25. Zero was arithmetically guaranteed — the code was not detecting
nothing, it was incapable of detecting anything.

The confident comment was the tell. It asserted a property of the data that I
had never measured.

The fix is to count in the right unit, and there are two:

| Pass | Unit | Catches |
|---|---|---|
| per-company | share of *that employer's* postings | EEO, benefits, about-us — the vast majority |
| cross-employer | number of *distinct companies* | text genuinely identical across employers (E-Verify notices, shared benefits-vendor copy) |

The second pass counts **employers, not documents**, for the same reason. One
company with 800 openings would otherwise push its own boilerplate over any
document-based threshold and have it misclassified as industry-wide text.

| | |
|---|---|
| Per-company blocks | <FILL> across <FILL> employers |
| Cross-employer blocks | <FILL> |
| Paragraph instances dropped | <FILL> |
| Characters removed | <FILL> (<FILL>% of corpus) |

Top offenders: <FILL: paste 2-3 from the build output>

The script also prints the most-repeated paragraphs even when nothing clears a
threshold, so a future corpus finding zero again shows immediately whether the
thresholds are wrong or the text simply never repeats — which would mean
paragraph splitting broke upstream. That diagnostic exists because its absence
is what let the original bug read as a clean result.

**Snapshot, not live.** Fitly (the sibling app) fetches postings live on every
search, which is correct for job hunting — a stale posting is worse than a
missing one. It is wrong for RAG: you cannot compare two chunking strategies
against a corpus that changes under you. So the corpus is frozen to a JSONL
file and everything downstream runs against the same file.

---

## 3. Chunking — two strategies, compared

**Strategy A — fixed window.** 1,600 characters with overlap, structure-blind.
The baseline.

**Strategy B — section-aware.** Split on the posting's own headings
(Responsibilities, Requirements, Qualifications, Benefits, About), then window
anything oversized.

The argument for B: a job posting has real internal structure, and a question
about requirements should retrieve the requirements section, not a window that
happens to straddle the end of the responsibilities list and the start of the
benefits blurb. A section that fits stays whole even if it is short — a
300-character requirements list is a better retrieval unit than the same list
padded to 1,600 characters with company prose, because the padding is exactly
what drags the embedding away from the query.

Every chunk carries a **prefix** with the company, title, location and section
name. Without it, a chunk reading "5+ years of experience with distributed
systems" is unattributable: it retrieves fine and is useless in the answer,
because nobody can tell which job it came from.

| | fixed | section |
|---|---|---|
| Chunks | <FILL> | <FILL> |
| Mean chars | <FILL> | <FILL> |
| Min / max | <FILL> | <FILL> |
| Sections identified | — | <FILL> |

**Result:** <FILL — see the retrieval matrix in §6. State plainly which won and
by how much. If section-aware did not win, say so; a negative result you
measured is worth more than a positive one you assumed.>

---

## 4. Embedding and storage

**Model:** `BAAI/bge-small-en-v1.5`, 384 dimensions, running locally on CPU.

Three reasons, in order of weight:

1. **Volume asymmetry.** Embedding is the high-volume, low-value call — the
   corpus produces roughly <FILL> chunks and the chunking comparison requires
   embedding all of it twice. Generation is the opposite: one call per
   question, and its quality *is* the product. So the pipeline has exactly one
   API dependency, exercised once per answer. If the provider is down
   mid-demo, retrieval still works.
2. **Dimension matched to chunk size.** 384 dims against ~400-token chunks. A
   1024-dim model on chunks this small is capacity you pay for and cannot use.
3. **bge's asymmetric prefix.** bge is trained with an instruction prefix on
   queries and none on documents. Skipping that costs recall for free, and it
   is the single most commonly missed detail with this model family.

**Store:** Chroma, persisted to disk, cosine space (not the L2 default —
embeddings are normalized and cosine is what the model was trained for). Two
collections, one per chunking strategy, so both are live simultaneously and
the UI can switch between them mid-demo.

Embedding time: <FILL>s for <FILL> chunks.

---

## 5. Retrieval

### Hybrid, fused with RRF

Dense embeddings and BM25 fail on opposite inputs. Dense misses rare exact
tokens — "TS/SCI", "SOC 2", "ICD-10" — because they are compressed into a
general semantic neighbourhood. BM25 misses paraphrase: "own the reliability
of a large system" and "site reliability engineering" share almost no tokens.

They are fused with **Reciprocal Rank Fusion** rather than by interpolating
scores. Cosine similarity and BM25 scores are on incompatible scales, and
normalizing them requires a weighting constant that must be tuned per corpus
and silently rots. RRF uses only *rank*, so it needs no tuning and no
calibration.

### Metadata is pre-filtered, never post-filtered

This is the most important structural choice in the retrieval layer, carried
over from Fitly.

Post-filtering — retrieve the 20 nearest chunks, then discard the ones in the
wrong country — looks equivalent to pre-filtering and is not. If all 20 nearest
chunks happen to be in California, an Atlanta user gets **zero results and an
app that looks broken**. Pre-filtering asks the vector store for the 20 nearest
chunks *that already satisfy the constraint*, so the user always gets 20
relevant, eligible results.

The constraint is never a scoring signal. A role in the wrong country is not a
worse match; it is not a match, and blending it into a similarity score lets a
strong semantic hit outrank the user's actual requirement.

BM25 has no notion of metadata, so the sparse side applies the identical
predicate in Python. Both halves of a hybrid retriever **must** filter
identically — otherwise fusion quietly reintroduces exactly what the filter
was meant to remove.

### The bug worth writing up: what the refusal threshold measures

The first implementation thresholded the **fused RRF score**: refuse when the
top result scores below `MIN_SCORE`. That is broken, and the way it is broken
is subtle enough to be worth the paragraph.

RRF scores are built from ranks. The top result of *any* query scores about
`1/(60+1) = 0.0164` per retriever that found it — whether that result is a
perfect answer or the least-irrelevant chunk in a corpus with nothing to say.
Dense retrieval always returns k results. So the threshold was measuring **how
many retrievers agreed**, not whether anything was relevant. Worse: dense-only
mode tops out at half the score of hybrid mode, so running the ablation would
have changed the refusal rate for reasons entirely unrelated to answer quality.

The fix splits the two jobs. **RRF decides order; raw cosine similarity decides
confidence.** Similarity is a real distance, comparable across modes, chunking
strategies and k. Every hit carries both.

### Cross-encoder reranking

The retriever's bi-encoder reads query and document *separately* — that is what
makes it fast enough for the whole corpus, and it means the model never sees
the pair together, so it cannot notice that a chunk mentions Kubernetes in the
sentence "no Kubernetes experience required."

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the pair together. Far more
accurate, far too slow for the corpus. So: retrieve widely and cheaply, then
rerank narrowly and accurately. When reranking is on, the retriever fetches 2×
candidates — handing a reranker exactly the number of results you intend to
keep gives it nothing to choose between.

**Reranking costs latency, so it is only worth keeping if it earns it.** See
§6. <FILL: if it did not help on this corpus, say so and say it is off by
default. That is the honest result and it is a better answer than keeping it
because it sounds sophisticated.>

---

## 6. Evaluation

### The eval set

18 questions in five buckets:

| Bucket | n | What it stresses |
|---|---|---|
| factual | 3 | Baseline single-posting retrieval |
| keyword | 3 | Rare exact tokens — where BM25 should carry hybrid |
| semantic | 3 | Paraphrase with no shared vocabulary — where dense should carry it |
| aggregate | 3 | Multi-posting synthesis — stresses `TOP_K_CONTEXT` and chunking |
| **unanswerable** | **6** | **The refusal guards** |

The unanswerable six are not filler. Two are deliberately hard:

- *"What is the company's parental leave policy in weeks?"* — benefits language
  **is** in the corpus, so retrieval will confidently return benefits chunks
  and guard 1 will not fire. This tests guard 2 in isolation.
- *"What is the best way to negotiate a job offer?"* — the model knows a
  perfectly good answer from training data. It must refuse anyway. Grounding
  is the promise; general helpfulness is not.

**On the labels.** The textbook approach marks which chunks are correct for
each question and measures recall@k. That needs hand-annotation and it breaks
the moment the corpus is re-chunked — which this project does on purpose. So
retrieval labels here are **term-level**: the vocabulary that must appear in
the retrieved context for the question to be answerable at all. That is a
weaker signal than relevance judgement and is reported as such — it measures
whether retrieval found the right vocabulary, not the best passage. The
*refusal* labels are exact and human-checkable, and they are what the app is
really judged on.

### Retrieval matrix

<FILL: paste the markdown table from `python eval/run_eval.py --retrieval`>

Read it for three things:

- **hybrid vs dense** — does BM25 earn its complexity? Check the `keyword`
  questions specifically.
- **section vs fixed** — does structure-aware chunking beat the baseline?
- **separation** — the gap between mean similarity on answerable vs
  unanswerable questions. If this is near zero, no threshold can work and the
  refusal guard is theatre.

### Threshold

<FILL: paste the sweep table>

`MIN_SIM = <FILL>`, chosen for <FILL: accuracy>, leaving <FILL> trap question(s)
reaching the model for guard 2 to catch. The two error directions are not
equally bad: answering an unanswerable question is a hallucination; refusing an
answerable one is merely unhelpful. The threshold is set accordingly.

### Generation

<FILL: paste the generation table>

- **Faithfulness** — the headline number. An LLM judge sees only the context
  and the answer, never the question's label, and counts claims the context
  does not support.
- **Refusal accuracy** — with both error directions broken out.
- **Dangling citations** — `[n]` indices pointing at nothing. Should be zero;
  the model cites by integer precisely because it cannot invent an integer
  that maps to a company.
- **Widen rescues** — questions the conditional retry recovered. This is the
  only reason the pipeline is a graph and not a function, so if it is zero,
  the graph is not earning its place. <FILL>

**Judge limitation, stated plainly:** the judge is the same model family as the
generator, so it is a soft grader of its own output and the faithfulness number
is optimistic. It is used because hand-grading 18 answers per config is not
feasible at this scale, and because a consistently optimistic number still
ranks configs correctly.

---

## 7. Architecture

```
build_corpus.py ──► data/corpus.jsonl ──► index.py ──► Chroma (2 collections)
   ATS boards          boilerplate           chunk           postings_fixed
   HTML → text          stripped             embed           postings_section
   geo parsed                                                       │
                                                                    ▼
resume_loader.py ──► query ──────────────────────► graph.py (LangGraph)
  LlamaParse                                            │
  → pypdf fallback                          START ──► retrieve ──► [confident?] ──► generate ──► END
  → deterministic skills                                  ▲              │
                                                          └───── widen ──┘  (once)
```

**Why a graph rather than `retrieve(); generate()`:** one node, `widen`. When
the first pass comes back below threshold there are two possible causes needing
different responses. Either the corpus genuinely does not cover the question —
refusing is correct — or the question used vocabulary the corpus does not
("infra leadership roles" against postings that all say "platform engineering
manager"), and a widened second pass finds it. That makes retrieval conditional
and stateful, which is a state machine. Expressing it as one makes the control
flow testable, and gives the trace something to show.

It retries **exactly once**. Retrying forever would eventually surface
something for any question, which is precisely the hallucination-by-retrieval
failure the refusal path exists to prevent.

---

## 8. What I'd do next

- **Real relevance labels.** Hand-annotate 30 question/chunk pairs and replace
  the term-level proxy for the questions where it matters most.
- **An independent judge.** A different model family for faithfulness, to
  remove the self-grading bias.
- **Incremental indexing.** Rebuilding the whole corpus to add one company is
  fine at this size and will not be at 50,000 postings.
- **Retrieval over the resume too.** Currently the resume is a query and a
  prompt block. Chunking and indexing it would let the app cite *which line of
  your resume* supports each claimed overlap — the same grounding contract,
  applied to the other document.
- **Query rewriting before retrieval.** The widen node is a blunt instrument;
  rewriting the question into corpus vocabulary would likely beat simply
  raising k.
