# Fitly RAG — Evidence-Grounded Job Intelligence

**Week 2 · Track 2 (LangChain + LangGraph) · Bring-your-own use case**

| | |
|---|---|
| **Repo** | https://github.com/phanindranalam/fitly-rag |
| **Demo video** | `<FILL: video url>` |

---

## The one-liner

> **My RAG app helps job seekers answer "what does this role actually require, and does it fit me" from 874 job postings fetched from 93 companies' Greenhouse, Ashby and Lever boards, in a Streamlit interface, at 97.9% claim-level faithfulness and zero missed refusals across a 20-question evaluation set, within a declared 5-second retrieval latency budget.**

Against the three rules the framework sets:

**Corpus named specifically.** Not "job data." 874 postings, 93 employers, fetched through public ATS board APIs, capped at 10 per employer, frozen to a JSONL snapshot so every experiment runs against identical input.

**Faithfulness, not relevance.** The target was set before building: *no answer may contain a claim the retrieved postings do not support.* Measured at **97.9%** — one unsupported claim in 47 — by an **independent** LLM judge: a different model family from the generator (`Qwen3-235B` grading `Llama-3.3-70B`), shown only the context and the answer. An earlier run scored 98.5% with the generator grading itself; that number is reported in §8 as Finding 06, because a model is a soft grader of its own output.

**Latency ceiling declared up front: 5 seconds, for retrieval** (`config.LATENCY_BUDGET_S`) — the half of the pipeline the engineering here controls. **Met: retrieval p95 is 4.29s** (§7). End-to-end, including two or three hosted LLM calls, is **p50 12.2s / p95 47.4s**, and that is reported rather than hidden: the generator and the guard-3 verifier dominate it, and the verifier is optional. A budget you can fail is the only kind that tells you anything — so both numbers are here, against the thing each was declared for.

---

## The Framework

| Field | Decision |
|---|---|
| **Use case** | A job seeker asks what real postings require — "which roles need Kubernetes in production?", "do any require a security clearance?" — or uploads a resume and asks where they fit. Surfaced as a Streamlit web app with three modes over one pipeline. |
| **Corpus** | 874 English-language job postings from 93 companies (Stripe, Databricks, Anthropic, SpaceX, Datadog…), fetched live from Greenhouse, Ashby and Lever public board APIs. HTML source, ~2.7M characters after cleaning. The employers own the source of truth; this is a read-only snapshot. |
| **Ingestion + cleaning** | `build_corpus.py` fetches boards concurrently, unescapes HTML entities, converts block tags to paragraph breaks before stripping tags, then removes boilerplate detected **from the data** — paragraph fingerprints repeating across an employer's postings. Removed **48.8%** of the corpus. Geography is parsed once at ingest into filterable fields. |
| **Ingestion + freshness** | **Deliberately frozen. No refresh.** See §3 — this is a real limitation with a real reason, and it is the one field where the right answer for RAG is the wrong answer for the product. |
| **Chunking + embedding** | Two strategies built and compared: fixed 1,600-char windows via LangChain's `RecursiveCharacterTextSplitter`, and section-aware splitting on the posting's own headings. Embedded with `BAAI/bge-small-en-v1.5` (384-dim) locally on CPU. Both indexes stay live so the app can switch between them mid-demo. |
| **Retrieve** | Chroma (cosine), **hybrid**: dense + BM25 fused with Reciprocal Rank Fusion, then cross-encoder reranking. `TOP_K_RETRIEVE=20` → `TOP_K_CONTEXT=5`. Metadata filters (country, work mode, city) applied **before** the vector search on both halves. |

---

## 1. The problem, and why RAG is the right shape

Job seekers do not have a search problem. Boards already return thousands of matches. They have a **reading** problem: deciding whether a posting is worth an evening of tailoring a resume means reading 800 words of which maybe 200 are about the job — forty times over.

The obvious fix fails in a specific, dangerous way. Ask a language model "which of these roles need Kubernetes at scale" and it produces a fluent, confident answer assembled from what postings *usually* say. Acting on that costs a real application to a role that never wanted what you thought.

So two requirements:

1. Every claim traces to a specific posting the user can open and check.
2. When the postings don't say, the app says the postings don't say.

Requirement 2 is the harder one, and the project is organized around it.

**Why not just semantic search?** The useful questions aren't lookups. *"Which jobs suit someone who likes turning messy data into things other teams can use"* contains no keyword. *"What do these postings say about on-call"* needs five postings synthesized into three sentences. Search returns documents; the user wanted an answer with the documents attached.

---

## 2. Corpus

| | |
|---|---|
| Postings with usable text | 883 → **874** after cleaning |
| Companies | **93** (of 112 boards probed; 19 had stale slugs) |
| Size after cleaning | 2,683,451 characters |
| Mean posting | 3,070 characters |
| Geography parsed | us 406 · unknown 183 · ca 58 · gb 49 · in 29 · jp 27 |
| Work mode parsed | remote 362 · hybrid 266 · onsite 246 |

**Why this corpus rather than something cleaner:** it is messy in instructive ways. Real HTML from three vendors, wildly inconsistent section headings, and heavy boilerplate.

**Why capped at 10 postings per employer:** the uncapped fetch returns 15,699 postings, but SpaceX alone accounts for 14% of them. Every aggregate question would really be answering "what does SpaceX ask for." A balanced sample is better evidence than a skewed census. Corpus size was also the cheapest thing to trade when embedding turned out to be slow (§4).

### The boilerplate problem

Job postings are mostly not about the job. Every one carries an EEO statement, a benefits blurb, an "about us" paragraph. These repeat across an employer's whole board, so they become the **densest region of the embedding space**: ask what a role requires and you retrieve five copies of "we are an equal opportunity employer," because that text is in every document and therefore sits close to everything.

The naive fix is a regex list of banned phrases — brittle, and it encodes guesses. Instead this fingerprints every paragraph and drops the ones that repeat. **This is where the project's first and most instructive bug lived — see §7, Finding 01.**

| | |
|---|---|
| Per-company boilerplate blocks | **1,812** across 89 employers |
| Cross-employer blocks | **0** |
| Paragraph instances dropped | 8,446 |
| Characters removed | 2,558,569 — **48.8% of the corpus** |
| Postings that were *entirely* boilerplate | 9 |

Top offenders:

> `[Lyft: 10/10 postings]` At Lyft, our purpose is to serve and connect. We aim to achieve this by cultivating a work environment…
> `[Airbnb: 10/10 postings]` Airbnb was born in 2007 when two hosts welcomed three guests to their San Francisco home…
> `[Pinterest: 10/10 postings]` By submitting this application, I certify that all information submitted in my application…

**Cross-employer detection found zero, in every run.** I predicted E-Verify notices and shared benefits-vendor copy would repeat verbatim across companies. They don't — every employer rewrites even legally-mandated language. A second pass that costs compute and catches nothing on this corpus, reported rather than quietly left in looking useful.

I also estimated boilerplate at "roughly 40%" from intuition. Measured: **48.8%**. Nearly half of every job posting is text that isn't about the job.

---

## 3. Ingestion and freshness — the honest gap

**This corpus does not refresh. There is no freshness SLA. That is a deliberate choice with a real cost.**

The sibling product (Fitly, a live job-search app) fetches postings on every search, because for job hunting a stale posting is worse than a missing one — you cannot apply to a closed role.

That is exactly wrong for RAG. **You cannot compare two chunking strategies against a corpus that changes under you.** A chunking comparison run against a moving corpus measures nothing. So the corpus is frozen to `data/corpus.jsonl` and everything downstream — both indexes, all eight retrieval configurations, the eval set — runs against that identical file.

The cost is real: by the time this is graded, some of these 874 roles will have closed, and the app will cheerfully answer questions about them.

**What production would need**, in order of effort:

1. A `fetched_at` timestamp per posting, surfaced in every citation, so a user sees the age of their evidence
2. Incremental re-index — the current pipeline rebuilds everything to add one company, which is fine at 874 postings and untenable at 50,000
3. A nightly delta job with tombstones for closed postings, plus a "this posting may have closed" flag past a threshold
4. Freshness as an eval dimension — a question is not answered correctly if it is answered from a dead posting

**This is the one framework field where I made the right call for the experiment and the wrong call for the product, and the two are irreconcilable in one artifact.**

---

## 4. Chunking, embedding, storage

**Strategy A — fixed window.** 1,600 characters via LangChain's `RecursiveCharacterTextSplitter` (separators `["\n\n", "\n", ". ", " ", ""]`).

Deliberately the **reference implementation** rather than hand-rolled windowing. If the baseline is my own first attempt, "section-aware won" only means "my second idea beat my first idea." Measured against the standard tool it means something. (This change *lowered* the baseline's score — §7, and it made the comparison more credible, not less.)

**Strategy B — section-aware.** Split on the posting's own headings, then window anything oversized. A short requirements list stays whole rather than being padded out with company prose that drags its embedding away from the query.

Every chunk carries a **prefix** naming company, title, location and section. Without it, a chunk reading "5+ years with distributed systems" retrieves fine and is useless — nobody can tell which job said it.

| | fixed | section |
|---|---|---|
| Chunks | 2,236 | 2,916 |
| Mean chars | 1,275 | 964 |
| Min / max | 139 / 1,600 | 81 / 1,600 |
| Embed time | 378s | 452s |

Sections identified: responsibilities 1,018 · other 865 · requirements 597 · compensation 247 · about_company 131 · nice_to_have 58

**A fix found while testing:** both strategies were emitting sub-20-character chunks — a bare `Requirements:` heading indexed as its own vector. Too short to answer anything, and short text embeds to a point that sits oddly close to everything. Filtered at 80 characters, **applied identically to both strategies** — a filter tuned to help one side would invalidate the comparison.

### Embedding model

`BAAI/bge-small-en-v1.5`, 384 dimensions, local CPU.

**Why local, not hosted:** embedding is the high-volume, low-value call — 5,152 chunks, embedded twice for the comparison. Generation is the opposite: one call per question, and its quality *is* the product. So the pipeline has exactly one API dependency, exercised once per answer. If the provider is down mid-demo, retrieval still works and the retrieved chunks are still demonstrable.

**Why bge specifically:** it is trained with an asymmetric instruction prefix on queries and none on documents. Skipping that costs recall for free, and it is the single most commonly missed detail with this model family.

**A claim I got wrong, and a reviewer caught:** an earlier draft justified 384 dimensions with *"a 1024-dim model on 400-token chunks is capacity you pay for and cannot use."* That reads like physics and isn't. Dimensionality and passage length aren't coupled that simply — capacity depends on the model, its training objective, the domain and the corpus. The honest version: **bge-small was chosen for the tradeoff between retrieval quality, CPU latency and index size on this hardware, and the alternatives were never benchmarked.** Benchmarking bge-small vs bge-base vs E5 is in §10.

**Store:** Chroma, persisted, cosine space (not the L2 default — embeddings are normalized and cosine is what the model was trained for). Two collections, one per strategy, both live simultaneously.

### The hardware finding

Embedding ran at **~6 chunks/second** — roughly 30× slower than bge-small should manage on CPU (Python 3.14, very new torch CPU wheels; the ONNX backend measured identically, indicating a silent fallback). Thread tuning and a shorter token window recovered ~20%, not enough.

Rather than switch to hosted embeddings — which would have changed the embedding model, the vector dimension, and the refusal threshold calibration all at once — the corpus was capped. **A 90-second benchmark (`bench_embed.py`) replaced what would have been a four-hour indexing run that produced the same answer.**

---

## 5. Retrieval

### Hybrid, fused with RRF

Dense embeddings and BM25 fail on opposite inputs. Dense misses rare exact tokens — `TS/SCI`, `SOC 2`, `ICD-10` — because they compress into a general neighbourhood. BM25 misses paraphrase: *"own the reliability of a large system"* and *"site reliability engineering"* share almost no tokens.

Fused with **Reciprocal Rank Fusion** rather than score interpolation. Cosine similarity and BM25 scores sit on incompatible scales; normalizing them needs a weighting constant tuned per corpus that silently rots. RRF uses only rank — nothing to tune, nothing to calibrate.

**Evidence it matters:** on the Kubernetes query, dense and BM25 returned *completely disjoint* top-5 sets. The two retrievers were not agreeing and reinforcing — they contributed different evidence entirely.

### Metadata is pre-filtered, never post-filtered

The most important structural choice in the retrieval layer.

Post-filtering — retrieve 20 nearest chunks, then discard the wrong country — looks equivalent and is not. If all 20 nearest chunks happen to be Californian, an Atlanta user gets **zero results and an app that looks broken**. Pre-filtering asks the store for 20 nearest chunks *that already satisfy the constraint*.

The constraint is never a scoring signal. A role in the wrong country is not a worse match; it is not a match, and blending it into a similarity score lets a strong semantic hit outrank the user's actual requirement.

BM25 has no notion of metadata, so the sparse side applies the identical predicate in Python. **Both halves of a hybrid retriever must filter identically** — otherwise fusion quietly reintroduces exactly what the filter was meant to remove. The smoke test asserts this by re-reading metadata on every returned hit.

### Cross-encoder reranking

The retriever's bi-encoder reads query and document *separately* — what makes it fast enough for the whole corpus, and why it cannot notice a chunk that says "no Kubernetes experience required."

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the pair together. Far more accurate, far too slow for 5,000 chunks. So: retrieve widely and cheaply, then rerank narrowly and accurately. When reranking is on, the retriever fetches 2× candidates — handing a reranker exactly the number you intend to keep gives it nothing to choose between.

---

## 6. Generation, and the three guards

**The refusal path was designed before the happy path**, as the framework instructs.

| Guard | Where | Catches |
|---|---|---|
| **1** | Retrieval-side | Nothing clears `MIN_SIM=0.60` → the model is never called. Cheap, fast, impossible to talk out of. |
| **2** | Prompt-side | Model answers only from numbered context, returns `INSUFFICIENT_CONTEXT` when it can't. Catches confident retrieval of the wrong thing. |
| **3** | Post-generation | A separate call reads the finished answer back against its context and can overturn it. Added after the first full evaluation. |

Guard 2 asks the model to judge sufficiency *while composing an answer* — the moment it is least able to. Guard 3 judges finished text with no obligation to be helpful. That is why it is a third guard rather than a better prompt.

**Citations are integers** into the numbered context, resolved back to company and URL in code. Free-text citations invite the model to invent plausible company names; an integer it cannot invent one that maps to nothing, and any out-of-range index is caught and counted. **Zero dangling citations across 20 questions.**

### The orchestration

```
START → retrieve → [confident?] → generate → verify → END
           ↑             │
           └──── widen ──┘  (once, never repeatedly)
```

LangGraph rather than a sequence of function calls because two steps are genuinely conditional: `widen` retries once with a wider net, and `verify` can reverse the previous node's output. Widening happens **once** — retrying forever eventually surfaces something for any question, which is the hallucination-by-retrieval failure the refusal path exists to prevent.

### Prompts used

**Generation** (`generate.SYSTEM_PROMPT`):

```
You answer questions about job postings using ONLY the numbered context provided.

Rules, in priority order:

1. Every factual claim must be supported by the context. Cite the source with a
   bracketed number like [2] immediately after the claim. A sentence with a fact
   and no citation is a failure.
2. If the context does not contain enough information to answer, reply with exactly
   INSUFFICIENT_CONTEXT and nothing else. Do not guess, do not generalize from world
   knowledge, do not say what is "typically" true. It is always better to refuse than
   to be plausibly wrong.
3. Never invent company names, job titles, salary figures, requirements or locations.
   If a number is not in the context, it does not exist.
4. Quote sparingly and paraphrase mostly, but stay close to the source wording for
   requirements and compensation, where precision matters more than style.
5. Be concise. Three to six sentences unless the question genuinely needs a list.

You are talking to a job seeker deciding where to spend an evening applying. Being
useful means being accurate about what a role actually asks for.
```

**Resume comparison** (`generate.FIT_PROMPT`) — note rule 2, which is the ethical core of the fit feature:

```
You are comparing one person's background against job postings.

Rules, in priority order:

1. Only claim the candidate has experience that appears in their RESUME below. Only
   claim a role requires something that appears in the numbered CONTEXT. Cite postings
   with [n].
2. Absence of a keyword in the resume is NOT evidence the person lacks the skill. Say
   "not evidenced in your resume", never "you lack" or "you don't have".
3. If the context does not support a comparison, reply with exactly INSUFFICIENT_CONTEXT.
4. Never invent requirements, never invent resume content, never estimate a fit
   percentage. You are explaining overlap and gaps, not scoring.
5. Structure: what lines up, what does not, and one sentence on whether it is worth
   applying. Six sentences maximum.

RESUME:
{resume}
```

**Guard 3 verification** (`generate.VERIFY_PROMPT`):

```
You check whether an ANSWER is supported by its CONTEXT.

Read the CONTEXT (numbered source passages) and the ANSWER. For every factual claim
the ANSWER makes, decide whether the CONTEXT states it.

Rules:
- A claim is UNSUPPORTED if the context does not state it, even if it is true in the
  real world.
- Reasonable paraphrase of the context is supported. Added specifics -- numbers,
  durations, names, requirements not present in the context -- are not.
- Framing that asserts nothing ("here are some options") is not a claim.
- Be strict about numbers. If the answer gives a figure the context does not contain,
  that is unsupported.

Reply with ONLY a JSON object and nothing else:
{"total_claims": <int>, "unsupported_claims": <int>, "worst": "<the least supported
sentence, or empty string>"}
```

The eval's faithfulness judge uses a fourth, near-identical prompt, deliberately **never shown the question's label** so it cannot cheat by knowing which questions were traps.

---

## 7. Evaluation

### The set

20 questions, six categories. **13 answerable, 5 unanswerable-but-on-topic, 2 off-domain.**

| Category | n | What it stresses |
|---|---|---|
| factual | 3 | Baseline single-posting retrieval |
| keyword | 3 | Rare exact tokens — where BM25 should carry hybrid |
| semantic | 3 | Paraphrase with no shared vocabulary — where dense should carry it |
| aggregate | 4 | Multi-posting synthesis |
| unanswerable | 5 | Guard 2 — on-topic, but the corpus lacks the answer |
| off-domain | 2 | Guard 1 — not about jobs at all |

**On the labels.** The textbook eval marks which chunks are correct per question and measures recall@k. That needs hand-annotation and breaks on every re-chunk — which this project does on purpose. So retrieval labels are **term-level**: the vocabulary that must appear in retrieved context for the question to be answerable. Weaker than relevance judgement, and reported as such — it measures whether retrieval surfaced the right *vocabulary*, not the best passage. The **refusal** labels are exact and human-checkable, and they are what the app is really judged on.

**Coverage is verified before the eval runs.** `eval/check_coverage.py` greps the corpus for each question's expected terms. A question the corpus cannot answer would score as a retrieval failure for reasons having nothing to do with retrieval — the metric would silently measure corpus coverage while claiming to measure recall. Two questions (security clearance, HIPAA) are flagged **THIN** — 3 and 2 supporting postings. Kept, because a needle in 874 postings is the hardest possible keyword test, and flagged in the report.

### Retrieval matrix — 8 configurations

| config | term hit@5 | MRR | sep off-content | sep off-domain | p95 |
|---|---|---|---|---|---|
| **`section/hybrid/rerank`** | **100%** | **1.000** | +0.084 | +0.203 | 4.29s |
| `fixed/dense/rerank` | 85% | 0.846 | +0.074 | +0.195 | 5.04s |
| `fixed/hybrid/norerank` | 92% | 0.846 | +0.071 | +0.186 | 0.11s |
| `fixed/hybrid/rerank` | 92% | 0.846 | +0.070 | +0.191 | 5.25s |
| `section/dense/rerank` | 85% | 0.846 | +0.085 | +0.210 | 4.11s |
| `section/dense/norerank` | 77% | 0.769 | +0.081 | +0.203 | 0.10s |
| `section/hybrid/norerank` | 85% | 0.769 | +0.081 | +0.203 | 0.13s |
| `fixed/dense/norerank` | 77% | 0.718 | +0.071 | +0.186 | 0.12s |

- **Hybrid beat dense-only at both chunk sizes** — 92% vs 77% (fixed), 85% vs 77% (section). BM25 justified its complexity.
- **Reranking improved every configuration it touched.** Best case MRR 0.769 → 1.000: the right chunk at rank 1 every time. Cost: ~30× the latency.
- **Section-aware won, against a reference baseline** — 100%/1.000 vs 92%/0.846.
- **One configuration breaches the 5s budget** — `fixed/hybrid/rerank` at 5.25s.

### Threshold sweep

`MIN_SIM` was swept across the labelled set rather than guessed. **0.60** is optimal: 80% accuracy, 3 traps correctly refused, **zero** over-refusals at the retrieval stage.

### Generation — final results

| metric | value |
|---|---|
| config | `section/hybrid/rerank` |
| judge | `Qwen/Qwen3-235B-A22B-Instruct-2507` — **independent of the generator** |
| **Refusal accuracy** | **90%** (18/20) |
| **Missed refusals (hallucination risk)** | **0** |
| Over-refusals (unhelpful) | 2 |
| **Faithfulness** | **97.9%** (1/47 claims unsupported) |
| Answers carrying a citation | 11 |
| **Dangling citations** | **0** |
| Questions rescued by the widen node | 0 |
| Guard 3 ran on / overturned | 12 / **1, and it was wrong** (Finding 07) |
| Retrieval p95 | **4.29s** — budget 5.0s, **met** |
| End-to-end p50 / p95 | 12.2s / 47.4s (2–3 hosted LLM calls) |
| Mean prompt tokens | 1,443 |

**Zero missed refusals across 20 questions, under a judge with no incentive to be kind** — no question that should have been declined was answered instead. Claim-level faithfulness is 97.9%: one claim in 47 was judged unsupported, inside an answer that was otherwise grounded. Those are two different measurements and this writeup keeps them apart, because "zero hallucinations" would be the stronger sentence and the less true one. Both errors are over-refusals, and they fail at different stages:

- **q03** — "which postings mention an on-call rotation" — refused at the retrieval guard despite 44 postings discussing it. `MIN_SIM=0.60` bought three correct refusals and cost this one. That is the precision-recall tradeoff made concrete, and 0.60 remains right: an unhelpful refusal is recoverable by rephrasing; a confident fabrication is not.
- **q10** — "what programming languages come up most often" — answered correctly, then **overturned by guard 3**. See Finding 07: the verifier was not malfunctioning, it was right on its own terms, and that is the interesting part.

Swapping the self-judge for an independent one moved faithfulness 98.5% → 97.9% and refusal accuracy 95% → 90%. Both fell. **That is the expected direction and the reason the swap was worth doing** — the earlier numbers were partly a model agreeing with itself.

---

## 8. Iterations: seven arguments that measurement overturned

Every design decision was argued for in a docstring before it was tested. **Seven were wrong.**

### Finding 01 — wrong unit of analysis

The boilerplate detector counted paragraph frequency **globally**, dropping anything in >15% of all documents. Run for real, it found **exactly zero**.

Boilerplate is written per employer — Stripe's EEO names Stripe, Airbnb's names Airbnb. Different text, different fingerprint. Across 93 employers, the ceiling for any fingerprint was ~25 documents; the cutoff was 312. **Zero was arithmetically guaranteed — the code wasn't detecting nothing, it was incapable of detecting anything.**

Counting per employer instead: 48.8% removed. The tell was a confident comment asserting a property of data nobody had measured.

### Finding 02 — a metric that could not fail

The eval checked retrieved text for expected terms with a plain substring test. It reported the HIPAA question as supported by 101 postings. The real number was **2** — the rest were "so**phi**sticated", "gra**phi**cs", "**Phi**ladelphia". Also matching: `go` inside "going"/"algorithm" (645→202), `secret` inside "secrets management" (10→3).

Word-boundary matching fixed it. Had the eval run first, it would have reported near-perfect recall on questions the corpus barely covers.

### Finding 03 — the design was half right

The refusal threshold was first applied to the **fused RRF score**. RRF is built from ranks, so its top value is about the same for a perfect answer and the least-irrelevant chunk in a corpus with nothing to say. It measures retriever *agreement*, not relevance — and it shifts when you toggle dense vs hybrid, so the ablation itself would have moved the refusal rate.

Fixed by splitting the jobs: **RRF decides order, cosine similarity decides confidence.**

Then the measurement rejected *that* too. Answerable questions scored 0.655, unanswerable 0.642 — a gap of **+0.013**. No threshold beat "answer everything."

The reason is the most transferable thing in this project (§9). Adding two genuinely off-domain controls moved separation to **+0.203** and a working threshold appeared.

### Finding 04 — the same error, after learning it

q16 asked for a parental leave policy in weeks, annotated as the hardest question because *"the specific number usually is not there."* The pipeline answered it and was scored as having hallucinated.

A grep found **190 sentences** mentioning parental leave, many with exact durations — Lyft's "18 weeks of paid parental leave," Reddit's "4+ months." **The label was wrong.**

This is Finding 01 repeated: a confident claim about the data, never measured — committed *after* the lesson had been written down explicitly. Relabelled with the correction recorded in `questions.yaml` rather than quietly applied; a test set edited without explanation is indistinguishable from one tuned to flatter the result.

**The evaluation set was, for a while, the only artifact in the project that nothing was checking.**

### Finding 05 — caught by a reviewer, not by me

The embedding section justified 384 dimensions with a rule that reads like physics and isn't (§4). It survived every measurement because nothing here measured it. An external reader caught it.

### Finding 06 — the grader was the defendant

The first full evaluation reported **98.5% faithfulness**. The generator was Llama-3.3-70B. The guard-3 verifier was Llama-3.3-70B. The eval's faithfulness judge was Llama-3.3-70B.

Three roles, one model, one set of blind spots. A claim the generator finds plausible is a claim the judge finds plausible, for the same reasons. "98.5% faithful" meant **"Llama agreed with Llama"**, which is not a measurement — it is a model's self-report with a percent sign on it.

Both checking roles were moved to a different model family (`Qwen3-235B`, configurable via `JUDGE_MODEL` / `VERIFY_MODEL`; `config.describe()` now prints `independent` or `SAME AS GENERATOR` on every run so the setup is visible in any transcript).

Faithfulness fell to **97.9%**, refusal accuracy to **90%**. **Both moved down, which is the whole point.** A believable 97.9% from a judge that shares none of the generator's habits is worth more than a suspicious 98.5% from a self-assessment — and the size of the drop is itself the measurement of how much the old number was inflated.

The old configuration is one env var away, so the comparison is reproducible rather than asserted.

### Finding 07 — the guard was right and the answer was right

Guard 3 had run on 12 answers and overturned nothing (see Finding 06's predecessor run). Under the independent verifier it acted for the first time — and overturned **q10**, *"across the postings you have, what programming languages come up most often?"* The answer had been correct.

The verifier was not malfunctioning. It was **right on its own terms.** The claim "Python appears most often" is a statement about all 2,916 chunks. The verifier was shown 5. Nothing in that context can support a claim of that scope, so it correctly reported the claim as unsupported.

**Aggregate questions are structurally unverifiable in a chunk-based RAG system.** Retrieval hands the generator a sample; a frequency claim is about the population. Every "most common," "how many," "what's typical" question has this shape, and no amount of prompt tuning fixes it — the evidence needed is not in the context by construction. The fix is architectural: route aggregate questions to a query over the metadata store rather than to the retriever (§10, next steps).

**Guard 3 stays on and this is reported rather than tuned away.** A guard that has never acted is untested; a guard that acted once, wrongly, for a diagnosable structural reason, has told you something about the system's shape. The second is more useful than the first, and only one of them is honest to keep quiet about.

---

**All seven share a shape: the code ran without error and produced a confident, wrong number.** Nothing crashed. A test suite would have passed. Only comparing output against what the data should plausibly contain caught them — and one of them needed another reader entirely.

**The negative results are the credible part.** Reranking helped and section-aware chunking won, but also: the similarity guard is blind to half its failure cases, the `widen` node rescued nothing across 20 questions, guard 3 acted exactly once and was wrong, and the headline faithfulness number dropped as soon as a judge with different blind spots was allowed to grade it.

---

## 9. The headline conclusion: retrieval relevance ≠ answerability

| | separation |
|---|---|
| answerable vs **off-content** (revenue, hiring manager, application counts) | **+0.084** |
| answerable vs **off-domain** (sourdough recipe, brake pads) | **+0.203** |

**2.4× difference, and it is the most transferable finding here.**

Similarity measures **topical relatedness**, not **answer-bearing-ness**. "What was revenue last quarter" is topically adjacent to a job posting — it's about the company. Retrieval correctly returns company chunks at high similarity that simply don't contain revenue.

So **guard 1 is structurally incapable of catching off-content questions.** Only guard 2, which reads the text, can. The two guards were never redundancy — they cover **disjoint failure modes**, and there is data proving it.

This generalizes well past job postings. An HR policy bot retrieves the parental-leave document perfectly when asked for a number the document doesn't state. A financial RAG retrieves the right 10-K section for a metric that isn't in it. **Perfect retrieval, zero answer.** Any system that gates refusal on retrieval score alone has this hole.

---

## 10. Limitations and what I'd do next

**Known limitations, stated plainly:**

- **Term-level retrieval labels**, a proxy for relevance. Measures vocabulary, not best passage.
- **The judge is a different model, not a human.** `Qwen3-235B` grading `Llama-3.3-70B` removes the shared-blind-spot problem (Finding 06) but not the LLM-as-judge problem. 47 claims graded by one model is a proxy for human adjudication, not a substitute.
- **Guard 3 has a known false-positive class.** It overturned one correct answer, and the reason is structural rather than incidental: aggregate claims cannot be supported by a 5-chunk sample (Finding 07). Until aggregate questions are routed away from the retriever, guard 3 will keep refusing them. It also costs ~2× generation latency.
- **The corpus is a frozen snapshot** (§3).
- **20 questions is small.** 13 answerable is enough to rank configurations, not enough for tight confidence intervals.
- **Titles and years of experience come from regexes** — creative titles are missed, career gaps inflate the count. Both shown as approximate.

**Next, in priority order — each pointed at by a result above:**

1. **Human adjudication on a sampled subset.** The judge is now a different model family (§8, Finding 06), which removes the shared-blind-spot problem but not the LLM-as-judge problem. 47 claims graded by one model is a proxy for human judgement, not a substitute — and it is the next real limit on how far these numbers can be trusted.
2. **Expand to 100+ questions**, reported by category, adding location, compensation, cross-document and adversarial classes.
3. **Query rewriting to replace `widen`.** Tripling k rescued nothing; rewriting into corpus vocabulary is the version that works.
4. **Fan-out for aggregate questions.** "Which languages appear most often" answered from a single citation — that needs several retrievals merged, a map-reduce shape LangGraph expresses naturally.
5. **Benchmark bge-small vs bge-base vs E5** on MRR, latency and index size — converting Finding 05's opinion into an experiment.
6. **Real relevance labels** for 30 hand-annotated question-chunk pairs, replacing the term-level proxy where it matters most.
7. **Freshness** (§3): `fetched_at` per posting, incremental indexing, tombstones for closed roles.

---

## 11. How I used AI coding tools

Built in a single extended session with **Claude (Cowork mode)** as a pair programmer, with a deliberate division of labour.

**What the AI did well:** scaffolding the pipeline quickly, writing the docstrings that made each design decision explicit and therefore *falsifiable*, and — most valuably — building the diagnostic tooling. `bench_embed.py`, `check_coverage.py` and `smoke_test.py` were all written specifically to test assumptions, and each caught something.

**What it got wrong.** Several of the most consequential errors in §8 came from AI-written code or AI-assisted analysis — and none of them crashed. The global-frequency boilerplate detector, the substring matcher, the RRF threshold, the mislabelled trap question, the embedding-dimension claim. **All were confidently argued in comments before being tested. None crashed.**

**The lesson, and it is the main one I take from this week:** the failure mode of AI-assisted development is not broken code — broken code announces itself. It is *plausible* code that runs cleanly and produces a confident wrong number. Most were caught only because a measurement was compared against what the data should plausibly look like. The fifth needed a human reader.

**So the workflow that actually worked:** have the AI write the argument down explicitly, then build the check that could falsify it, then run the check. That changed how I used the tool: every confident architectural claim in a docstring became a hypothesis to try to falsify. This project falsified seven of them.

The corollary is uncomfortable and worth stating: **knowing this pattern did not prevent it.** Finding 04 is Finding 01 repeated after it had been diagnosed in writing. What catches these is a check that runs, not a lesson that was learned.

---

## Appendix — reproducing this

```bash
pip install -r requirements.txt
cp .env.example .env          # add NEBIUS_API_KEY

python build_corpus.py --fitly-path ../fitly --limit-per-board 10 --out data/corpus.jsonl
python index.py               # builds both collections

python smoke_test.py                                  # 12 end-to-end checks
python eval/check_coverage.py                         # corpus supports the questions?
python eval/run_eval.py --retrieval --sweep-threshold # no API calls
python eval/run_eval.py --generate --config section/hybrid/rerank

streamlit run app.py
```

Every eval run writes timestamped JSON and markdown to `eval/results/`, committed to the repo. **Every number in this document can be checked against those files without re-running anything.**
