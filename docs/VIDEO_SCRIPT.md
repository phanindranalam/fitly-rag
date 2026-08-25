# Demo video — 5 minutes, shot list and script

**Hard rule: 5:00 max.** Practice once with a timer. If you run long, cut
section 5 (the resume tab) before you cut section 4 (the refusal) — the
refusal is the thesis of the whole project.

**Setup before you hit record**
- Terminal open, venv active, in `fitly-rag/`
- `streamlit run app.py` already running in a second terminal, browser tab open
- Sidebar set to: Anywhere / no work-mode filter / section / hybrid / rerank off
- Your resume file on the desktop, ready to drag
- `eval/results/eval-<latest>.md` open in a third tab
- **Close Slack, email, notifications.** Nothing on screen you don't want graded.

---

## 0:00–0:30 — What it is and what problem it solves

> "This is a RAG app over live job postings — about <FILL> descriptions pulled
> from real company hiring boards.
>
> Job seekers don't have a search problem, they have a reading problem. Boards
> already return thousands of matches. Deciding whether one is worth an evening
> means reading eight hundred words, of which maybe two hundred are about the
> actual job.
>
> You could just ask a language model. The problem is it'll give you a fluent,
> confident answer assembled from what postings *usually* say — and acting on
> that costs you a real application. So this app has two rules: every claim
> points at a specific posting, and when the postings don't say, it says the
> postings don't say."

## 0:30–1:15 — The corpus and the boilerplate problem

Screen: terminal, scroll the saved `build_corpus.py` output.

> "The corpus comes from Greenhouse, Ashby and Lever board APIs — <FILL>
> postings from <FILL> companies.
>
> The interesting part is cleaning. Job postings are about forty percent
> boilerplate — the EEO statement, the benefits blurb, the 'about us'
> paragraph. Those blocks are near-identical across thousands of documents, and
> if you chunk naively they become the densest region of your embedding space.
> You ask 'what does this role require' and you get five copies of 'we are an
> equal opportunity employer,' because that text is in every document and
> therefore sits close to everything.
>
> I didn't want a regex list of banned phrases — that's brittle and it encodes
> my guesses. So this detects it from the data: fingerprint every paragraph,
> drop the ones appearing in more than fifteen percent of documents. A
> paragraph repeated by different employers across fifteen percent of postings
> is definitionally not about any particular job.
>
> That removed <FILL> paragraphs, <FILL> percent of the corpus."

Point at the "top offenders" list on screen.

## 1:15–2:15 — A real question, and the retrieval panel

Screen: app, **Ask the postings** tab. Type:
`Which roles require running Kubernetes in production?`

Let the answer render. Read one sentence of it aloud, then scroll to the
sources.

> "Answer with citations, and each citation opens the actual posting. Under it
> is the part most demos hide."

Expand **How this answer was retrieved**. Point at the table.

> "This is the retrieval trace. Dense rank and BM25 rank are separate columns
> because retrieval is hybrid — embeddings for meaning, BM25 for exact tokens.
> They fail on opposite inputs: dense misses rare strings like 'TS/SCI', BM25
> misses paraphrase.
>
> They're fused with Reciprocal Rank Fusion, not by blending scores — cosine
> similarity and BM25 are on incompatible scales, and normalizing them needs a
> constant you have to tune per corpus and that silently rots. RRF only uses
> rank, so there's nothing to tune.
>
> Look at row <FILL: pick a row with a dash in one column> — BM25 found that
> chunk and dense missed it entirely. That's the case hybrid exists for."

## 2:15–3:00 — The refusal (the most important 45 seconds)

Click the example button: **How many people applied to this job?**

> "Now the part I designed first. This question can't be answered from job
> postings — application counts don't exist in the corpus. But it's exactly the
> kind of number a model will happily estimate."

Point at the warning and the caption underneath.

> "It refuses, and it tells you why: top similarity was <FILL>, below the
> threshold of <FILL>. The model was never called.
>
> There are two independent guards. That was guard one, retrieval-side."

Type: `What is the company's parental leave policy in weeks?`

> "This one's harder. Benefits language *is* in the corpus, so retrieval comes
> back confident — guard one doesn't fire. The specific number isn't there, so
> the model itself returns INSUFFICIENT_CONTEXT. Guard one catches *no
> results*. Guard two catches *wrong results*. You need both."

## 3:00–4:00 — Evaluation

Screen: the eval markdown, or run `python eval/run_eval.py --retrieval` live if
it takes under twenty seconds.

> "None of that is worth anything as a claim, so all of it is measured.
> Eighteen questions, six of them unanswerable by design.
>
> This matrix is two chunking strategies — fixed windows versus splitting on the
> posting's own section headings — crossed with dense-only versus hybrid, and
> reranking on or off. Eight configurations.
>
> <FILL: state the actual result. e.g. 'Section-aware chunking beat fixed
> windows by X points of MRR. Hybrid beat dense-only by Y, and the gap is
> almost entirely in the keyword questions, which is what you'd predict.'>
>
> <FILL: and if reranking didn't help — say so. 'Reranking cost 200ms and moved
> nothing, so it's off by default. That's a negative result but it's a measured
> one, and shipping it on because it sounds sophisticated would be worse.'>
>
> The last column is separation — the gap in similarity between answerable and
> unanswerable questions. If that were near zero, no threshold could work and
> the refusal guard would be theatre. It's <FILL>."

Show the sweep table.

> "And that's how MIN_SIM got picked. Not by eye — swept across the labelled
> set."

If time allows, one line on faithfulness:

> "Faithfulness is <FILL> — an LLM judge reading only the context and the
> answer, counting unsupported claims. It's the same model family as the
> generator, so it's a soft grader and that number is optimistic. Worth saying
> out loud."

## 4:00–4:40 — Resume matching

Drag the resume in.

> "Same pipeline, different query. It parses the file — LlamaParse when it's
> available, pypdf as a fallback, and it tells you which one ran, because a
> two-column resume that parses badly gives you a confident wrong match.
>
> Skills come out of a keyword taxonomy, not an LLM. Ask a model to extract
> skills from a resume and it'll produce a reasonable-looking list including
> skills the person doesn't have, because it's completing a pattern. If
> 'Kubernetes' is in this list, it was in the document.
>
> And the query isn't the whole resume — five thousand characters averaged into
> one vector is a blurry average of a person, and it retrieves blurry averages
> of job postings. It's the target role plus the highest-signal terms."

Click **Find matching postings**.

## 4:40–5:00 — Close

> "One thing I'd change: retrieval labels here are term-level, not real
> relevance judgements — it measures whether retrieval found the right
> vocabulary, which is a proxy. Real labels need hand-annotation and break
> every time you re-chunk. That's the honest limitation, and it's in the
> writeup.
>
> Repo and writeup are linked below. Thanks."

---

## Recording notes

- **One take is fine.** Small stumbles read as real; a stiff re-read doesn't.
- **Don't narrate the UI** ("now I'm going to click here"). Say what it means.
- **Every number you say out loud must be on screen.** If it isn't, don't say it.
- Zoom the browser to 125% — retrieval tables are unreadable at default size in
  a compressed video.
- Loom or the built-in screen recorder are both fine. Do not spend time editing.
