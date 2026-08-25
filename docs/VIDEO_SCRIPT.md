# Demo video — 5:00 hard cap

**The handout requires four things.** Miss any and you lose marks that have nothing to do with the build:

1. Walk through your application
2. Explain what you built
3. **Describe how you used AI coding tools** ← easy to forget, explicitly required
4. Demonstrate the final result **live**

Every beat below is tagged with which requirement it satisfies.

---

## Before you hit record

**Set this in `.env` for the recording only:**

```
VERIFY_ANSWERS=false
```

Guard 3 is a second full API call per answer — with it on, one question takes ~60s. Three questions is three minutes of spinner in a five-minute video. Turn it back on afterwards. You'll say the honest line about it in section 4.

**Setup checklist:**

- [ ] `python smoke_test.py` → 12 passed (never record against a broken index)
- [ ] `streamlit run app.py` already running, browser open, **zoom to 125%** — retrieval tables are unreadable at default size in a compressed video
- [ ] Sidebar: Anywhere / no work-mode filter / **section** / **hybrid** / rerank **on**
- [ ] Second tab open on `eval/results/eval-<latest>.md`
- [ ] Terminal visible with the corpus-build output scrolled to the boilerplate stats
- [ ] Your resume file on the desktop
- [ ] **Notifications off. Slack closed.** Nothing on screen you don't want graded.

**Practice once with a timer.** If you run long, cut section 5 before section 3 — the refusal is the thesis.

---

## 0:00–0:25 — Open with the thesis *(req 2)*

> "Most RAG demos show you when the AI knows the answer. I wanted to build one that knows when it doesn't.
>
> This is Fitly — evidence-grounded job intelligence. It answers questions about 874 real job postings from 93 companies, with citations you can open. And when the postings don't contain the answer, it says so instead of guessing.
>
> That refusal path is the part I designed first, and it's what most of the engineering went into."

**Do not** open by listing libraries. Nobody remembers a stack list.

---

## 0:25–1:10 — Corpus and the cleaning finding *(req 2)*

Screen: terminal, `build_corpus.py` output.

> "The corpus is live postings from Greenhouse, Ashby and Lever board APIs. The interesting part is cleaning.
>
> Job postings are mostly not about the job — every one has an EEO statement, a benefits blurb, an 'about us' paragraph. Chunk naively and those become the densest region of your embedding space. Ask what a role requires and you get five copies of 'we are an equal opportunity employer,' because that text is in every document and therefore sits close to everything.
>
> I didn't want a regex blocklist — that's brittle and it encodes my guesses. So this detects boilerplate from the data: fingerprint every paragraph, drop the ones that repeat.
>
> **First version found exactly zero.** Because boilerplate is written per employer — Stripe's EEO names Stripe, Airbnb's names Airbnb. Different fingerprints. My cutoff was 312 documents when the ceiling was 25. Zero was arithmetically guaranteed.
>
> Counting per employer instead: **48.8% of the corpus removed.**"

Point at the top-offenders list on screen.

---

## 1:10–2:10 — A real answer, and the retrieval panel *(reqs 1, 4)*

**Ask the postings** tab. Type: `Which roles require running Kubernetes in production?`

Read one sentence of the answer aloud. Scroll to sources.

> "Answer with citations, each one opens the actual posting."

Expand **How this answer was retrieved**. Point at the columns.

> "This is the part most demos hide. Dense rank and BM25 rank are separate columns because retrieval is hybrid — embeddings for meaning, BM25 for exact tokens. They fail on opposite inputs: dense misses rare strings like TS/SCI, BM25 misses paraphrase.
>
> They're fused with Reciprocal Rank Fusion, not by blending scores — cosine and BM25 sit on incompatible scales, and normalizing them needs a constant you tune per corpus that silently rots. RRF uses rank only. Nothing to tune.
>
> Look at this — **on this query the two retrievers returned completely disjoint results.** They're not agreeing and reinforcing, they're contributing different evidence. That's the whole argument for hybrid, on screen."

---

## 2:10–3:00 — The refusal. **The most important 50 seconds.** *(req 4)*

Click the example: **How many people applied to this job?**

> "Now the part I designed first. Application counts don't exist in job postings — but that's exactly the kind of number a model will happily estimate."

Point at the warning and the caption.

> "It refuses, and it tells you why: top similarity was below the threshold. **The model was never called.** That's guard one, retrieval-side."

Type: `What is a good recipe for sourdough bread?`

> "Same guard, obviously off-domain."

Then type: `Who is the hiring manager for this role?`

> "This one's different. It's still *about* jobs — so retrieval comes back confident and guard one doesn't fire. The model itself returns INSUFFICIENT_CONTEXT.
>
> **Guard one catches 'no results.' Guard two catches 'wrong results.'** You need both, and I have the numbers showing why."

---

## 3:00–3:50 — Evaluation, including what I got wrong *(req 2)*

Screen: the eval markdown.

> "None of that is worth anything as a claim, so all of it is measured. Twenty questions, eight configurations — two chunking strategies, dense versus hybrid, rerank on or off.
>
> Results: **hybrid beat dense-only at both chunk sizes. Reranking improved every configuration** — best case, the right chunk at rank one every single time. Section-aware chunking won, and it won against LangChain's standard splitter, not against my own first attempt.
>
> End to end: **90% refusal accuracy, 97.9% faithfulness, zero hallucinations, zero dangling citations** — and that faithfulness number is graded by a *different model family* than the one that wrote the answers."

Then — this is the moment that separates you:

> "But here's the finding I'd actually put my name on.
>
> I assumed similarity could tell answerable questions from unanswerable ones. **The evaluation proved me wrong** — the gap was 0.013. Nothing.
>
> The reason: every trap question was still *about jobs*. 'What was revenue last quarter' is topically adjacent to a job posting, so retrieval returns company chunks at high similarity that just don't contain revenue.
>
> **Similarity measures topic. It does not measure answerability.** When I added genuinely off-domain questions the gap jumped to 0.203 — two and a half times.
>
> So guard one catches off-*domain* questions and is blind to off-*content* ones. The two guards aren't redundancy — they cover different failure modes. That applies to any RAG system: an HR bot retrieves the parental-leave document perfectly when you ask for a number the document doesn't state."

Then, if you have the seconds — **this is your strongest single line and worth protecting:**

> "One more. My first evaluation said 98.5% faithful. But the generator, the verifier and the judge were all the same model. That's not a measurement, that's Llama agreeing with Llama. So I moved the two checking roles to a different model family — and **both scores went down.** Faithfulness to 97.9, refusal accuracy to 90. That drop is the number I actually trust.
>
> And the independent verifier immediately overturned an answer that was *correct* — 'what languages come up most often.' It was right to. That claim is about three thousand chunks; it only saw five. **Aggregate questions are structurally unverifiable in chunk-based RAG.** No prompt fixes that — the evidence isn't in the context by construction. It's off in this demo because it doubles latency. Reported, not tuned away."

---

## 3:50–4:25 — Resume matching *(reqs 1, 4)*

Drag the resume in.

> "Same pipeline, different query. It parses the file — LlamaParse when available, pypdf as fallback — and tells you which ran, because a two-column resume that parses badly gives you a confident wrong match.
>
> Skills come from a keyword taxonomy, not an LLM. Ask a model to extract skills and it produces a reasonable-looking list including skills the person doesn't have, because it's completing a pattern. If Kubernetes is in this list, it was in the document.
>
> And the prompt is explicit that **absence of a keyword is not evidence of absence of skill** — it says 'not evidenced in your resume,' never 'you lack.' Same philosophy as the refusal path, applied to the person instead of the corpus."

Click **Find matching postings**.

---

## 4:25–5:00 — AI coding tools, and close *(req 3 — do not skip)*

> "On tooling: I built this with Claude as a pair programmer. It was fast at scaffolding, and most valuably it wrote the *arguments* for each design decision into the docstrings — which is what made them falsifiable.
>
> And every single one of the five things I got wrong was in AI-written code or AI-written analysis. All confidently argued in comments. **None of them crashed.**
>
> That's the real lesson: the failure mode of AI-assisted development isn't broken code — broken code announces itself. It's *plausible* code that runs clean and produces a confident wrong number. Four of my five were caught only because I built a check that could falsify them. The fifth needed another human to read it.
>
> Repo and writeup are linked below. Thanks."

---

## Recording notes

- **One take is fine.** Small stumbles read as real; a stiff re-read doesn't.
- **Don't narrate the UI** ("now I'm clicking here"). Say what it *means*.
- **Every number you say aloud must be on screen.** If it isn't, don't say it.
- Loom or the built-in recorder are both fine. **Don't spend time editing.**
- If you overrun: cut §3:50–4:25 (resume). Keep the refusal and keep the AI-tools section — one is your thesis, the other is a grading requirement.
