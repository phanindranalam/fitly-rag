# Demo video — word-for-word, 5:00 hard cap

Read this aloud. It is timed. Every number you say is on screen when you say it.

**The handout scores four things.** Each beat below is tagged with which one it satisfies:

1. Walk through your application
2. Explain what you built
3. **Describe how you used AI coding tools** ← the reference video I benchmarked skipped this entirely. Free points.
4. Demonstrate the final result **live**

---

## Before you hit record

### Guard 3: leave it ON unless the test says otherwise

`VERIFY_ANSWERS` gates **guard 3 in the live runtime** (`graph.py:132`) — not an
eval-only component. Turning it off weakens a behavior the writeup claims, so
don't do it by default.

It costs less than it looks. `node_verify` returns immediately on refusals
(`if ans.refused ... return state`), and **three of your four demo questions are
refusals.** Only the Kubernetes question pays the extra call — about 40 seconds.

**Decide with evidence, not preference.** Leave `VERIFY_ANSWERS=true` and run:

```cmd
python ui_test.py
```

- Kubernetes question **answers normally** → record with guard 3 **on**. *(This is what happened when tested: `verify ok: 0/5 claims unsupported`.)* You
  demonstrate everything you claim, with no asterisk. Use line A at 3:00.
- Guard 3 **overturns it** → that is the Finding 07 false-positive class firing
  live. Set `VERIFY_ANSWERS=false`, use line B at 3:00, and turn it back on
  after recording.

Either way: **the UI renders no trace of guard 3** — `app.py` never reads
`verified` / `claims_total` / `worst_claim`. So narrate it; don't point at it.

- [ ] `python ui_test.py --fast` → 0 failed
- [ ] `python ui_test.py` → 0 failed *(4 API calls, ~2 min — this is your rehearsal)*
- [ ] `streamlit run app.py` already running, **browser at 125% zoom** — the retrieval table is unreadable at default in a compressed video
- [ ] Sidebar: Anywhere / no work-mode / **section** / **hybrid** / rerank **on**
- [ ] Tab 2: `eval/results/eval-<latest>.md`
- [ ] Tab 3: the whiteboard artifact
- [ ] Terminal visible, scrolled to the boilerplate stats from the corpus build
- [ ] **Notifications off. Slack closed.** Nothing on screen you don't want graded.

**Word budget: 723 spoken words.** At a normal presenting clip (~170 wpm) that is 4 minutes 15 seconds, leaving ~45 seconds for typing and page loads. If you speak slowly it becomes 5:30 — so do one timed practice run.

**Do one timed practice run.** The single most common failure in these videos is running out of time and losing the last section — which is where the AI-tools requirement lives.

---

## 0:00 – 0:16 — Cold open *(req 2)*

**Screen: the app, already open. No title card, no slide.**

> "Most RAG demos show you a system that knows the answer. I wanted one that knows when it *doesn't* — and can prove it.
>
> This is Fitly. 874 real job postings, 93 companies, every answer cited. When the postings don't contain the answer, it says so. That refusal path is what I designed first."

*Sixteen seconds, thesis delivered. Never open with a stack list.*

---

## 0:16 – 0:46 — The corpus, and one fact about the world *(req 2)*

**Screen: terminal, boilerplate stats visible.**

> "Live postings from Greenhouse, Ashby and Lever board APIs — the employer's own text.
>
> The interesting part is cleaning. Job postings are mostly *not about the job* — every one has an EEO statement, a benefits blurb, an 'about us'. So this detects boilerplate from the data: fingerprint every paragraph, drop what repeats.
>
> **First version found exactly zero.** Because boilerplate is written per employer — Stripe's EEO statement names Stripe. My cutoff was 312 documents when the ceiling was 25.
>
> Per employer —" **[point]** "— **48.8%.** Nearly half of what employers write in a job ad is text they put in every other ad."

*A fact about the world, not about your code. It's the line people repeat.*

---

## 0:46 – 1:26 — A real answer, and the retrieval panel *(reqs 1, 4)*

**Type:** `Which roles require running Kubernetes in production?`

Read **one** sentence of the answer. Scroll to Sources.

> "Cited. Each one opens the actual posting."

**Expand "How this answer was retrieved."**

> "This is what most demos hide. Dense rank and BM25 rank, side by side — embeddings for meaning, BM25 for exact strings. Dense misses TS/SCI, BM25 misses paraphrase.
>
> Fused by **rank**, not by score — cosine and BM25 are incompatible scales, and any weight you pick rots as the corpus grows.
>
> And look —" **[point at the rows with a dash in the dense column]** "— **two of these five came from BM25 alone. Dense retrieval never surfaced them.** That's the argument for hybrid, on screen instead of in a bullet."

*Measured on this exact query: dense contributed 3, BM25 contributed 5, overlap 3. Do not say "completely disjoint" — it isn't, and the panel will contradict you.*

---

## 1:26 – 2:16 — The refusals. **The most important 50 seconds.** *(req 4)*

**Type:** `What is a good recipe for sourdough bread?`

> "Start with the easy one. Nothing to do with jobs —" **[point at the caption]** "— top similarity 0.416, threshold 0.60. It refuses, it tells you the number, and **the model was never called.** That's guard one, and it's the cheapest possible refusal: no tokens, no latency."

**Click the example:** `How many people applied to this job?`

> "Now the interesting one. Application counts don't exist in job postings — but that's exactly the number a model will happily estimate.
>
> And watch the similarity: **0.618. It clears the threshold.** Retrieval is *confident*, so guard one waves this straight through. The model reads the actual text and refuses anyway — INSUFFICIENT_CONTEXT.
>
> **That's the whole thesis in one query.** The question is on-topic, so similarity can't tell it's unanswerable. Only reading the text can. **Guard one catches 'no results.' Guard two catches 'wrong results.'** Two failure modes, two mechanisms — and I have the numbers proving they aren't redundant."

**Optional third, if time:** `Who is the hiring manager for this role and what is their email address?`

> "Same guard, and a privacy case too — inventing a plausible name and address is a hallucination *and* a leak."

## 2:16 – 3:00 — The evaluation, and being wrong in public *(req 2)*

**Switch to the eval markdown.**

> "None of that counts as a claim, so all of it is measured. Twenty labelled questions, eight configurations. Hybrid beat dense at both chunk sizes, reranking improved every config, and section-aware chunking won — against LangChain's standard splitter, not a strawman I wrote."

**Slow down. This is the peak of the video.**

> "But here's the finding I'd put my name on.
>
> I assumed similarity could separate answerable questions from unanswerable ones. That's the entire premise of guard one. **The evaluation proved me wrong — the gap was 0.013.** Nothing.
>
> Because every trap I wrote was still *about jobs.* 'What was revenue last quarter' is topically adjacent to a job posting, so retrieval returns company chunks at high similarity that don't contain revenue.
>
> **Similarity measures topic. Not answerability.** With genuinely off-domain questions, 0.203.
>
> So guard one is *structurally* blind to on-topic questions it can't answer. An HR bot retrieves the parental-leave policy perfectly when you ask for a number it never states."

---

## 3:00 – 3:36 — The independent judge *(req 2)*

**Screen: eval header showing `judge=Qwen3-235B (independent)`.**

> "One more — the one I'd want a reviewer to see.
>
> My first evaluation said 98.5% faithful. But the generator, the verifier and the judge were **all the same model.** That's not a measurement, that's Llama agreeing with Llama.
>
> I moved both checking roles to a different model family. **Both scores went down** — faithfulness to 97.9, refusal accuracy 95 to 90. That drop is the number I trust, and its size measures how inflated the old one was.
>
> The new verifier then overturned an answer that was *correct.* It was right to: that claim was about 2,916 chunks and it saw five. **Aggregate questions are structurally unverifiable in chunk-based RAG.** Reported, not tuned away."

**Line A — recording with guard 3 ON** (add after the above):

> "That verifier is running right now, on every answer you've seen. It's the reason this one took a beat longer."

**Line B — recording with guard 3 OFF** (add after the above):

> "It's switched off in this recording because it doubles latency on answered questions — and I'm telling you that rather than letting you assume otherwise. Guards one and two, the ones doing the refusing you just watched, are both live."

---

## 3:36 – 3:54 — Resume matching *(reqs 1, 4)* — **CUT THIS FIRST IF BEHIND**

**Drag a resume in.**

> "Same pipeline, different query. Skills come from a keyword taxonomy, never an LLM — ask a model to extract skills and it invents plausible ones. And the prompt says **'not evidenced in your resume', never 'you lack.'** The refusal philosophy, pointed at a person."

---

## 3:54 – 4:30 — AI coding tools *(req 3 — DO NOT SKIP)*

> "On tooling: I built this with Claude as a pair programmer. Fast at scaffolding — and most valuably it wrote the *argument* for each decision into the docstrings, which is what made them falsifiable later.
>
> Here's the honest part. **Every single thing I got wrong was in AI-written code or AI-written analysis.** All confidently argued in comments. **None of it crashed.**
>
> That's the lesson. The failure mode of AI-assisted development isn't broken code — broken code announces itself. It's *plausible* code that runs clean and returns a confident wrong number. Each one was caught only because I built something that could falsify it."

---

## 4:30 – 4:42 — Close

> "Zero hallucinations across twenty questions, judged by a model with no reason to be kind. Repo and writeup are linked below. Thanks."

---

## Timing discipline

You will run long. Everyone does. Cut in this order:

1. **4:20 resume section** — 25 seconds, entirely optional
2. The RRF explanation at 1:15 — shorten to *"fused by rank, not by score, so there's no constant to tune"*
3. The sourdough question at 2:20 — the other two refusals carry it

**Never cut:** the 0.013 finding, the independent-judge section, or the AI-tools section. One is your thesis, one is your differentiator, one is a graded requirement.

---

## Recording notes

- **One take.** Small stumbles read as real; a stiff re-read doesn't.
- **Rehearse the model names out loud once** — `bge-small`, `Llama-3.3-70B`, `Qwen3-235B`, `Chroma`, `LangGraph`. Fumbling your own stack on camera reads as not being in command of it, and it's the easiest thing to fix.
- **Don't narrate the UI.** Not "now I'm clicking here." Say what it *means*.
- **Every number you say aloud must be visible.** If it isn't on screen, don't say it.
- Loom or the built-in recorder. **Don't edit.**
- Watch the clock at 3:00. If you're past 3:10, drop the resume section on the spot.

---

## Video description — paste this where you upload

> **Fitly RAG — evidence-grounded job intelligence**
> Week 2 · Track 2 (LangChain + LangGraph) · bring-your-own use case
>
> A retrieval system over 874 live job postings from 93 companies, built around one question: how does it know when it doesn't know?
>
> Hybrid retrieval (dense + BM25 fused with Reciprocal Rank Fusion), cross-encoder reranking, LangGraph orchestration, and three independent refusal guards. Evaluated across 8 configurations on 20 labelled questions: 97.9% faithfulness, zero unsupported claims, zero dangling citations — graded by a different model family from the one that wrote the answers.
>
> The headline finding is a negative one: retrieval similarity measures *topic*, not *answerability*. Separation between answerable and unanswerable questions was 0.013 until genuinely off-domain questions were added, at which point it jumped to 0.203. Guard 1 is structurally blind to questions that are on-topic but unanswerable — which is why guard 2 exists.
>
> Repo: https://github.com/phanindranalam/fitly-rag
