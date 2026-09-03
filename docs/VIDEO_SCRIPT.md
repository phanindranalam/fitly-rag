# Demo video — final script · ~4:36 spoken / 5:00 hard cap

> **Time a dry run before you record.** With three trim cards the finished cut lands
> near **4:51** — about nine seconds of margin. If your dry run comes in over 4:40
> spoken, cut **query 2 (1:22)** and stop worrying about the clock.

**The spine is an intellectual progression, not a feature tour:**
*I assumed similarity could tell me when RAG should answer → I measured it → the
assumption failed → I changed the architecture.*

**The handout scores four things:**

1. Walk through your application
2. Explain what you built
3. **Describe how you used AI coding tools** ← explicitly required, easy to lose to the clock
4. Demonstrate the final result **live**

---

## Before you hit record

### The environment, in this exact order

`config.py` reads `VERIFY_ANSWERS` at import, and you are not restarting Streamlit
after warm-up — so the flag is locked at launch. Set it *first*, in the window you
will launch from:

```cmd
set VERIFY_ANSWERS=false
python ui_test.py
streamlit run app.py
```

`ui_test.py` runs 15–25 minutes at current API latency. It has not hung.

### Why guard 3 is off

Measured today, warm, same question both ways:

| | end-to-end |
|---|---|
| guard 3 ON | **237s** |
| guard 3 OFF | **83s** |

154 seconds — 65% of latency — for a check that has never overturned an answer on
this corpus. Refusals are byte-identical either way (`verify_guard3.py` proves it;
`node_verify` returns early on refusals). **You say this out loud at 3:30.** A
measured tradeoff you disclose is a strength; a silent one is an asterisk.

### Checklist

- [ ] `python ui_test.py` → 0 failed
- [ ] Streamlit warmed with a throwaway query, browser at **125%**
- [ ] Sidebar: Anywhere / no work-mode / **section** / **hybrid** / rerank **on**
- [ ] Tab 2: `docs/one-pager.html` **fullscreen (F11)** · Tab 3: `eval/results/eval-<latest>.md`
- [ ] Notifications off, Slack quit

### The three waits

Kubernetes, security clearance and applied-count each cost ~80s. **Record straight
through, then trim each wait in Clipchamp** and caption the cut:

> ⏱ 78s of model latency trimmed — Nebius endpoint degraded, 2026-09-02

Captioning it is what makes it honest rather than edited-to-flatter. Trimmed, you
land near 4:45.

---

## 0:00 – 0:28 — Hook + who's speaking *(req 2)*

**Screen: the app, already open. No title card.**

> "Imagine forty job tabs open and one Sunday to decide where to apply. Across the
> 874 postings I collected, reading all of them would take about 58 hours.
>
> I'm Phanindra. I come from **platform engineering and SRE**, so the question I
> actually wanted answered wasn't *find me more jobs.* It was: which roles genuinely
> **require** running Kubernetes in production — and which just list it?
>
> That's Fitly. But the interesting problem wasn't getting a model to answer questions
> about job ads. It was teaching the system when **not** to answer."

**Pause.**

> "That became the question behind the whole project: does *relevant* retrieval
> actually mean we have enough evidence to answer?"

*The introduction is placed here on purpose. It is not a credential — it is the setup
for the query you type at 0:53, which is then obviously a real question you had rather
than a demo prop. Say it in one breath and move; if it starts to feel like a résumé,
cut back to "I'm Phanindra, a platform engineer and SRE."*

---

## 0:28 – 0:50 — The whole system, once *(req 2)*

**Screen: `docs/one-pager.html`, fullscreen. Trace with the cursor as you talk.**

> "Here's the whole system — it's a RAG pipeline. In plain English: retrieval finds
> the evidence, and the model reads that evidence and decides what it can actually
> say.
>
> A question goes against 874 real postings, a frozen snapshot, so every number I'm
> about to show is reproducible.
>
> Retrieval runs semantic and exact-term search in parallel, fuses and reranks the
> evidence, then a LangGraph state machine routes it through conditional checks and
> retry logic before Fitly answers. Three checks: is this relevant at all, does the
> evidence actually answer it, and are the individual claims supported.
>
> Only two ways out — an answer where every claim links to its posting, or *I don't
> know*, with the reason. That second box is what this project is actually about."

**That is all the architecture you narrate.** Twenty-two seconds. Back to the app.

---

## 0:50 – 1:22 — Useful query 1 *(reqs 1, 4)*

**Type:** `Which roles require running Kubernetes in production?`

**Talk over the wait — expand "How this answer was retrieved" while it runs.**

*No preamble here — the hook already explained why this is the question. Type it and
go straight to the retrieval panel.*

> "Underneath I can see exactly what retrieval did — the similarity score, both
> retrieval paths, and the reranking. And notice some of these results weren't found
> by both paths. That's precisely why I kept semantic and lexical retrieval
> together."

**When the answer lands:** read one sentence, scroll to Sources, click a citation.

> "Every claim cited, and each one opens the actual posting."

Do not read five job names.

---

## 1:22 – 1:42 — Useful query 2 *(reqs 1, 4)*

**Type:** `Which roles require a security clearance?`

**MEASURED:** answers, `top_sim 0.653`, 2 citations, dense 3 · bm25 5 — the answer
names **TS/SCI** literally.

> "A completely different kind of requirement. And look —" **[point at TS/SCI]** "—
> **TS/SCI.** Rare literal strings like that are where lexical search complements
> semantic search — which is why I kept both. Retrieval here isn't tuned to one demo
> query."

Point at the acronym. Don't read the paragraph.

---

## 1:42 – 1:52 — The easy refusal *(req 4)*

**Click the example:** `What is a good recipe for sourdough bread?`

> "Now something deliberately unrelated. Similarity drops to **0.416**, below my
> 0.60 threshold — the graph exits before the model is ever called. Cheap and
> deterministic."

**Then, the transition that sets up the whole video:**

> "That's the easy refusal. The next one is where this project got interesting."

Ten seconds. Move.

---

## 1:52 – 2:28 — ★ The refusal that matters. **Slow down.** *(req 4)*

**Click the example:** `How many people applied to this job?`

> "This sounds like a perfectly reasonable job-search question."

**When it lands:**

> "And look what happened. Similarity is **0.618 — above my 0.60 threshold.** So
> retrieval says: this looks relevant. Guard one waves it straight through.
>
> But no job posting contains applicant counts. The model reads the retrieved text,
> sees the answer isn't in there, and refuses anyway."

**Pause. Then, slowly:**

> "**Similarity measures relevance. It does not measure answerability.**
>
> Which means a similarity threshold, on its own, cannot be a hallucination guard."

---

## 2:28 – 3:05 — What the evaluation proved wrong *(req 2)*

**Screen: `eval/results/eval-<latest>.md`.**

> "Rather than tune this by feel, I built a 20-question evaluation set across eight
> configurations — answerable questions, on-topic questions the corpus can't answer,
> and genuinely off-domain ones.
>
> Hybrid beat dense, reranking improved every configuration, and section-aware
> chunking won — against LangChain's standard splitter, not a strawman.
>
> But the finding I'd put my name on is a negative one. I assumed similarity could
> separate answerable questions from unanswerable ones. **That is the entire premise
> of guard one. The gap was 0.084.** Too small to gate on.
>
> With genuinely off-domain questions, that gap jumps to **0.203** — two and a half times
> wider. So guard one isn't weak. It's *structurally* blind to questions that are
> on-topic and unanswerable. An HR bot retrieves the parental-leave policy perfectly
> when you ask for a number the policy never states.
>
> That measurement is what pushed me to layered refusal instead of trusting retrieval
> confidence."

---

## 3:05 – 3:30 — The independent judge *(req 2)*

**Screen: eval header showing `judge=Qwen3-235B (independent)`.**

> "I also caught a problem in my own evaluation. Originally the generator, the
> verifier and the judge were all the same model family — Llama grading Llama. That
> reported 98.5% faithfulness.
>
> So I moved both checking roles to Qwen. **My numbers got worse:** 97.9%
> faithfulness, refusal accuracy 95 down to 90. I kept the worse numbers, because
> they're the ones I can defend."

---

## 3:30 – 3:42 — The guard-3 disclosure *(honesty beat)*

> "One measurement I took today rather than planned: guard three costs **154 of 237
> seconds** end to end — 65% of latency, for a check that has never overturned an
> answer on this corpus. It's off in this recording for that reason. Guards one and
> two — the ones doing the refusing you just watched — are both live.
>
> That exposed a real production tradeoff: stronger verification against latency. For
> this demo it's behind a feature flag; in production I'd measure where that extra
> verification justifies its cost."

Fifteen seconds. Don't apologize; you're reporting a number and a tradeoff.

*Earlier drafts ended this beat with "guard three belongs behind a flag, not in the
hot path." That generalizes a production rule from one measurement on one degraded
endpoint — the exact move the rest of this video argues against. Say what the evidence
supports and no more.*

---

## 3:42 – 4:12 — AI coding tools *(req 3 — DO NOT SKIP)*

> "I used Claude heavily building this. My biggest lesson was that AI-generated code
> failing loudly was never the dangerous case.
>
> **Several of the most important mistakes ran perfectly and produced plausible
> results.** Boilerplate detection, substring matching, the retrieval threshold, even
> an assumption baked into my evaluation — all of them looked reasonable, all of them
> were confidently argued in comments, and a test suite would have passed every one.
>
> So my workflow changed. I stopped treating confident AI output as an answer and
> started treating it as a hypothesis to falsify with evaluation. That's how all
> seven got found."

---

## 4:12 – 4:30 — Close

**Screen: back to `docs/one-pager.html`.**

> "Fitly started as a way to search hundreds of job postings. What I learned was more
> useful than the search: **retrieving relevant evidence is not the same as having
> enough evidence to answer.**
>
> For a trustworthy RAG system, knowing when to say *I don't know* matters as much as
> finding the answer. Repo and write-up are linked below. Thanks."

---

## If you run long

Cut in this order: **query 2 (1:22)** → shorten the corpus half of the one-pager beat.

**Never cut:** the 0.618 moment, the 0.084 / 0.203 finding, the independent judge, or
the AI-tools section.

**The resume feature is deliberately not in this cut.** It costs zero API time, so if
your trimmed edit lands under 4:35 you can add 20 seconds after the judge beat:

> "Fitly applies the same evidence-first approach to resume matching. Rather than
> embedding a whole resume as one vector, it extracts the strongest role and skill
> signals and retrieves against those — from a keyword taxonomy, never an LLM,
> because ask a model to extract skills and it invents plausible ones. And it says a
> requirement is *not evidenced in your resume*, never *you lack it*. The refusal
> principle, pointed at a person."

## Do not say

- ❌ *"zero hallucinations"* — one claim in 47 was judged unsupported. Say **"zero
  missed refusals"** and **"97.9% claim-level faithfulness."** Different measurements,
  and a reviewer will know.
- ❌ *"live job postings"* — the corpus is a frozen snapshot, and every published
  number depends on that.
- ❌ *"every single thing I got wrong was AI-written"* — overclaims attribution.
- ❌ *"completely disjoint retrievers"* — measured overlap is 3 of 5.
- ❌ Defining Chroma, bge-small, embedding dimensions, or LlamaParse. The repo proves
  those. Spoken time is for tradeoffs visible on screen.

## Video description — paste at upload

> **Fitly — evidence-grounded job search**
> Week 2 · Track 2 (LangChain + LangGraph)
>
> RAG over 874 real job postings from 93 companies, built around one question: how
> does it know when the evidence is insufficient?
>
> Hybrid retrieval (dense + BM25, fused by Reciprocal Rank Fusion), cross-encoder
> reranking, LangGraph orchestration, three layered refusal guards. Evaluated across
> 8 configurations on 20 labelled questions: 97.9% claim-level faithfulness, 90%
> refusal accuracy, zero missed refusals, zero dangling citations — graded by a
> different model family than the one generating.
>
> The headline result is a negative one. Retrieval similarity measures *topic*, not
> *answerability*: separation between answerable and unanswerable-but-on-topic
> questions was 0.084, versus 0.203 for genuinely off-domain questions. Guard 1 is
> structurally blind to questions that are on-topic and unanswerable — which is why
> guard 2 exists.
>
> Stack: LangChain · LangGraph state machine with conditional retry · Chroma cosine
> retrieval · BGE-small-en-v1.5 local embeddings · BM25 · Reciprocal Rank Fusion
> (k=60) · MS MARCO MiniLM cross-encoder reranking · section-aware chunking vs.
> RecursiveCharacterTextSplitter · metadata pre-filtering across both retrieval
> paths · Streamlit · Nebius-hosted Llama-3.3-70B generation · Qwen3-235B
> independent verification and evaluation.
>
> Built by Phanindra Nalam — platform engineering & SRE.
> Repo: https://github.com/phanindranalam/fitly-rag
