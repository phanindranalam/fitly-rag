# Demo video — word-for-word, 4:40 target / 5:00 hard cap

**Capability first, failure-awareness second.** A reviewer should leave thinking
*"I'd use this"* before they think *"clever guardrails."* That ordering is the
main change in this version.

**The handout scores four things:**

1. Walk through your application
2. Explain what you built
3. **Describe how you used AI coding tools** ← explicitly required, easy to lose to the clock
4. Demonstrate the final result **live**

---

## Before you hit record

### Test the second query first — it is the only unmeasured thing in this script

```cmd
python -c "from graph import ask; from retrieve import Filters; s=ask('Which roles require a security clearance?', strategy='section', mode='hybrid', use_rerank=True, filters=Filters()); a=s['answer']; r=s['retrieval']; print('refused', a.refused, '| top_sim', round(r.top_sim,3), '| citations', len(a.citations or [])); print((a.text or a.reason)[:300]); print('dense', sum(1 for h in r.hits if h.dense_rank is not None), 'bm25', sum(1 for h in r.hits if h.sparse_rank is not None))"
```

- **Answers with citations** → keep beat 4, and check whether BM25 carried it.
- **Refuses, or the answer is thin** → **cut beat 4 entirely** and give those 20
  seconds back to the evaluation. Do not demo an unmeasured query on camera.

### Guard 3 stays ON

`VERIFY_ANSWERS=true`. Last three runs: `verify ok: 0/5 claims unsupported` on the
Kubernetes answer. Saying "I built three guards" and then disabling one is an
asterisk you don't need.

**Two answered questions × ~40s of verification is ~80 seconds of spinner.** That
is fine, because both waits are scripted:

- **Wait 1** → you're reading the retrieval panel out loud anyway.
- **Wait 2** → that's where the guard-3 explanation goes. Latency becomes evidence
  of the architecture instead of dead air.

### Checklist

- [ ] `python ui_test.py` → 0 failed
- [ ] Streamlit running, **warmed with a throwaway query**, browser at **125%**
- [ ] Sidebar: Anywhere / no work-mode / **section** / **hybrid** / rerank **on**
- [ ] Tab 2: `eval/results/eval-<latest>.md` · Tab 3: the whiteboard artifact
- [ ] Terminal on the boilerplate stats
- [ ] **`data\test_resumes\platform_sre.pdf` open in the file picker already**, or
      in a folder containing exactly one file — live file-browsing is a classic
      30-second demo failure and there is nothing to learn from watching it
- [ ] Notifications off, Slack quit

**Word budget: ~640 spoken words** → about 3:45 at a normal clip, leaving ~55
seconds for typing and model latency.

---

## 0:00 – 0:18 — Hook *(req 2)*

**Screen: the app, already open. No title card.**

> "Most RAG demos show you a system that knows the answer. I wanted one that knows when it *doesn't* — and can prove it.
>
> This is Fitly. 874 real job postings, 93 companies, every answer cited back to the posting it came from."

---

## 0:18 – 0:40 — The corpus *(req 2)*

**Screen: terminal, boilerplate stats.**

> "Postings come from Greenhouse, Ashby and Lever board APIs — the employer's own text.
>
> The interesting part is cleaning. Job ads are mostly *not about the job*. So this fingerprints every paragraph and drops what repeats — per employer, because Stripe's EEO statement names Stripe.
>
> **48.8%.** Nearly half of what employers write is text they put in every other ad."

---

## 0:40 – 1:20 — Useful query 1 *(reqs 1, 4)*

**Type:** `Which roles require running Kubernetes in production?`

**Talk over the wait — expand "How this answer was retrieved" while it runs.**

> "Two retrievers run side by side. Embeddings for meaning, BM25 for exact strings — they fail on opposite inputs, so I fuse them by *rank*, not by score.
>
> And look —" **[point at the dense column]** "— **two of these five came from BM25 alone.** Dense retrieval never surfaced them."

**When the answer lands:** read one sentence, scroll to Sources, click a citation.

> "Every claim cited, and each one opens the actual posting."

---

## 1:20 – 1:40 — Useful query 2 *(reqs 1, 4)* — **CUT IF THE TEST ABOVE FAILED**

**Type:** `Which roles require a security clearance?`

> "Completely different requirement — and this is where the lexical half earns its place. Literal strings like TS/SCI are exactly what embeddings blur together."

**Don't read the answer.** One glance at the panel, then move on. The point is that
retrieval isn't tuned to one demo query.

---

## 1:40 – 1:52 — The obvious refusal *(req 4)*

**Type:** `What is a good recipe for sourdough bread?`

> "First, something obviously unrelated. Similarity **0.416**, threshold 0.60 — the graph exits before the model is ever called. Cheap and deterministic."

Ten seconds. Move on.

---

## 1:52 – 2:25 — The refusal that matters. **Slow down here.** *(req 4)*

**Click the example:** `How many people applied to this job?`

> "Now the one I care about. Application counts don't exist in job postings — but that's exactly the number a model will happily estimate.
>
> Watch the similarity: **0.618. It clears my threshold.** Retrieval is confident. Guard one waves it straight through. The model reads the retrieved text and refuses anyway."

**Pause. Then, slowly:**

> "**Similarity tells me whether I retrieved something related to the question. It does not tell me whether that text contains the answer.**
>
> Which is why retrieval confidence alone cannot be my hallucination guard."

---

## 2:25 – 3:05 — What the evaluation proved wrong *(req 2)*

**Switch to the eval markdown.**

> "Twenty labelled questions, eight configurations. Hybrid beat dense at both chunk sizes, reranking improved every config, section-aware chunking won — against LangChain's standard splitter, not a strawman.
>
> But here's the finding I'd put my name on. I assumed similarity could separate answerable questions from unanswerable ones. **That's the entire premise of guard one. The gap was 0.013.** Nothing.
>
> Because every trap I wrote was still *about jobs.* With genuinely off-domain questions the gap jumps to 0.203 — two and a half times wider.
>
> So guard one is *structurally* blind to on-topic questions it can't answer. An HR bot retrieves the parental-leave policy perfectly when you ask for a number the policy never states."

---

## 3:05 – 3:30 — The independent judge *(req 2)*

**Screen: eval header showing `judge=Qwen3-235B (independent)`.**

> "Originally I reported 98.5% faithfulness. Then I noticed the generator, the verifier and the judge were all the same model family — Llama grading Llama. So I moved both checking roles to Qwen.
>
> **My numbers got worse.** 97.9% faithfulness, refusal accuracy 95 down to 90. I kept the worse numbers, because they're the ones I can defend.
>
> That verifier also exposed a structural limit: aggregate questions can't be validated from five chunks when the claim is about the whole corpus."

---

## 3:30 – 3:45 — Resume matching *(reqs 1, 4)*

**Upload the pre-positioned resume.**

> "Same pipeline, different query. Skills come from a keyword taxonomy, never an LLM — ask a model to extract skills and it invents plausible ones. And the prompt says **'not evidenced in your resume,' never 'you lack.'** The refusal principle, pointed at a person."

---

## 3:45 – 4:20 — AI coding tools *(req 3 — DO NOT SKIP)*

> "On tooling: I built this with Claude as a pair programmer. Fast at scaffolding — and most valuably, it wrote the *argument* for each decision into the docstrings.
>
> **Several of the most important mistakes came from AI-written code or AI-assisted analysis — and none of them crashed.** All confidently argued in comments. A test suite would have passed every one.
>
> That changed how I used the tool. Every confident architectural claim became a hypothesis I tried to falsify — which is how all seven of them got found."

---

## 4:20 – 4:40 — Close

> "The final system reaches **97.9% claim-level faithfulness with zero missed refusals**, judged by a model from a different family than the one writing the answers.
>
> But the more useful result is the one that made me change the architecture: **retrieval confidence stops being evidence exactly where the question stops being obvious.** Repo and write-up are linked below. Thanks."

---

## If you run long

Cut in this order: **query 2 (1:20)** → **resume (3:30)** → shorten the corpus beat.

**Never cut:** the 0.618 moment, the 0.013 finding, the independent judge, or the
AI-tools section.

## Do not say

- ❌ *"zero hallucinations"* — one claim in 47 was judged unsupported. Say **"zero missed refusals"** and **"97.9% claim-level faithfulness"**. They're different measurements and a reviewer will know.
- ❌ *"every single thing I got wrong was AI-written"* — overclaims attribution and sounds like blame.
- ❌ *"completely disjoint retrievers"* — measured overlap is 3 of 5.
- ❌ Defining Chroma, bge-small, embedding dimensions, or LlamaParse. The repo proves those. Spoken time is for tradeoffs that are visible on screen.

## Video description — paste at upload

> **Fitly RAG — evidence-grounded job intelligence**
> Week 2 · Track 2 (LangChain + LangGraph)
>
> RAG over 874 live job postings from 93 companies, built around one question: how does it know when the evidence is insufficient?
>
> Hybrid retrieval (dense + BM25, fused by Reciprocal Rank Fusion), cross-encoder reranking, LangGraph orchestration, three independent refusal guards. Evaluated across 8 configurations on 20 labelled questions: 97.9% claim-level faithfulness, zero missed refusals, zero dangling citations — graded by a different model family than the one generating.
>
> The headline result is a negative one. Retrieval similarity measures *topic*, not *answerability*: separation between answerable and unanswerable-but-on-topic questions was 0.013, versus 0.203 for genuinely off-domain questions. Guard 1 is structurally blind to questions that are on-topic and unanswerable — which is why guard 2 exists.
>
> Repo: https://github.com/phanindranalam/fitly-rag
