# Cue card — the only thing on your second screen

Full wording lives in `VIDEO_SCRIPT.md`. **Don't read that while recording.** Glance here.

**Recording with `VERIFY_ANSWERS=false`.** Set it before Streamlit launches; the flag is
read at import and you are not restarting after warm-up.

---

## The eleven beats

| # | screen | first line — say this one close to exactly | numbers |
|---|---|---|---|
| 1 | app, untouched | "Forty job tabs open and one Sunday to decide where to apply." → then **"I'm Phanindra. I come from platform engineering and SRE, so the question I actually wanted answered wasn't *find me more jobs*."** | 874 · 58 hours |
| 2 | one-pager, **F11** | "Here's the whole system — it's a RAG pipeline. In plain English: retrieval finds the evidence, the model reads it and decides what it can actually say." | 22 seconds, then move |
| 3 | type Kubernetes q | *(no preamble — beat 1 already set it up)* "Underneath I can see exactly what retrieval did." | **0.727** · dense 3 · bm25 5 |
| 4 | type clearance q | "And look — **TS/SCI.** Rare literal strings are where lexical search complements semantic search." | **0.653** · point, don't read |
| 5 | click sourdough | "Something deliberately unrelated." → **"That's the easy refusal. The next one is where this project got interesting."** | **0.416** < 0.60 · no model call |
| 6 | click applied-count ★ | "This sounds like a perfectly reasonable job-search question." | **0.618 — above the threshold** |
| 7 | eval markdown | "The finding I'd put my name on." | **0.084** → **0.203** · **2.4× wider** |
| 8 | eval judge header | "Originally I reported 98.5%." | → **97.9%**, refusal 95 → **90** |
| 9 | no change | "One measurement I took today rather than planned." | **154 of 237s** · 65% |
| 10 | no change | "I stopped treating confident AI output as an answer." | "several of the most important mistakes" |
| 11 | back to one-pager | "Retrieving relevant evidence is not the same as having enough evidence to answer." | — |

**If you remember one sentence in the whole video, make it beat 6:**

> "0.618. It clears the threshold. Retrieval is confident — and it refuses anyway."

**Then, slowly:** *"Similarity measures relevance. It does not measure answerability."*

---

## Dead air — three waits, ~83s each

Beats **3, 4 and 6** each cost about **83 seconds**. That is too long to talk through.
**Record straight through and trim each wait in Clipchamp**, with a caption on the cut:

> ⏱ 78s of model latency trimmed — Nebius endpoint degraded, 2026-09-02

Sourdough (beat 5) is instant — guard 1 fires before any API call.

---

## The nine numbers you must not get wrong

| say | never say |
|---|---|
| 97.9% claim-level faithfulness | "zero hallucinations" — 1 claim in 47 was unsupported |
| zero missed refusals | — |
| 0.084 on-topic, 0.203 off-domain, **2.4× wider** | "0.013" — different run, contradicts the repo |
| 874 postings, 93 companies | "live postings" — it's a frozen snapshot |
| 0.618 clears the 0.60 threshold | — |
| 154 of 237 seconds, 65% | "guard 3 doesn't belong in the hot path" — one measurement |
| refusal accuracy 95 → 90 | — |
| "several of the most important mistakes" | "every single one" |
| 48.8% boilerplate | — |

---

## If something breaks

- **Wrong answer / weird output** → "That's a real system, not a rehearsal." Move on.
- **Small stumble** → keep going. Never say "let me start over" *while recording*.
- **App errors out** → stop, `python ui_test.py`, fix, fresh take. Don't talk over a traceback.
- **Running long at beat 6** → cut beat 4. Never cut 6, 7, 8 or 10.

---

## Clock check — 4:36 spoken, ~4:51 after trim cards, 5:00 cap

| by | you should be |
|---|---|
| 0:28 | on the one-pager |
| 0:50 | typing the Kubernetes question |
| 1:42 | at sourdough |
| 1:52 | at the applied-count question |
| 2:28 | starting the evaluation |
| 3:42 | starting AI tools |

**Past 2:40 and not at the evaluation? Drop beat 4 in the edit.**
