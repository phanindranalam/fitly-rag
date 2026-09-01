# Cue card — the only thing on your second screen

Full wording lives in `VIDEO_SCRIPT.md`. **Don't read that while recording.** Glance here.

---

## The eleven beats

| # | do | first line — say this one exactly | numbers |
|---|---|---|---|
| 1 | app open, don't touch anything | "Most RAG demos show a system that knows the answer. I wanted one that knows when it **doesn't**." → then **the one-liner**: *"the hard problem isn't finding the right posting — it's knowing when the one you found doesn't answer the question."* | 850-odd |
| 2 | switch to terminal | "The interesting part is cleaning." | 312 vs 25 → **48.8%** |
| 3 | type Kubernetes question | *(talk over the wait)* "Two retrievers run side by side." | **0.727** · dense 3 · bm25 5 |
| 4 | expand panel while it loads | "**Two of these five came from BM25 alone.**" | 2 rows with a dash in dense |
| 5 | type security clearance | "And look — **TS/SCI**." | **0.653** · point at the acronym |
| 6 | type sourdough | "Something obviously unrelated." | **0.416** < 0.60 · model never called |
| 7 | click applied-count example | "Now the one I care about." | **0.618 — clears the threshold** |
| 8 | switch to eval markdown | "The finding I'd put my name on." | **0.013** → **0.203** |
| 9 | point at the judge line | "Originally I reported 98.5% faithful." | → **97.9%**, 95 → **90** |
| 10 | upload the pre-opened resume | "Same pipeline, different query." | keyword taxonomy, never an LLM |
| 11 | no screen change → close | "On tooling: I built this with Claude as a pair programmer." | "several of the most important mistakes" — **not** "every single one" |

**Closing line:** "97.9% claim-level faithfulness, zero missed refusals, judged by a different model family. But the more useful result is the one that made me change the architecture." → **never say "zero hallucinations"**

**If you only remember one sentence in the whole video, make it beat 6:**

> "0.618. It clears the threshold. Retrieval is confident — and it refuses anyway."

---

## Dead air — plan it, don't suffer it

The Kubernetes answer takes **~40 seconds** with guard 3 on. That is a long silence on camera. Talk through it:

> "While that runs — there's a third guard I haven't mentioned. After the model writes an answer, a separate model reads it back against the retrieved text and can overturn it. That's the extra few seconds you're watching right now. It's a different model family from the one generating, which matters, and I'll come back to why."

That fills the gap **and** sets up beat 8. If it returns early, stop mid-sentence and move on — nobody notices.

Shorter filler for the refusals (7–20s):

> "No generation call on this one at all — the guard fires before the model is ever contacted."

---

## If something breaks

- **Wrong answer / weird output** → "That's a real system, not a rehearsal." Move on. Do not restart.
- **Small stumble** → keep going. Never say "sorry, let me start over" *while recording*.
- **App errors out** → stop recording, `python ui_test.py`, fix, start fresh. Don't try to talk over a traceback.
- **Running long at beat 7** → skip the resume section entirely. Never cut beats 8 or 9.

---

## Clock check

| by | you should be at |
|---|---|
| 0:40 | typing the Kubernetes question |
| 1:20 | typing the security-clearance question |
| 1:52 | at the applied-count question |
| 2:25 | starting the evaluation section |
| 3:45 | starting AI tools |

**Past 2:40 and not at the evaluation? Drop the resume beat (10).**
