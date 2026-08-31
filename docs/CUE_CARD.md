# Cue card — the only thing on your second screen

Full wording lives in `VIDEO_SCRIPT.md`. **Don't read that while recording.** Glance here.

---

## The eight beats

| # | do | first line — say this one exactly | numbers |
|---|---|---|---|
| 1 | app open, don't touch anything | "Most RAG demos show you a system that knows the answer. I wanted one that knows when it **doesn't**." | 874 · 93 |
| 2 | switch to terminal | "The interesting part is cleaning." | 312 vs 25 → **48.8%** |
| 3 | type Kubernetes question | "Cited. Each one opens the actual posting." | top sim **0.727** |
| 4 | expand retrieval panel | "This is what most demos hide." | dense 3 · bm25 5 · **2 rows BM25-only** |
| 5 | type sourdough | "Start with the easy one." | **0.416** < 0.60 · model never called |
| 6 | click applied-count example | "Now the interesting one." | **0.618 — clears the threshold** |
| 7 | switch to eval markdown | "But here's the finding I'd put my name on." | **0.013** → **0.203** |
| 8 | point at judge line | "My first evaluation said 98.5% faithful." | → **97.9%**, 95 → **90** |
| 9 | no screen change | "On tooling: I built this with Claude as a pair programmer." | — |
| 10 | — | "Zero hallucinations across twenty questions. Thanks." | — |

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
| 0:50 | typing the Kubernetes question |
| 2:15 | finishing the refusals |
| 3:00 | starting the independent judge |
| 4:15 | starting AI tools |

**Past 3:10 and not at beat 8? Skip the resume.**
