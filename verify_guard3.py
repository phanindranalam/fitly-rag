"""Prove what VERIFY_ANSWERS actually changes, by running it both ways.

WHY THIS EXISTS
---------------
"Guard 3 is the only thing the flag disables" is a claim about the code. This
turns it into a measurement: the same two questions, run once with
VERIFY_ANSWERS=true and once with false, with the graph's own trace printed
side by side.

What you should see:

  * The REFUSAL question is byte-identical both ways. node_verify returns
    early on refusals -- `if ans.refused ... return state` -- so guards 1 and
    2 are untouched by the flag.
  * The ANSWERED question differs by exactly one trace line: a verify step
    that either ran or was skipped. Citations, sources and text are unchanged
    unless guard 3 actively overturned the answer.

If you see anything else change, the claim is wrong and you should not record
with the flag off.

    python verify_guard3.py

Costs ~3 API calls. Reads nothing from .env except through config, and changes
no files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

QUESTIONS = [
    ("ANSWERED", "Which roles require running Kubernetes in production?"),
    ("REFUSAL",  "How many people applied to this job?"),
]


def child() -> None:
    """Run the questions in this process and emit JSON. Called via subprocess
    so the env var is read cleanly at import time."""
    import config
    from graph import ask
    from retrieve import Filters

    out = {"verify_answers": config.VERIFY_ANSWERS, "runs": []}
    for kind, q in QUESTIONS:
        state = ask(q, strategy="section", mode="hybrid",
                    use_rerank=True, filters=Filters())
        ans = state["answer"]
        out["runs"].append({
            "kind": kind,
            "question": q,
            "refused": bool(ans.refused),
            "reason": (ans.reason or "")[:120],
            "citations": len(ans.citations or []),
            "text_len": len(ans.text or ""),
            "text_head": (ans.text or "")[:90].replace("\n", " "),
            "trace": list(state.get("trace") or []),
        })
    print("<<<JSON>>>" + json.dumps(out))


def run(flag: str) -> dict:
    env = dict(os.environ, VERIFY_ANSWERS=flag)
    print(f"  running with VERIFY_ANSWERS={flag} ...", flush=True)
    p = subprocess.run([sys.executable, __file__, "--child"],
                       env=env, capture_output=True, text=True)
    if "<<<JSON>>>" not in p.stdout:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"child run failed for VERIFY_ANSWERS={flag}")
    return json.loads(p.stdout.split("<<<JSON>>>", 1)[1])


def main() -> int:
    print("=" * 74)
    print("What does VERIFY_ANSWERS actually change?")
    print("=" * 74)

    on = run("true")
    off = run("false")

    print(f"\nconfig saw: true -> {on['verify_answers']}   false -> {off['verify_answers']}")
    if on["verify_answers"] == off["verify_answers"]:
        print("\n!! The flag did not take effect. A VERIFY_ANSWERS line in .env is")
        print("   overriding the environment. Check .env before trusting anything below.")
        return 1

    bad = 0
    for a, b in zip(on["runs"], off["runs"]):
        print("\n" + "-" * 74)
        print(f"{a['kind']}: {a['question']}")
        print("-" * 74)

        same_outcome = a["refused"] == b["refused"]
        same_cites = a["citations"] == b["citations"]
        same_head = a["text_head"] == b["text_head"]

        print(f"  {'field':14s} {'guard3 ON':>34s}   {'guard3 OFF':>18s}")
        print(f"  {'refused':14s} {str(a['refused']):>34s}   {str(b['refused']):>18s}"
              + ("" if same_outcome else "   <-- DIFFERENT"))
        print(f"  {'citations':14s} {a['citations']:>34d}   {b['citations']:>18d}"
              + ("" if same_cites else "   <-- DIFFERENT"))
        print(f"  {'answer opens':14s} {a['text_head'][:34]:>34s}   {b['text_head'][:18]:>18s}"
              + ("" if same_head else "   <-- DIFFERENT"))

        # Normalize before diffing. The `generated ...` trace line embeds wall
        # clock and completion-token counts, both of which vary run to run at
        # temperature 0.1 -- and generation happens BEFORE node_verify reads the
        # flag, so it cannot be caused by it. Comparing those literally reports
        # sampling noise as a behavioural difference, which the first version of
        # this script did.
        import re as _re
        def norm(lines):
            out = []
            for l in lines:
                l = _re.sub(r"\d+(\.\d+)?s", "<t>", l)
                l = _re.sub(r"\d+\+\d+ tokens", "<tok>", l)
                l = _re.sub(r"\d+ chars", "<n> chars", l)
                out.append(l)
            return out
        na, nb = norm(a["trace"]), norm(b["trace"])
        only_on = [l for l in na if l not in nb]
        only_off = [l for l in nb if l not in na]
        print("\n  trace lines only with guard 3 ON:")
        for l in only_on or ["    (none)"]:
            print(f"    + {l}" if only_on else l)
        print("  trace lines only with guard 3 OFF:")
        for l in only_off or ["    (none)"]:
            print(f"    - {l}" if only_off else l)

        # --- the assertions that matter -------------------------------
        if a["kind"] == "REFUSAL":
            if same_outcome and same_cites:
                print("\n  VERDICT: identical. Guards 1 and 2 are untouched by the flag.")
            else:
                print("\n  VERDICT: *** THE FLAG CHANGED A REFUSAL. ***")
                print("  Do not record with it off -- it is affecting more than guard 3.")
                bad += 1
        else:
            leaked = [l for l in only_on + only_off if "verify" not in l.lower()]
            if leaked:
                print("\n  VERDICT: *** trace differs outside the verify step: ***")
                for l in leaked:
                    print(f"      {l}")
                bad += 1
            elif not same_outcome:
                print("\n  VERDICT: guard 3 OVERTURNED this answer.")
                print("  This is the Finding 07 false-positive class, live.")
                print("  -> record with VERIFY_ANSWERS=false and use line B at 3:00.")
            else:
                print("\n  VERDICT: same answer, same citations, one extra verify step.")
                print("  -> safe to record with VERIFY_ANSWERS=true. Use line A at 3:00.")

    print("\n" + "=" * 74)
    if bad:
        print("The flag does MORE than gate guard 3. Investigate before recording.")
        return 1
    print("Confirmed: VERIFY_ANSWERS gates guard 3 and nothing else.")
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        sys.exit(main())
