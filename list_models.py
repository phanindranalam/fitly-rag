#!/usr/bin/env python3
"""What models can this API key actually reach?

    python list_models.py
    python list_models.py --test          # also send one real request

Provider model lineups change constantly, and a model ID copied from a blog
post that has since been retired fails at generation time -- which in this
app means it fails in the middle of a demo. This asks the account itself.
"""

from __future__ import annotations

import argparse
import sys

import config


def list_models() -> list[str]:
    from openai import OpenAI

    client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY)
    return sorted(m.id for m in client.models.list().data)


def test_call(model: str) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=10,
    )
    u = resp.usage
    print(f"\n{model}\n  reply: {resp.choices[0].message.content!r}"
          f"\n  tokens: {u.prompt_tokens} in, {u.completion_tokens} out")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="Send one tiny request to the configured model to prove the key works.")
    args = p.parse_args()

    if not config.NEBIUS_API_KEY:
        print("NEBIUS_API_KEY is not set. Put it in .env first.", file=sys.stderr)
        return 1

    try:
        models = list_models()
    except Exception as exc:
        print(f"Could not reach {config.NEBIUS_BASE_URL}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    chat = [m for m in models if not any(k in m.lower() for k in ("bge", "embed", "e5-"))]
    embed = [m for m in models if m not in chat]

    print(f"{len(models)} model(s) reachable with this key.\n")
    print("Generation:")
    for m in chat:
        mark = "  <- configured" if m == config.NEBIUS_MODEL else ""
        print(f"  {m}{mark}")
    print("\nEmbeddings (only used if you set EMBED_PROVIDER=nebius; default is local):")
    for m in embed:
        print(f"  {m}")

    if config.NEBIUS_MODEL not in models:
        print(f"\nWARNING: NEBIUS_MODEL={config.NEBIUS_MODEL!r} is NOT in this list. "
              f"Generation will fail. Pick one from above and set it in .env.")

    if args.test:
        test_call(config.NEBIUS_MODEL if config.NEBIUS_MODEL in models else chat[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
