"""Endpoint and model configuration, in one place.

DESIGN DECISION: EMBEDDINGS RUN LOCALLY, GENERATION RUNS HOSTED
---------------------------------------------------------------
Embeddings are the high-volume, low-value call: this corpus produces roughly
6,000 chunks, and the chunking comparison requires embedding the whole thing
TWICE. Paying an API round trip per chunk buys nothing, costs credits, adds
minutes to every re-index, and puts a network dependency in the middle of a
live demo. bge-small runs on CPU in about two minutes for the whole corpus
and never fails.

Generation is the opposite: one call per question, and the quality of that
call is the product. That is worth a hosted model and worth the credits.

So the pipeline has exactly one API dependency, exercised once per answer.
If Nebius is down mid-demo, retrieval still works and you can show the
retrieved chunks; only the prose is missing.

SWITCHING
---------
Set LLM_PROVIDER=nebius (default) or anthropic. Everything else follows.
To use hosted embeddings instead of local, set EMBED_PROVIDER=nebius and
EMBED_MODEL to a model ID from `python list_models.py`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nebius").strip().lower()

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
# Llama 3.3 70B: strong enough to follow a strict grounding instruction and
# refuse when the context doesn't support an answer, which is the behaviour
# this whole app is judged on. The 8B variant is ~20x cheaper and noticeably
# worse at refusing, which is exactly the wrong place to economize.
#
# This default was `Meta-Llama-3.1-70B-Instruct` for most of the build, which
# does not exist on Token Factory and 404s on the first generation call. Every
# result in the writeup was produced with 3.3, set via .env -- so the default
# was wrong for anyone who cloned the repo and never for the person who wrote
# it. `python list_models.py` lists what the endpoint actually serves; it
# existed the whole time and would have caught this on day one.
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# INDEPENDENT EVALUATION
# ----------------------
# The generator, the guard-3 verifier and the eval's faithfulness judge were
# all originally the same model. That is the single largest methodological
# weakness in this project: a model is a soft grader of its own output, and
# three roles filled by one model share one set of blind spots. "98.5%
# faithful" then means "Llama agreed with Llama", which is not a measurement.
#
# Both checking roles now default to a DIFFERENT model family from the
# generator. Expect the faithfulness score to drop. That is the point -- a
# believable 94% from an independent judge is worth more than a suspicious
# 100% from a self-assessment.
#
# Set either to the generator's model to reproduce the old (weaker) setup and
# measure the difference.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# Low but not zero. Zero makes the model repeat the source text verbatim
# rather than synthesizing, which reads worse and doesn't actually improve
# faithfulness.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").strip().lower()

# 384 dimensions, matched deliberately to ~400-token chunks. A 1024-dim model
# on chunks this small is capacity you pay for and cannot use; a 384-dim model
# on 2000-token chunks would lose signal. Size the model to the chunk.
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
NEBIUS_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-en-icl")

# bge models are trained with an asymmetric query prefix: queries get an
# instruction, documents do not. Skipping this costs a few points of recall
# for free, and it's the single most-missed detail with bge.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# CPU embedding speed knobs. These exist because the first real index run on a
# Windows laptop clocked ~1 chunk/second -- roughly 100x slower than this model
# should be -- which turned a four-minute step into a two-hour one. Defaults
# below are the fast configuration; bench_embed.py measures them so the choice
# is made from a number rather than a hunch.
#
# threads: 0 means "use every core". torch does not always pick this itself.
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "0"))
# backend: "onnx" is usually several times faster than "torch" for a small
# encoder on CPU, and onnxruntime is already installed as a chromadb
# dependency. Falls back to torch automatically if the export fails.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "onnx").strip().lower()
# Token window. bge-small defaults to 512; chunks here run ~350 tokens, and
# attention cost is quadratic in length, so trimming the padding is free speed.
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "384"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.jsonl")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")

# Retrieve wide, then narrow. 20 candidates per retriever gives fusion
# something to work with; 5 is what the generator sees, because past that
# the context window fills with near-duplicates and answer quality drops.
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))

# Refusal threshold, in COSINE SIMILARITY between the query and the best
# retrieved chunk -- not in fused RRF score. RRF is built from ranks, so its
# top value is roughly the same for a perfect answer and for the
# least-irrelevant chunk in a corpus that has nothing to say; thresholding it
# measures retriever agreement, not relevance. See RetrievalResult.confident.
#
# 0.60, and that number came from `--sweep-threshold` walking the labelled set,
# not from a hunch. The default was 0.45 -- the pre-measurement guess -- long
# after the sweep had replaced it, which meant a fresh clone did not reproduce
# any published result. Every figure in the writeup assumes 0.60: the applied-
# count question sits at 0.618 and is refused by guard 2 *because* it clears
# this line. At 0.45 that finding does not exist.
#
# Re-run the sweep if the corpus changes; the right threshold is a property of
# the corpus and the embedding model, not a constant.
MIN_SIM = float(os.getenv("MIN_SIM", "0.60"))

# Guard 3: verify the finished answer against its context before returning it.
# Costs one extra API call per answered question and roughly doubles latency on
# the generation half. Added because the eval found the one hallucination in
# twenty questions was structurally invisible to guards 1 and 2 -- see
# generate.verify_answer. Set VERIFY_ANSWERS=false to measure the difference.
VERIFY_ANSWERS = os.getenv("VERIFY_ANSWERS", "true").strip().lower() != "false"

# Refuse when MORE than this share of the answer's claims are unsupported.
# Not zero: one weak sentence in a five-claim answer is worth flagging, not
# worth discarding the other four. The parental-leave failure this guard was
# built for was unsupported end to end, which any limit below 1.0 catches.
VERIFY_MAX_UNSUPPORTED = float(os.getenv("VERIFY_MAX_UNSUPPORTED", "0.34"))

# Latency ceiling declared up front so retrieval doesn't get over-engineered.
LATENCY_BUDGET_S = float(os.getenv("LATENCY_BUDGET_S", "5.0"))


def describe() -> str:
    """One-line config summary, printed by every entry point so a run's
    settings end up in the terminal transcript you record for the demo."""
    llm = (f"nebius:{NEBIUS_MODEL}" if LLM_PROVIDER == "nebius"
           else f"anthropic:{ANTHROPIC_MODEL}")
    emb = (f"local:{LOCAL_EMBED_MODEL}" if EMBED_PROVIDER == "local"
           else f"nebius:{NEBIUS_EMBED_MODEL}")
    judge = JUDGE_MODEL.split("/")[-1]
    indep = "independent" if JUDGE_MODEL != NEBIUS_MODEL else "SAME AS GENERATOR"
    return (f"llm={llm} embed={emb} top_k={TOP_K_RETRIEVE}->{TOP_K_CONTEXT} "
            f"min_sim={MIN_SIM} judge={judge} ({indep})")


def check() -> list[str]:
    """Return a list of configuration problems, empty if we're good. Called
    at startup so a missing key fails immediately with a clear message
    rather than deep inside a retrieval call."""
    problems = []
    if LLM_PROVIDER == "nebius" and not NEBIUS_API_KEY:
        problems.append("NEBIUS_API_KEY is not set. Put it in .env")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        problems.append("ANTHROPIC_API_KEY is not set. Put it in .env")
    if LLM_PROVIDER not in ("nebius", "anthropic"):
        problems.append(f"LLM_PROVIDER must be nebius or anthropic, got {LLM_PROVIDER!r}")
    if EMBED_PROVIDER == "nebius" and not NEBIUS_API_KEY:
        problems.append("EMBED_PROVIDER=nebius needs NEBIUS_API_KEY")
    return problems
