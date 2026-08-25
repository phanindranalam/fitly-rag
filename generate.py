"""Grounded generation with citations, and the refusal path.

THE REFUSAL IS THE FEATURE
--------------------------
Designed before the happy path, as the brief instructs. A RAG app that
invents an answer when retrieval fails is strictly worse than one that says
"nothing in the corpus covers this", because a confident wrong answer costs
the user a wasted application and costs the product its credibility. There
are two independent guards:

  1. Retrieval-side: if nothing clears MIN_SIM, we never call the model.
     Cheaper, faster and impossible to talk the model out of.
  2. Prompt-side: the model is told to answer only from context and to emit
     INSUFFICIENT_CONTEXT when it can't. This catches the case where
     retrieval returns something confidently but it's about the wrong thing.

Guard 1 catches "no results". Guard 2 catches "wrong results". Both are
needed, and the eval measures each separately.

CITATIONS
---------
The model cites by bracketed index into the numbered context, [1], [2],
which is then resolved back to a company, role and URL in code. Asking for
free-text citations invites the model to invent plausible-looking company
names; asking for an integer it cannot invent an integer that maps to
nothing, and any out-of-range index is caught and dropped in
extract_citations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import config
from retrieve import Hit, RetrievalResult

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = """You answer questions about job postings using ONLY the numbered context provided.

Rules, in priority order:

1. Every factual claim must be supported by the context. Cite the source with a bracketed number like [2] immediately after the claim. A sentence with a fact and no citation is a failure.
2. If the context does not contain enough information to answer, reply with exactly INSUFFICIENT_CONTEXT and nothing else. Do not guess, do not generalize from world knowledge, do not say what is "typically" true. It is always better to refuse than to be plausibly wrong.
3. Never invent company names, job titles, salary figures, requirements or locations. If a number is not in the context, it does not exist.
4. Quote sparingly and paraphrase mostly, but stay close to the source wording for requirements and compensation, where precision matters more than style.
5. Be concise. Three to six sentences unless the question genuinely needs a list.

You are talking to a job seeker deciding where to spend an evening applying. Being useful means being accurate about what a role actually asks for."""

FIT_PROMPT = """You are comparing one person's background against job postings.

Rules, in priority order:

1. Only claim the candidate has experience that appears in their RESUME below. Only claim a role requires something that appears in the numbered CONTEXT. Cite postings with [n].
2. Absence of a keyword in the resume is NOT evidence the person lacks the skill. Say "not evidenced in your resume", never "you lack" or "you don't have".
3. If the context does not support a comparison, reply with exactly INSUFFICIENT_CONTEXT.
4. Never invent requirements, never invent resume content, never estimate a fit percentage. You are explaining overlap and gaps, not scoring.
5. Structure: what lines up, what does not, and one sentence on whether it is worth applying. Six sentences maximum.

RESUME:
{resume}"""


@dataclass
class Answer:
    text: str
    refused: bool = False
    reason: str = ""
    citations: list[Hit] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    strategy: str = ""
    # Guard 3 bookkeeping. `verified` False means the check did not run --
    # never conflate "not checked" with "checked and clean".
    verified: bool = False
    claims_total: int = 0
    claims_unsupported: int = 0
    worst_claim: str = ""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _call_nebius(system: str, user: str, model: str | None = None) -> tuple[str, int, int]:
    from openai import OpenAI

    client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY)
    resp = client.chat.completions.create(
        model=model or config.NEBIUS_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))


def _call_anthropic(system: str, user: str, model: str | None = None) -> tuple[str, int, int]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def call_llm(system: str, user: str, model: str | None = None) -> tuple[str, int, int]:
    """`model` overrides the configured generator. Used by the verifier and
    the eval judge so a checking role never shares a model with the thing it
    is checking."""
    if config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(system, user, model)
    return _call_nebius(system, user, model)


# ---------------------------------------------------------------------------
# Context assembly and citation resolution
# ---------------------------------------------------------------------------

def format_context(hits: list[Hit]) -> str:
    """Numbered blocks. The number is the citation handle, so it has to be
    unmissable and stable."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] {h.citation()}"
        if h.location:
            header += f" | {h.location}"
        if h.compensation:
            header += f" | pay: {h.compensation}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


_CITE_RE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str, hits: list[Hit]) -> list[Hit]:
    """Resolve [n] back to hits, dropping any index the model invented.

    Out-of-range indices are silently discarded rather than surfaced,
    because a dangling citation is a generation bug, not something the user
    can act on. The eval counts them separately so the bug stays visible to
    us.
    """
    seen, out = set(), []
    for m in _CITE_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= len(hits) and idx not in seen:
            seen.add(idx)
            out.append(hits[idx - 1])
    return out


def count_bad_citations(text: str, hits: list[Hit]) -> int:
    return sum(1 for m in _CITE_RE.finditer(text) if not 1 <= int(m.group(1)) <= len(hits))


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def answer(question: str, result: RetrievalResult, resume: str | None = None) -> Answer:
    import time

    # Guard 1: nothing retrieved well enough. Never reaches the model.
    if not result.confident:
        top = f"{result.top_sim:.3f}"
        return Answer(
            text=("I could not find anything in the indexed postings that answers this. "
                  "Nothing retrieved cleared the confidence threshold, so rather than "
                  "guess, I would rather tell you the corpus does not cover it."),
            refused=True,
            reason=f"top similarity {top} < MIN_SIM {config.MIN_SIM}",
            hits=result.hits,
            strategy=result.strategy,
        )

    context = format_context(result.hits)
    if resume and resume.strip():
        system = FIT_PROMPT.format(resume=resume.strip()[:6000])
        user = f"CONTEXT:\n\n{context}\n\nQUESTION: {question}"
    else:
        system = SYSTEM_PROMPT
        user = f"CONTEXT:\n\n{context}\n\nQUESTION: {question}"

    t0 = time.time()
    text, ptok, ctok = call_llm(system, user)
    latency = time.time() - t0
    text = (text or "").strip()

    # Guard 2: the model saw context and judged it insufficient.
    if REFUSAL_TOKEN in text:
        return Answer(
            text=("The postings I retrieved do not actually answer this. They came back "
                  "as related, but none of them contain the information needed, so I am "
                  "not going to fill the gap with a guess."),
            refused=True,
            reason="model returned INSUFFICIENT_CONTEXT despite retrieval clearing the threshold",
            hits=result.hits,
            latency_s=latency,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            strategy=result.strategy,
        )

    return Answer(
        text=text,
        citations=extract_citations(text, result.hits),
        hits=result.hits,
        latency_s=latency,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        strategy=result.strategy,
    )


# ---------------------------------------------------------------------------
# Guard 3: post-generation claim verification
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """You check whether an ANSWER is supported by its CONTEXT.

Read the CONTEXT (numbered source passages) and the ANSWER. For every factual claim the ANSWER makes, decide whether the CONTEXT states it.

Rules:
- A claim is UNSUPPORTED if the context does not state it, even if it is true in the real world.
- Reasonable paraphrase of the context is supported. Added specifics -- numbers, durations, names, requirements not present in the context -- are not.
- Framing that asserts nothing ("here are some options") is not a claim.
- Be strict about numbers. If the answer gives a figure the context does not contain, that is unsupported.

Reply with ONLY a JSON object and nothing else:
{"total_claims": <int>, "unsupported_claims": <int>, "worst": "<the least supported sentence, or empty string>"}"""


@dataclass
class Verdict:
    total: int = 0
    unsupported: int = 0
    worst: str = ""
    ran: bool = False
    error: str = ""

    @property
    def ratio(self) -> float:
        return self.unsupported / self.total if self.total else 0.0


def verify_answer(context: str, answer_text: str) -> Verdict:
    """Ask the model whether its own answer is supported by the context.

    WHY THIS EXISTS, AND WHY IT IS A THIRD GUARD RATHER THAN A BETTER SECOND
    -----------------------------------------------------------------------
    The eval caught exactly one hallucination in twenty questions, and it was
    the one predicted to be hardest: "what is the company's parental leave
    policy in weeks?". Benefits language IS in the corpus, so retrieval
    returned benefits chunks at high similarity and guard 1 passed them
    through by design. Guard 2 -- the model instructed to answer only from
    context -- then saw plausible benefits text and answered anyway.

    That failure is structural, not a prompting accident. Guard 2 asks the
    model to judge sufficiency WHILE it is composing an answer, which is the
    moment it is least able to. Guard 3 asks a separate call to judge the
    finished text against the context, with no obligation to be helpful.

    This is the same check the eval already ran as its faithfulness judge.
    Moving it into the request path costs one extra API call per answered
    question -- the honest trade for a measured failure mode.
    """
    try:
        raw, _, _ = call_llm(VERIFY_PROMPT,
                             f"CONTEXT:\n\n{context}\n\n---\n\nANSWER:\n\n{answer_text}",
                             model=config.VERIFY_MODEL)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        return Verdict(total=int(data.get("total_claims", 0)),
                       unsupported=int(data.get("unsupported_claims", 0)),
                       worst=str(data.get("worst", "")), ran=True)
    except Exception as exc:
        # A verifier that fails must not silently approve. It reports that it
        # did not run, and the caller keeps the answer with that fact visible
        # rather than pretending it was checked.
        return Verdict(ran=False, error=f"{type(exc).__name__}: {exc}")


def apply_verdict(ans: "Answer", verdict: Verdict) -> "Answer":
    """Decide what to do about an answer the verifier doubts.

    Refusing on ANY unsupported claim would be too blunt: the eval measured
    one unsupported claim in sixty-three, and throwing away a five-claim
    answer over one weak sentence costs more than it saves. Refusing when
    MOST of the answer is unsupported catches the failure this guard was
    built for -- the parental-leave answer was unsupported end to end -- while
    leaving a mostly-grounded answer intact with the weak claim recorded.
    """
    ans.verified = verdict.ran
    ans.claims_total = verdict.total
    ans.claims_unsupported = verdict.unsupported
    ans.worst_claim = verdict.worst

    if not verdict.ran or verdict.total == 0:
        return ans
    if verdict.ratio > config.VERIFY_MAX_UNSUPPORTED:
        return Answer(
            text=("I retrieved postings that looked relevant, but when I checked my own "
                  "answer against them, most of what I had written was not actually "
                  "supported by the text. Rather than hand you that, I would rather say "
                  "the postings do not answer this."),
            refused=True,
            reason=(f"guard 3: {verdict.unsupported}/{verdict.total} claims unsupported "
                    f"(limit {config.VERIFY_MAX_UNSUPPORTED:.0%})"),
            hits=ans.hits,
            latency_s=ans.latency_s,
            prompt_tokens=ans.prompt_tokens,
            completion_tokens=ans.completion_tokens,
            strategy=ans.strategy,
            verified=True,
            claims_total=verdict.total,
            claims_unsupported=verdict.unsupported,
            worst_claim=verdict.worst,
        )
    return ans
