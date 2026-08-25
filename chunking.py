"""Two chunking strategies, so the comparison is an experiment rather than
an opinion.

THE COMPARISON THIS FILE EXISTS TO SUPPORT
------------------------------------------
Strategy A, fixed-size, is the default everyone reaches for: split every
document into equal character windows with a little overlap. It is
structure-blind by definition.

Strategy B, section-aware, uses the fact that job postings are already
structured documents. Every posting has the same skeleton -- what the
company does, what you'll do, what they require, what's nice to have, what
they pay -- announced by headings. Splitting on those boundaries means a
"requirements" chunk contains requirements and nothing else, so a query
about requirements retrieves a chunk that is entirely about requirements
rather than one that is 40% company blurb.

Both produce chunks of comparable size, so the comparison isolates the
BOUNDARY decision rather than confounding it with chunk length.

THE CONTEXT-PREFIX DETAIL
-------------------------
Both strategies prepend "Company - Title (Section)" to every chunk. This
matters more than it looks. A raw requirements bullet list contains no
company name and no job title, so a retriever can find it and the generator
still cannot say WHICH job it came from -- and an uncitable answer is
useless in a product whose whole promise is citations. Prefixing also gives
the embedding a little topical anchoring, which measurably helps when many
postings share near-identical requirement language.
"""

from __future__ import annotations

import re

# Chunk target in characters. ~1600 chars is roughly 400 tokens, matched to
# the 384-dim embedding model (see config.py): small enough that one chunk is
# about one idea, large enough that a requirements list survives intact.
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200

# Section headings, in the wording real postings actually use. Ordered
# longest-first at match time so "What you'll do" wins over "What".
SECTION_PATTERNS = [
    ("responsibilities", r"what you.?ll (?:do|be doing)|responsibilities|the role|about the role|"
                         r"your (?:role|impact|mission)|in this role|day to day|what the job involves"),
    ("requirements", r"requirements|qualifications|what (?:we.?re looking for|you.?ll (?:need|bring))|"
                     r"about you|who you are|minimum qualifications|basic qualifications|"
                     r"skills(?: and experience)?|experience required|you (?:have|should have)"),
    ("nice_to_have", r"nice to have|bonus|preferred qualifications|preferred|"
                     r"even better|plus(?:es)?|desirable|it.?s a plus"),
    ("compensation", r"compensation|salary|pay(?: range| transparency)?|what we offer|"
                     r"benefits|perks|total rewards"),
    ("about_company", r"about (?:us|the company|the team)|who we are|our (?:mission|story|team)|"
                      r"company overview"),
]

_HEADING_RE = re.compile(
    r"^\s*(?:[#*\-•]\s*)?(" + "|".join(p for _, p in SECTION_PATTERNS) + r")\s*:?\s*$",
    re.I | re.M,
)


def _section_for(heading: str) -> str:
    h = heading.lower()
    for name, pattern in SECTION_PATTERNS:
        if re.search(pattern, h, re.I):
            return name
    return "other"


def _prefix(doc: dict, section: str | None = None) -> str:
    """Self-describing header so a chunk can be cited without its neighbours."""
    parts = [doc.get("company") or "Unknown company", doc.get("title") or "Untitled role"]
    head = " - ".join(p for p in parts if p)
    loc = doc.get("location")
    if loc:
        head += f" ({loc})"
    if section and section != "other":
        head += f" [{section.replace('_', ' ')}]"
    return head


def _window(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph boundaries where possible, hard-cut only when a
    single paragraph exceeds the window. Hard-cutting mid-sentence is what
    makes naive fixed-size chunking produce unquotable fragments."""
    if len(text) <= size:
        return [text] if text.strip() else []

    out, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            out.append(current)
            # Carry the tail forward so a requirement split across the seam
            # appears in both chunks rather than neither.
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = para
        while len(current) > size:
            out.append(current[:size])
            current = current[size - overlap:]
    if current.strip():
        out.append(current)
    return out


# Fields that must survive chunking untouched, because retrieval filters on
# them. A chunk that loses its country is a chunk that cannot be excluded
# from a search for Atlanta roles, and pre-filtering silently degrades into
# no filtering at all -- which looks like a working app returning wrong
# answers, the worst failure shape there is.
CARRY_FIELDS = ("company", "title", "location", "url", "compensation",
                "posted_date", "geo_country", "geo_state", "geo_city",
                "work_mode", "remote_scope", "geo_confidence")


def _carry(doc: dict) -> dict:
    return {f: doc.get(f) for f in CARRY_FIELDS}


# ---------------------------------------------------------------------------
# Strategy A: fixed-size
# ---------------------------------------------------------------------------

def _split_fixed(text: str) -> list[str]:
    """LangChain's RecursiveCharacterTextSplitter, with a local fallback.

    WHY THE BASELINE USES A LIBRARY AND THE OTHER STRATEGY DOES NOT
    ---------------------------------------------------------------
    The point of strategy A is to be the thing everyone else would build. If
    the baseline is my own hand-rolled windowing, then "section-aware beat
    fixed-size" only ever means "my second idea beat my first idea", which is
    a much weaker claim than it sounds.

    RecursiveCharacterTextSplitter is the reference implementation: it tries
    paragraph breaks first, then lines, then sentences, then characters,
    backing off only as far as it must. Measuring against it means the
    comparison is against what a reader would actually have written.

    Strategy B stays hand-written because it encodes domain knowledge no
    general splitter has -- where a job posting's own section headings are.
    Using a library for the generic half and custom code for the specific
    half is the split that makes sense.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CHARS,
            chunk_overlap=CHUNK_OVERLAP,
            # Explicit rather than defaulted: this is the back-off order the
            # comparison depends on, so it should be visible here and not
            # buried in a library default that could change.
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:
        return _window(text)


def chunk_fixed(doc: dict) -> list[dict]:
    """Equal windows, structure-blind. The baseline, using the standard
    splitter so the comparison is against a reference implementation rather
    than against my own first attempt."""
    prefix = _prefix(doc)
    chunks = []
    for i, piece in enumerate(_split_fixed(doc["text"])):
        chunks.append({
            "id": f"{doc['id']}::fixed::{i}",
            "text": f"{prefix}\n\n{piece}",
            "raw": piece,
            "section": "unknown",
            "strategy": "fixed",
            "doc_id": doc["id"],
            **_carry(doc),
        })
    return chunks


# ---------------------------------------------------------------------------
# Strategy B: section-aware
# ---------------------------------------------------------------------------

def split_sections(text: str) -> list[tuple[str, str]]:
    """Returns [(section_name, body)]. Text before the first heading is
    'other', which is usually the intro paragraph."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("other", text)]

    sections = []
    if matches[0].start() > 0:
        lead = text[: matches[0].start()].strip()
        if lead:
            sections.append(("other", lead))

    for i, m in enumerate(matches):
        name = _section_for(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def chunk_sections(doc: dict) -> list[dict]:
    """Split on the posting's own structure, then window anything oversized.

    A section that fits stays whole even if it's short: a 300-character
    requirements list is a better retrieval unit than the same list padded
    out to 1600 characters with company boilerplate, because the padding is
    exactly what pulls the embedding away from the query.
    """
    prefix_base = _prefix(doc)
    chunks = []
    idx = 0
    for section, body in split_sections(doc["text"]):
        for piece in _window(body):
            header = _prefix(doc, section)
            chunks.append({
                "id": f"{doc['id']}::section::{idx}",
                "text": f"{header}\n\n{piece}",
                "raw": piece,
                "section": section,
                "strategy": "section",
                "doc_id": doc["id"],
                **_carry(doc),
            })
            idx += 1
    if not chunks:  # pathological posting with no usable body
        return chunk_fixed(doc)
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "section": chunk_sections}


# A chunk shorter than this is a bare heading or a stray fragment -- "Requirements:"
# on its own line. Both strategies produce them: the recursive splitter breaks on
# blank lines, and the section splitter can find a heading with almost nothing
# under it. They are pure noise in an index: too short to answer anything, and
# short text embeds to a vector that sits oddly close to everything.
#
# Applied identically to both strategies, which is the part that matters. A
# filter tuned to help one side of a comparison would invalidate the comparison.
MIN_CHUNK_CHARS = 80


def chunk_corpus(docs: list[dict], strategy: str) -> list[dict]:
    fn = STRATEGIES[strategy]
    out = []
    for doc in docs:
        for chunk in fn(doc):
            if len(chunk.get("raw", chunk["text"])) >= MIN_CHUNK_CHARS:
                out.append(chunk)
    return out


def stats(chunks: list[dict]) -> dict:
    """Numbers for the comparison report. Section coverage is the one that
    tells you whether strategy B actually found structure or silently fell
    back to windowing."""
    lengths = [len(c["raw"]) for c in chunks] or [0]
    sections = {}
    for c in chunks:
        sections[c["section"]] = sections.get(c["section"], 0) + 1
    return {
        "chunks": len(chunks),
        "mean_chars": sum(lengths) // len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "sections": dict(sorted(sections.items(), key=lambda x: -x[1])),
    }
