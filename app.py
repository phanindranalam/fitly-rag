"""Streamlit front end.

WHAT THIS UI IS TRYING TO DO THAT MOST RAG DEMOS DO NOT
-------------------------------------------------------
Show its work. A RAG demo that renders a paragraph of prose is
indistinguishable from a chatbot with no retrieval at all -- the interesting
claims (it retrieved, it fused, it filtered, it refused) are invisible. So
every answer here ships with the retrieval trace next to it: which chunks
came back, what each retriever contributed, whether the reranker moved
anything, and what the confidence was.

That panel is also the demo video. Pointing at a question that was refused,
with the similarity score sitting below the threshold on screen, is worth
more than any amount of narration about how the refusal path works.

THREE MODES, ONE PIPELINE
-------------------------
Resume, pasted JD, and free question all run through the same graph. What
differs is only how the QUERY is built, which is the honest way to do it:
"resume matching" is not a different retrieval system, it is a different
query. Keeping that visible in the code stops the app from growing three
divergent half-tested paths.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

import config
import mailer
import vendor_matching as matching
from graph import ask
from resume_loader import SUPPORTED_EXTENSIONS, load as load_resume, to_query
from retrieve import Filters

st.set_page_config(page_title="Fitly RAG", page_icon="🎯", layout="wide")

COUNTRIES = {
    "Anywhere": None, "United States": "us", "Canada": "ca",
    "United Kingdom": "gb", "India": "in", "Germany": "de", "Australia": "au",
}
WORK_MODES = {"Remote": "remote", "Hybrid": "hybrid", "On-site": "onsite"}


# ---------------------------------------------------------------------------
# Shared rendering
# ---------------------------------------------------------------------------

def render_answer(state: dict, question: str) -> None:
    ans = state["answer"]
    res = state["retrieval"]

    if ans.refused:
        st.warning(f"**Not enough evidence to answer**\n\n{ans.text}")
        st.caption(f"Why: {ans.reason}")
    else:
        st.markdown(ans.text)

    if ans.citations:
        st.subheader("Sources")
        for i, h in enumerate(res.hits, start=1):
            if h not in ans.citations:
                continue
            head = f"**[{i}] {h.company or 'Unknown'} — {h.title or 'Untitled role'}**"
            meta = " · ".join(x for x in [h.location, h.compensation] if x)
            st.markdown(f"{head}  \n{meta}" + (f"  \n[Open posting]({h.url})" if h.url else ""))
            with st.expander("The exact text this came from"):
                st.text(h.text[:2000])

    # The panel that makes the retrieval visible. Kept expanded by default on
    # purpose: hiding it is how a RAG demo becomes indistinguishable from a
    # chatbot.
    with st.expander("How this answer was retrieved", expanded=True):
        st.code("\n".join(state.get("trace", [])), language="text")
        rows = []
        for i, h in enumerate(res.hits, start=1):
            rows.append({
                "#": i,
                "fused (RRF)": round(h.score, 4),
                "similarity": round(h.sim, 3) if h.sim is not None else None,
                "dense rank": h.dense_rank or "—",
                "BM25 rank": h.sparse_rank or "—",
                "rerank": round(h.rerank_score, 3) if h.rerank_score is not None else "—",
                "section": h.section,
                "source": h.citation(),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Order comes from Reciprocal Rank Fusion of the dense and BM25 rankings. "
            f"The refusal decision uses raw similarity ({res.top_sim:.3f}) against "
            f"MIN_SIM={config.MIN_SIM}, because a fused rank score has no absolute meaning. "
            f"Total {state['elapsed_s']:.2f}s against a {config.LATENCY_BUDGET_S}s budget.")

    if mailer.available():
        with st.form(f"email_{abs(hash(question)) % 10000}"):
            addr = st.text_input("Email these results to", placeholder="you@example.com")
            if st.form_submit_button("Send") and addr:
                ok, msg = mailer.send(addr, "Your Fitly RAG matches",
                                      mailer.build_html(question, ans.text, ans.citations, ans.refused))
                (st.success if ok else st.error)(msg)


def sidebar_controls() -> dict:
    st.sidebar.header("Search scope")
    country = st.sidebar.selectbox("Country", list(COUNTRIES), index=0)
    modes = st.sidebar.multiselect("Work mode", list(WORK_MODES), default=[])
    city = st.sidebar.text_input("City (optional)", placeholder="atlanta")

    st.sidebar.caption(
        "Location is a hard constraint applied **before** the vector search, "
        "not a ranking bonus after it. A role in the wrong country is not a "
        "worse match — it is not a match.")

    st.sidebar.header("Retrieval")
    strategy = st.sidebar.radio("Chunking", ["section", "fixed"], index=0,
                                help="Two indexes were built. Switch to compare them live.")
    mode = st.sidebar.radio("Retriever", ["hybrid", "dense"], index=0,
                            help="hybrid = dense + BM25 fused with RRF. dense = embeddings only.")
    # Defaults to ON because every published number -- retrieval hit@5, the
    # demo similarities, the whole eval matrix's best row -- was measured with
    # reranking on. A default that doesn't match the measured configuration
    # means the app a reader opens is not the app the writeup describes.
    rerank = st.sidebar.toggle("Cross-encoder rerank", value=True,
                               help="Slower and usually more accurate. The eval measures whether it actually helps here.")

    st.sidebar.divider()
    st.sidebar.caption(config.describe())

    filters = Filters(
        country=COUNTRIES[country],
        work_modes=tuple(WORK_MODES[m] for m in modes),
        city=city.strip().lower() or None,
    )
    return {"filters": filters, "strategy": strategy, "mode": mode, "rerank": rerank}


def run(question: str, ctrl: dict, resume: str | None = None) -> None:
    with st.spinner("Retrieving, then answering only from what came back..."):
        state = ask(question, resume=resume, strategy=ctrl["strategy"],
                    filters=ctrl["filters"], use_rerank=ctrl["rerank"], mode=ctrl["mode"])
    render_answer(state, question)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("Fitly")
st.markdown("#### Find the jobs worth your time")
st.caption("Ask across 874 real job postings. Every answer comes with the posting behind it — "
           "or Fitly tells you when the evidence isn't there.")
st.caption("874 postings · 93 companies · 97.9% claim-level faithfulness · 0 missed refusals")

problems = config.check()
if problems:
    for p in problems:
        st.error(p)
    st.stop()

ctrl = sidebar_controls()

# Resume lives above the tabs because two of the three modes need it, and
# re-uploading per tab is the kind of small friction that makes a demo drag.
with st.container(border=True):
    up = st.file_uploader("Your resume (PDF or DOCX) — optional",
                          type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS])
    if up is not None and st.session_state.get("_resume_name") != up.name:
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            fh.write(up.getbuffer())
            tmp = fh.name
        try:
            st.session_state["resume"] = load_resume(tmp)
            st.session_state["_resume_name"] = up.name
        finally:
            os.unlink(tmp)

    parsed = st.session_state.get("resume")
    if parsed:
        st.success(f"Parsed **{st.session_state['_resume_name']}** — {parsed.summary()}")
        for w in parsed.warnings:
            st.warning(w)
        if parsed.competencies:
            cols = st.columns(min(len(parsed.competencies), 4))
            for col, (group, terms) in zip(cols * 4, parsed.competencies.items()):
                col.markdown(f"**{group.replace('_', ' ').title()}**  \n" +
                             ", ".join(terms[:8]))

tab_match, tab_jd, tab_ask = st.tabs(
    ["Match my resume", "Compare a job description", "Ask the postings"])

with tab_match:
    st.markdown("Retrieval query is built from your strongest signals — target role and "
                "highest-value skill terms — not from the whole resume. A 5,000-character "
                "document averaged into one vector retrieves averages.")
    role = st.text_input("Target role (optional)", placeholder="Platform Engineering Manager")
    if st.button("Find matching postings", type="primary", disabled=not st.session_state.get("resume")):
        p = st.session_state["resume"]
        query = to_query(p, role)
        st.code(f"query -> {query}", language="text")
        run(f"Which of these roles fit this background, and where are the gaps? "
            f"Background: {query}", ctrl, resume=p.text)
    if not st.session_state.get("resume"):
        st.info("Upload a resume above to use this tab.")

with tab_jd:
    st.markdown("Paste a posting you found elsewhere. The deterministic scorer compares it "
                "to your resume term by term; the retriever then finds comparable roles in "
                "the corpus.")
    jd = st.text_area("Job description", height=220, placeholder="Paste the full posting text...")
    if st.button("Compare", type="primary", disabled=not (jd.strip() and st.session_state.get("resume"))):
        p = st.session_state["resume"]
        fit = matching.compute_fit(p.text, jd, max_gaps=12)
        c1, c2, c3 = st.columns(3)
        c1.metric("Keyword fit", f"{fit.fit_percent:.0f}%")
        c2.metric("Matched terms", len(fit.matched_keywords))
        c3.metric("Confidence", fit.confidence)
        st.markdown("**Present in both:** " + (", ".join(sorted(fit.matched_keywords)[:20]) or "—"))
        st.markdown("**In the posting, not evidenced in your resume:** " +
                    (", ".join(fit.gap_keywords[:12]) or "—"))
        st.caption("Absence of a keyword is not proof of absent skill — it is proof the "
                   "resume does not say it, which is what an ATS reads.")
        st.divider()
        run("What comparable roles are in these postings, and what do they require?",
            ctrl, resume=p.text)

with tab_ask:
    st.markdown("Free question against the indexed postings. Try one the corpus cannot "
                "answer — *what is the company's parental leave policy in weeks?* — and "
                "watch it refuse instead of guessing.")
    q = st.text_input("Question", placeholder="Which roles require running Kubernetes in production?")

    # Every demo query is one click. Typing live on camera is the easiest way
    # to lose fifteen seconds to a typo in a five-minute recording.
    row1 = st.columns(3)
    row2 = st.columns(2)
    for col, sample in zip(list(row1) + list(row2), [
        "Which roles require running Kubernetes in production?",
        "Which roles require a security clearance?",
        "What do these postings say about on-call?",
        "How many people applied to this job?",
        "What is a good recipe for sourdough bread?",
    ]):
        if col.button(sample, width="stretch"):
            q = sample
    if q:
        run(q, ctrl)
