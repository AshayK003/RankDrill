"""RankDrill dashboard. Thin surface only: all scoring lives in recommender.py.

Security notes: no unsafe_allow_html anywhere near dynamic content (titles and
terms render through Streamlit widgets, which escape HTML). Query text is capped
at MAX_QUERY_CHARS. No secrets, uploads, or runtime network calls.
"""

import streamlit as st

import recommender as rec

MAX_QUERY_CHARS = 2000

st.set_page_config(page_title="RankDrill", layout="centered")
st.markdown(
    "<style>h1 {font-size: 2rem;} .stCaption {font-size: 0.85rem;}</style>",
    unsafe_allow_html=True,
)

st.title("RankDrill")
st.caption("Paste a job description. Get the DSA problems worth your hours.")


@st.cache_resource
def _engine():
    """Load corpus once per session; recommend stays a thin cached call."""
    problems, companies = rec._load_corpus()
    return problems, companies


@st.cache_data(ttl=600, show_spinner=False)
def _rank(query, top_k, company, method):
    return rec.recommend(query, top_k=top_k, company=company or None, method=method)


try:
    problems, companies = _engine()
    st.caption(f"{len(problems)} problems · {len(companies)} salary-tracked companies · 2025-2026 ask data")
except FileNotFoundError:
    st.error("Corpus not built yet. Run `python data/build_corpus.py` first.")
    st.stop()

_EXAMPLES = {
    "Backend SDE-1": "Backend SDE-1 role at Flipkart: Java, Spring Boot, MySQL, Redis caching, message queues, distributed systems, low-latency APIs.",
    "Frontend": "Frontend developer (React): DOM manipulation, string parsing, state management, palindrome checks, text processing, component performance.",
    "OA prep": "Product company online assessment: arrays, hashmaps, sliding window, dynamic programming, graphs, binary search.",
}

st.caption("No JD handy? Try an example:")
ex_cols = st.columns(3)
for (label, text), col in zip(_EXAMPLES.items(), ex_cols):
    with col:
        if st.button(label, width="stretch"):
            st.session_state["jd_input"] = text
            st.rerun()

query = st.text_area(
    "Job description",
    height=150,
    max_chars=MAX_QUERY_CHARS,
    key="jd_input",
    placeholder="Backend SDE-1 role: REST APIs, caching with Redis, MySQL, distributed systems…",
)
col1, col2, col3 = st.columns(3)
with col1:
    company = st.text_input("Company (optional)", placeholder="Flipkart")
with col2:
    method = st.radio("Ranker", ["bm25", "tfidf"], horizontal=True)
with col3:
    top_k = st.slider("Results", 5, 20, 10)

with st.expander("How ranking works", expanded=False):
    st.caption(
        "BM25 and TF-IDF score your text against problem titles and topics "
        "(titles count double). Company ask-frequency from the last six months "
        "boosts frequently-asked problems. Typos are corrected by edit distance "
        "and every correction is shown below."
    )

if st.button("Rank problems", type="primary"):
    if not query.strip():
        st.warning("Paste a job description first.")
    else:
        with st.spinner("Ranking…"):
            try:
                hits = _rank(query.strip(), top_k, company.strip(), method)
            except Exception as exc:
                st.error(f"Ranking failed: {exc}")
                hits = None
        if hits is not None:
            if not hits:
                diag = rec.diagnose_query(query.strip())
                if diag["unknown"]:
                    st.info(
                        "No recognized DSA terms — these words matched nothing: "
                        + ", ".join(diag["unknown"])
                        + ". Add skills like arrays, dynamic programming, graphs, or SQL."
                    )
                else:
                    st.info("No matches. Try a longer description with skills and tools.")
            else:
                first = hits[0]
                if first.get("corrections"):
                    fixes = ", ".join(f"{a} → {b}" for a, b in first["corrections"].items())
                    st.caption(f"Typo fixed: {fixes}")
                if first.get("detected_company"):
                    line = f"Company: {first['detected_company']}"
                    if first.get("salary_band"):
                        line += f" · {first['salary_band']}"
                    st.caption(line)
                for rank, h in enumerate(hits, start=1):
                    st.subheader(f"{rank}. {h['title']}")
                    st.caption(
                        f"{h['difficulty']} · score {h['score']:.3f} · "
                        f"matched: {', '.join(h['matched_terms'])}"
                    )
                    if h.get("companies"):
                        top_cos = sorted(h["companies"].items(), key=lambda kv: -kv[1])[:3]
                        st.caption("Asked by: " + ", ".join(f"{c} ({n:.0f})" for c, n in top_cos))
                    st.link_button("Practice", h["link"])
