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

tab_rank, tab_company, tab_roadmap, tab_resume = st.tabs(
    ["Recommend", "Company patterns", "Roadmaps", "Resume check"])

with tab_rank:
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
                    st.download_button(
                        "Download prep checklist",
                        data=rec.to_checklist(query.strip(), company.strip(), method, hits),
                        file_name="rankdrill-prep.md",
                        mime="text/markdown",
                    )

with tab_company:
    names = sorted({c for p in problems for c in (p.get("companies") or {})})
    picked = st.selectbox("Company", names, index=names.index("Google") if "Google" in names else 0)
    prof = rec.company_profile(problems, companies, picked)
    if prof is None:
        st.info("No ask-data for this company yet.")
    else:
        if prof["salary_band"]:
            st.caption(f"Fresher band: {prof['salary_band']}")
        st.caption(f"OA meter: {prof['meter']['verdict']} (rule of thumb from ask-data)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Problems", prof["problem_count"])
        m2.metric("Easy", prof["difficulty"]["Easy"])
        m3.metric("Medium", prof["difficulty"]["Medium"])
        m4.metric("Hard", prof["difficulty"]["Hard"])
        st.subheader("Most-asked topics")
        for topic, score in prof["top_topics"]:
            st.caption(f"{topic} · {score:.0f}")
        st.subheader("Top problems")
        for p in prof["top_problems"]:
            st.caption(f"{p['title']} [{p['difficulty']}] · asked {p['frequency']:.0f}")
            st.link_button("Practice", p["link"])

with tab_roadmap:
    st.caption("Pick target companies — topics order by what they actually ask.")
    all_names = sorted({c for p in problems for c in (p.get("companies") or {})})
    default = [n for n in ("tcs", "Infosys", "Google") if n in all_names]
    targets = st.multiselect("Target companies", all_names, default=default)
    per_topic = st.slider("Problems per topic", 1, 10, 3)
    if not targets:
        st.warning("Select at least one company.")
    else:
        rm = rec.roadmap(problems, companies, targets, per_topic=per_topic)
        if rm["unresolved"]:
            st.caption("Skipped (no data): " + ", ".join(rm["unresolved"]))
        for i, t in enumerate(rm["topics"], start=1):
            st.subheader(f"{i}. {t['topic']}")
            for p in t["problems"]:
                st.caption(f"{p['title']} [{p['difficulty']}] · asked {p['frequency']:.0f}")
                st.link_button("Practice", p["link"])

with tab_resume:
    st.caption("Paste both as plain text (PDF upload not supported). Nothing leaves your browser session — no storage, no accounts.")
    from resume_check import resume_gap

    rc1, rc2 = st.columns(2)
    with rc1:
        resume_text = st.text_area("Your resume", height=200, placeholder="Backend projects in Python and Django. MySQL…")
    with rc2:
        jd_text = st.text_area("Job description", height=200, key="rc_jd",
                               placeholder="Backend SDE-1: Python, Django, MySQL, Redis, Docker…")
    if st.button("Check alignment", type="primary"):
        if not resume_text.strip() or not jd_text.strip():
            st.warning("Paste both texts first.")
        else:
            gap = resume_gap(resume_text, jd_text)
            st.metric("Keyword coverage", f"{gap['coverage']:.0%}")
            if gap["missing"]:
                st.subheader("Missing (most-mentioned first)")
                for m in gap["missing"][:15]:
                    st.caption(f"{m['term']} · mentioned {m['jd_count']}x in JD")
            else:
                st.success("No missing keywords found. Tailor phrasing, not skills.")
            if gap["matched"]:
                with st.expander("Matched", expanded=False):
                    st.caption(", ".join(m["term"] for m in gap["matched"]))
