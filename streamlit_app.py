"""RankDrill dashboard. Thin surface only: all scoring lives in recommender.py."""

import streamlit as st

from recommender import recommend

st.set_page_config(page_title="RankDrill", layout="centered")
st.markdown(
    "<style>h1 {font-size: 2rem;} .stCaption {font-size: 0.85rem;}</style>",
    unsafe_allow_html=True,
)

st.title("RankDrill")
st.caption("Paste a job description. Get the DSA problems worth your hours.")

query = st.text_area("Job description", height=150, placeholder="Backend SDE-1 role: REST APIs, caching with Redis, MySQL, distributed systems…")
col1, col2, col3 = st.columns(3)
with col1:
    company = st.text_input("Company (optional)", placeholder="Flipkart")
with col2:
    method = st.radio("Ranker", ["bm25", "tfidf"], horizontal=True)
with col3:
    top_k = st.slider("Results", 5, 20, 10)

if st.button("Rank problems", type="primary"):
    if not query.strip():
        st.warning("Paste a job description first.")
    else:
        with st.spinner("Ranking…"):
            try:
                hits = recommend(query, top_k=top_k, company=company or None, method=method)
            except FileNotFoundError:
                st.error("Corpus not built yet. Run `python data/build_corpus.py` first.")
                hits = None
            except NotImplementedError:
                st.error("Engine scoring is still being built. Check back soon.")
                hits = None
        if hits is not None:
            if not hits:
                st.info("No matches. Try a longer description with skills and tools.")
            for rank, h in enumerate(hits, start=1):
                st.subheader(f"{rank}. {h['title']}")
                st.caption(
                    f"{h['difficulty']} · score {h['score']:.3f} · "
                    f"matched: {', '.join(h['matched_terms'])}"
                )
                if h.get("companies"):
                    st.caption("Asked by: " + ", ".join(f"{c} ({n}x)" for c, n in h["companies"].items()))
                if h.get("salary_band"):
                    st.caption(f"Pay: {h['salary_band']}")
                st.link_button("Practice", h["link"])
