# RankDrill

Paste a job description, get ranked DSA problems to practice.

RankDrill is a text-based recommendation system: your JD is the query, a curated set of
DSA problems is the corpus, and classic information retrieval (TF-IDF, BM25, cosine
similarity) does the ranking. Every recommendation shows *why* it matched, and company
ask-frequency plus fresher salary bands ground the results in 2025–2026 hiring data.

## Quick start

```bash
git clone https://github.com/AshayK003/RankDrill.git
cd RankDrill
pip install -r requirements.txt
python data/build_corpus.py   # builds the corpus from seed files
python run.py --query "backend role, REST APIs, caching, distributed systems"
```

Dashboard:

```bash
streamlit run streamlit_app.py
```

## How it works

1. **Preprocess** — lowercase, tokenize, stopword removal.
2. **Index** — inverted index over problem titles + topic descriptions.
3. **Rank** — TF-IDF + cosine vs hand-rolled BM25, side by side; company ask-frequency
   boosts problems frequently asked by the JD's company.
4. **Explain** — every hit lists its matched terms, so no black-box scores.

## Evaluation

`eval.py` reports Precision@5, Recall@5, and MRR over 15 frozen JD queries
(`data/seeds/eval_queries.json`, labelled before the engine was built). Results land in
`results/`.

## Data sources (2025–2026 only)

- Problem list structure follows Striver's A2Z DSA sheet (takeuforward.org).
- Company ask-frequency: [leetcode-company-wise-problems](https://github.com/liquidslr/leetcode-company-wise-problems) (updated Aug 2026).
- Salary bands: hand-curated from [levels.fyi](https://www.levels.fyi), attributed per row.
- Only titles, topics, and links are stored — never full problem statements.

## Status

Early scaffold — engine scoring lands next. See issues for the roadmap.

## License

MIT — see LICENSE.
