# RankDrill

Paste a job description. Get the DSA problems worth your hours.

RankDrill is a text-based recommendation system for placement prep. Your job
description is the query; 1,119 DSA problems asked across 37 companies in the last
six months are the corpus; classic information retrieval — TF-IDF, BM25, cosine
similarity — does the ranking. Every recommendation shows *why* it matched, which
companies asked it recently, and the fresher pay band where verified.

## Features

- **JD → problems** — paste any job description, get a ranked practice list.
- **Weakness mode** — describe a weak spot in plain words ("I keep failing sliding
  window") and get targeted problems.
- **Company-aware** — a company named in the JD is detected automatically
  ("JPMorgan" resolves to "J.P. Morgan"); problems it asks frequently rank higher.
- **Salary context** — verified 2026 fresher bands for 12 companies, shown beside results.
- **Transparent** — every hit lists matched terms and typo corrections. No black-box scores.
- **Typo-tolerant** — misspellings fall back to the closest corpus term by edit distance.

## Quick start

```bash
git clone https://github.com/AshayK003/RankDrill.git
cd RankDrill
pip install -r requirements.txt
python data/build_corpus.py   # downloads recent ask-data, builds the corpus
python run.py --query "Backend SDE-1 role: REST APIs, Redis caching, MySQL" --company Flipkart
```

Dashboard:

```bash
streamlit run streamlit_app.py
```

## How it works

1. **Preprocess** — lowercase, tokenize, stopword removal.
2. **Enrich** — each problem is represented by its title (double-weighted) plus topic
   tags plus a short synonym glossary per topic, fixing vocabulary mismatch without
   any embeddings or training.
3. **Rank** — hand-rolled BM25 Okapi or length-normalized TF-IDF, side by side.
   Company ask-frequency from the trailing six months multiplies the lexical score.
4. **Evaluate** — Precision@5, Recall@5, MRR over 15 JD queries labelled *before*
   the engine was built (`data/seeds/eval_queries.json`).

## Results

| Ranker | P@5 | R@5 | MRR |
|---|---|---|---|
| BM25 | 0.533 | 0.023 | 0.70 |
| TF-IDF | 0.507 | 0.021 | 0.65 |

R@5 is low by construction: relevance labels are topic-overlap, so the relevant set
spans hundreds of problems at 1,119 docs. P@5 and MRR are the meaningful metrics —
two-thirds of queries get a relevant problem at rank 1. Full table: run `python eval.py`.

## Data

- **Ask-frequency** — per-company six-month windows (updated Aug 2026), 2025–2026
  only. Downloaded at build time, never vendored.
- **Salaries** — hand-curated fresher bands for 12 companies, each attributed to its
  levels.fyi page. Unverifiable entries are marked, never filled in.
- Only problem titles, topics, and links are stored — never full problem statements.

## Project structure

```
recommender.py      scoring engine (TF-IDF, BM25, company boost, explanations)
run.py              CLI
eval.py             P@5 / R@5 / MRR harness + runner
streamlit_app.py    dashboard (thin surface over the engine)
data/build_corpus.py  corpus builder (download → merge → problems.json)
data/seeds/         frozen eval labels, seed problems, salary bands
tests/              25 tests, all passing
```

## License

MIT — see LICENSE.
