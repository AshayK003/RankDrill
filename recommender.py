"""Text-based recommendation engine: preprocess, index, rank DSA problems.

IR concepts applied (all explainable, zero training):
- Field-weighted scoring: title tokens count 2x, topic/glossary tokens 1x.
- Topic glossary: short synonym phrases per topic fix vocabulary mismatch
  ("sliding window" matches even when the title lacks the phrase).
- BM25 Okapi (k1=1.5, b=0.75) and length-normalized TF-IDF, side by side.
- Company ask-frequency multiplies the lexical score by (1 + ln(1 + freq)).
- Company auto-detection: a company named in the JD text is picked up
  without the user selecting it.
"""

import json
import math
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75
TITLE_WEIGHT = 2

TOPIC_GLOSSARY = {
    "arrays": "array list index elements subarray",
    "hashing": "hash map set dictionary frequency counting",
    "strings": "string text characters palindrome substring",
    "two pointers": "two pointers left right converge",
    "sliding window": "sliding window subarray substring contiguous",
    "binary search": "binary search sorted log time divide conquer",
    "stack": "stack push pop lifo parentheses histogram",
    "queue": "queue fifo bfs level",
    "graphs": "graph nodes edges traversal bfs dfs connected components islands",
    "depth first search": "dfs recursion traversal backtrack",
    "breadth first search": "bfs queue level order traversal shortest path",
    "dynamic programming": "dynamic programming dp memoization optimization knapsack subsequence",
    "sorting": "sort order merge intervals scheduling",
    "recursion": "recursion recursive backtracking",
    "linked list": "linked list nodes pointers lru cache",
    "greedy": "greedy choice interval activity selection",
    "heaps": "heap priority queue top k",
    "bit manipulation": "bits xor mask powers of two",
    "math": "math combinatorics probability modular",
    "tries": "trie prefix tree autocomplete",
}

_CACHE = {}


def preprocess(text):
    """Lowercase, tokenize, drop stopwords and single characters."""
    if not isinstance(text, str):
        return []
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def build_index(docs):
    """Build an inverted index: {term: {doc_id: term_frequency}}."""
    index = {}
    for doc in docs:
        counts = {}
        for tok in preprocess(doc["text"]):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            index.setdefault(tok, {})[doc["id"]] = tf
    return index


def _doc_text(problem):
    """Title (weighted) + topics + glossary phrases for vocabulary mismatch."""
    topics = problem.get("topics", [])
    glossary = " ".join(TOPIC_GLOSSARY.get(t.lower(), "") for t in topics)
    return problem["title"] + " " + " ".join(topics) + " " + glossary


def _load_corpus():
    """Load problems + companies from data/ (cached). Raises FileNotFoundError."""
    if "problems" in _CACHE:
        return _CACHE["problems"], _CACHE["companies"]
    root = Path(__file__).parent / "data"
    with open(root / "problems.json", encoding="utf-8") as f:
        problems = json.load(f)
    with open(root / "companies.json", encoding="utf-8") as f:
        companies = json.load(f)
    _CACHE["problems"], _CACHE["companies"] = problems, companies
    return problems, companies


def _corpus_stats(problems):
    """Weighted per-doc tokens, doc frequency, doc count, average doc length."""
    doc_tokens = {}
    for p in problems:
        title_toks = preprocess(p["title"]) * TITLE_WEIGHT
        topic_text = " ".join(p.get("topics", [])) + " " + " ".join(
            TOPIC_GLOSSARY.get(t.lower(), "") for t in p.get("topics", []))
        doc_tokens[p["id"]] = title_toks + preprocess(topic_text)
    df = {}
    for toks in doc_tokens.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(problems)
    avgdl = sum(len(t) for t in doc_tokens.values()) / n if n else 0
    return doc_tokens, df, n, avgdl


def _tfidf_scores(qtokens, doc_tokens, df, n):
    scores = {}
    for pid, toks in doc_tokens.items():
        length = len(toks) or 1
        total = 0.0
        for t in qtokens:
            if t not in df:
                continue
            tf = toks.count(t)
            total += tf * math.log(n / df[t])
        scores[pid] = total / math.sqrt(length)
    return scores


def _bm25_scores(qtokens, doc_tokens, df, n, avgdl):
    scores = {}
    for pid, toks in doc_tokens.items():
        length = len(toks) or 1
        total = 0.0
        for t in qtokens:
            if t not in df:
                continue
            tf = toks.count(t)
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / (avgdl or 1)))
            total += idf * norm
        scores[pid] = total
    return scores


def _boost(score, problem, company):
    if not company:
        return score
    for name, freq in (problem.get("companies") or {}).items():
        if name.lower() == company.lower():
            return score * (1 + math.log(1 + freq))
    return score


def _detect_company(query, companies):
    """Return a company name when the query text names one, else None."""
    lowered = query.lower()
    for c in companies:
        if c["name"].lower() in lowered:
            return c["name"]
    return None


def _salary_band(companies, company):
    if not company:
        return None
    for c in companies:
        if c["name"].lower() == company.lower():
            return c["band_label"]
    return None


def recommend(query, top_k=10, company=None, method="bm25"):
    """Rank problems for a raw query string. Returns ranked hit dicts."""
    if method not in ("bm25", "tfidf"):
        raise ValueError(f"unknown method: {method}")
    qtokens = preprocess(query)
    if not qtokens:
        return []
    problems, companies = _load_corpus()
    if company is None and isinstance(query, str):
        company = _detect_company(query, companies)
    by_id = {p["id"]: p for p in problems}
    doc_tokens, df, n, avgdl = _corpus_stats(problems)
    if method == "bm25":
        scores = _bm25_scores(qtokens, doc_tokens, df, n, avgdl)
    else:
        scores = _tfidf_scores(qtokens, doc_tokens, df, n)
    ranked = sorted(
        ((pid, _boost(s, by_id[pid], company)) for pid, s in scores.items() if s > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )[:top_k]
    hits = []
    for pid, score in ranked:
        p = by_id[pid]
        dtoks = set(doc_tokens[pid])
        hits.append({
            "id": pid,
            "title": p["title"],
            "topics": p.get("topics", []),
            "difficulty": p.get("difficulty", "?"),
            "link": p.get("link", ""),
            "score": round(score, 4),
            "matched_terms": sorted(set(qtokens) & dtoks),
            "companies": p.get("companies", {}),
            "salary_band": _salary_band(companies, company),
            "detected_company": company,
        })
    return hits
