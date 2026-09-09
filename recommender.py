"""Text-based recommendation engine: preprocess, index, rank DSA problems.

Contract (frozen): recommend() takes a raw query string and returns a ranked list
of problem dicts, each with score, matched_terms, companies, and salary_band.
Scoring (TF-IDF / BM25) lands in the engine task; preprocessing and the inverted
index below are final and covered by tests.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9]+")


def preprocess(text):
    """Lowercase, tokenize, drop stopwords and single characters.

    Returns a list of tokens. Empty / non-string input returns [].
    """
    if not isinstance(text, str):
        return []
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def build_index(docs):
    """Build an inverted index: {term: {doc_id: term_frequency}}.

    docs: iterable of {"id": str, "text": str}. Pure function, no I/O.
    """
    index = {}
    for doc in docs:
        counts = {}
        for tok in preprocess(doc["text"]):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            index.setdefault(tok, {})[doc["id"]] = tf
    return index


def recommend(query, top_k=10, company=None, method="bm25"):
    """Rank problems for a raw query string.

    Returns [{id, title, topics, difficulty, link, score, matched_terms,
    companies, salary_band}]. Raises NotImplementedError until scoring lands.
    """
    raise NotImplementedError("scoring lands in the engine task")
