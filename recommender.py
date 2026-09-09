"""Text-based recommendation engine: preprocess, index, rank DSA problems.

IR concepts applied (all explainable, zero training):
- Field-weighted scoring: title tokens count 2x, topic/glossary tokens 1x.
- Topic glossary: short synonym phrases per topic fix vocabulary mismatch
  ("sliding window" matches even when the title lacks the phrase).
- Approximate matching: unknown query tokens fall back to the closest corpus term
  by edit distance (difflib, cutoff 0.8) — typos degrade gracefully, visibly.
- BM25 Okapi (k1=1.5, b=0.75) and length-normalized TF-IDF, side by side.
- Company ask-frequency multiplies the lexical score by (1 + ln(1 + freq)).
- Company detection is punctuation-proof ("JPMorgan" finds "J.P. Morgan") with a
  small alias table for true synonyms ("fb" -> "Meta").
"""

import difflib
import json
import math
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75
TITLE_WEIGHT = 2
FUZZY_CUTOFF = 0.8

# True synonyms only — spelling/punctuation variants are handled generically.
COMPANY_ALIASES = {
    "fb": "Meta",
    "facebook": "Meta",
    "tata consultancy": "tcs",
}

TOPIC_GLOSSARY = {
    # Keys are LeetCode topic names (lowercased). Values add synonyms and the
    # JD words students actually type ("sql", "subarray", "shortest path").
    "array": "array arrays list index elements subarray",
    "arrays": "array arrays list index elements subarray",
    "string": "string strings text characters palindrome substring",
    "strings": "string strings text characters palindrome substring",
    "hash table": "hash hashmap map set dictionary frequency counting",
    "hashing": "hash hashmap map set dictionary frequency counting",
    "dynamic programming": "dynamic programming dp memoization optimization knapsack subsequences",
    "sorting": "sort sorted order merge intervals scheduling",
    "math": "math modular arithmetic combinatorics probability",
    "depth-first search": "dfs graph traversal recursion backtrack islands components",
    "binary search": "binary search sorted log time divide conquer",
    "greedy": "greedy choice interval activity selection",
    "two pointers": "two pointers left right converge",
    "breadth-first search": "bfs queue level order traversal shortest path islands",
    "matrix": "matrix grid rows columns 2d island",
    "tree": "tree trees nodes traversal",
    "binary tree": "binary tree traversal inorder preorder postorder",
    "prefix sum": "prefix sum cumulative subarray range sum",
    "stack": "stack push pop lifo parentheses histogram monotonic",
    "bit manipulation": "bits xor mask powers of two bitmask",
    "heap (priority queue)": "heap priority queue top k largest smallest",
    "heaps": "heap priority queue top k largest smallest",
    "database": "database sql dbms query select join",
    "simulation": "simulation simulate process steps",
    "graph theory": "graph graphs nodes edges traversal components",
    "graphs": "graph graphs nodes edges traversal components",
    "sliding window": "sliding window subarray substring contiguous",
    "linked list": "linked list nodes pointers lru cache",
    "design": "design data structure implement lru cache",
    "backtracking": "backtracking permutations combinations subsets",
    "counting": "counting frequency occurrences",
    "union-find": "union find disjoint set dsu connected components",
    "divide and conquer": "divide conquer merge sort quicksort",
    "recursion": "recursion recursive backtracking",
    "monotonic stack": "monotonic stack next greater histogram",
    "binary search tree": "bst binary search tree validate insert",
    "ordered set": "ordered set balanced tree",
    "number theory": "number theory prime gcd math",
    "segment tree": "segment tree range query fenwick",
    "trie": "trie prefix tree autocomplete",
    "tries": "trie prefix tree autocomplete",
    "enumeration": "enumeration enumerate brute force list all",
    "memoization": "memoization dp cache recursion",
    "topological sort": "topological sort dag dependencies course schedule",
    "data stream": "data stream online median moving",
    "bracket sequences": "brackets parentheses valid stack",
    "directed acyclic graph": "dag directed acyclic topological",
    "bitmask": "bitmask bits dp mask subset",
    "knapsack problem": "knapsack dp weight capacity",
    "shortest path": "shortest path dijkstra bfs bellman ford",
    "dp on trees": "tree dp dfs subtree",
    "game theory": "game minimax nim optimal play",
    "binary indexed tree": "fenwick bit range query",
    "hash function": "hash rolling string matching",
    "monotonic queue": "monotonic queue sliding window deque",
    "string matching": "pattern matching kmp z algorithm substring search",
    "minimax": "minimax game optimal",
    "z algorithm": "z algorithm string matching",
    "merge sort": "merge sort divide conquer",
    "interactive": "interactive queries ask",
    "doubly-linked list": "doubly linked list lru",
    "counting sort": "counting sort linear",
    "dijkstra's algorithm": "dijkstra shortest path graph",
    "euclidean algorithm": "euclid gcd math",
    "queue": "queue fifo bfs level",
}


def _topic_phrases(topic: str) -> str:
    """Glossary enrichment for a topic, falling back to the topic's own words.

    The fallback guarantees every topic — including rare ones like
    "Boruvka's Algorithm" — contributes matchable vocabulary.
    """
    return TOPIC_GLOSSARY.get(topic.lower(), topic)

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
            _topic_phrases(t) for t in p.get("topics", []))
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


def _squash(text):
    """Lowercase alphanumeric only — 'J.P. Morgan' -> 'jpmorgan'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _correct_typos(qtokens, df):
    """Map unknown tokens to the closest corpus term (edit distance).

    Returns (tokens, corrections). Unknown tokens with no close match are dropped.
    """
    vocab = list(df)
    fixed, corrections = [], {}
    for t in qtokens:
        if t in df:
            fixed.append(t)
            continue
        close = difflib.get_close_matches(t, vocab, n=1, cutoff=FUZZY_CUTOFF)
        if close:
            fixed.append(close[0])
            corrections[t] = close[0]
    return fixed, corrections


def _boost(score, problem, company):
    if not company:
        return score
    for name, freq in (problem.get("companies") or {}).items():
        if name.lower() == company.lower():
            return score * (1 + math.log(1 + freq))
    return score


def _canonical_company(name, companies):
    """Resolve aliases and punctuation variants to the stored company name."""
    if not name:
        return None
    lowered = name.lower().strip()
    if lowered in COMPANY_ALIASES:
        return COMPANY_ALIASES[lowered]
    squashed = _squash(lowered)
    for c in companies:
        if c["name"].lower() == lowered or _squash(c["name"]) == squashed:
            return c["name"]
    return name


def _detect_company(query, companies):
    """Return a company name when the query text names one, else None.

    Word-boundary match on spaced text first ("meta" must not fire on "metal");
    squashed match second so punctuation variants ("JPMorgan") still resolve.
    """
    spaced = " " + re.sub(r"[^a-z0-9]+", " ", query.lower()) + " "
    nospace = _squash(query)
    for alias, canonical in COMPANY_ALIASES.items():
        if f" {alias} " in spaced:
            return canonical
    for c in companies:
        name = c["name"].lower()
        if f" {name} " in spaced:
            return c["name"]
    for c in companies:
        squashed = _squash(c["name"])
        if len(squashed) > 4 and squashed in nospace:
            return c["name"]
    return None


def _salary_band(companies, company):
    if not company:
        return None
    for c in companies:
        if c["name"].lower() == company.lower():
            return c["band_label"]
    return None


def diagnose_query(query):
    """Explain why a query may return nothing.

    Returns {usable, unknown, corrections}. Usable = tokens (after typo
    correction) present in the corpus vocabulary. Pure read path.
    """
    qtokens = preprocess(query)
    try:
        problems, _ = _load_corpus()
    except FileNotFoundError:
        return {"usable": [], "unknown": qtokens, "corrections": {}}
    _, df, _, _ = _corpus_stats(problems)
    usable, corrections = _correct_typos(qtokens, df)
    unknown = [t for t in qtokens if t not in df and t not in corrections]
    return {"usable": usable, "unknown": unknown, "corrections": corrections}


def recommend(query, top_k=10, company=None, method="bm25"):
    """Rank problems for a raw query string. Returns ranked hit dicts."""
    if method not in ("bm25", "tfidf"):
        raise ValueError(f"unknown method: {method}")
    if top_k is None or top_k < 1:
        return []
    qtokens = preprocess(query)
    if not qtokens:
        return []
    problems, companies = _load_corpus()
    if not problems:
        return []
    if company is None and isinstance(query, str):
        company = _detect_company(query, companies)
    else:
        company = _canonical_company(company, companies)
    by_id = {p["id"]: p for p in problems}
    doc_tokens, df, n, avgdl = _corpus_stats(problems)
    qtokens, corrections = _correct_typos(qtokens, df)
    if not qtokens:
        return []
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
            "corrections": corrections,
        })
    return hits
