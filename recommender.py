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
    "tata consultancy": "TCS",
    "tata consultancy services": "TCS",
    "tcs ninja": "TCS",
    "tcs digital": "TCS",
    "tcs prime": "TCS",
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
    "design": "design data structure implement lru cache rest api caching",
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

    Cutoff scales with token length: short tokens need >=0.9 similarity
    ("redis" must not become "edits"), longer tokens use 0.8. Returns
    (tokens, corrections). Unknown tokens with no close match are dropped.
    """
    vocab = list(df)
    fixed, corrections = [], {}
    for t in qtokens:
        if t in df:
            fixed.append(t)
            continue
        cutoff = 0.9 if len(t) <= 5 else FUZZY_CUTOFF
        close = difflib.get_close_matches(t, vocab, n=1, cutoff=cutoff)
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
    if not isinstance(name, str) or not name.strip():
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


def to_checklist(query, company, method, hits):
    """Render ranked hits as a markdown prep checklist. Pure function."""
    if not hits:
        return ""
    lines = ["# RankDrill prep list", "",
             f"Query: {query} | Company: {company or '-'} | Ranker: {method}", ""]
    for h in hits:
        lines.append(f"- [ ] {h['title']} [{h['difficulty']}]")
        lines.append(f"  {h['link']}")
        asked = ", ".join(f"{c} ({n:.0f})" for c, n in
                          sorted(h.get("companies", {}).items(), key=lambda kv: -kv[1])[:3])
        if asked:
            lines.append(f"  asked by: {asked}")
    return "\n".join(lines) + "\n"


_DIFFICULTY_CLASSES = {"Easy": "pill-easy", "Medium": "pill-medium", "Hard": "pill-hard"}


def difficulty_badge(difficulty):
    """Small HTML pill for a difficulty label (static markup only).

    Unknown values collapse to a neutral badge, so upstream data can never
    inject markup or text into the page. Pair with the .pill CSS classes.
    """
    css = _DIFFICULTY_CLASSES.get(difficulty, "pill-unknown")
    label = difficulty if difficulty in _DIFFICULTY_CLASSES else "?"
    return f'<span class="pill {css}">{label}</span>'


def difficulty_meter(difficulty):
    """Verdict flag from difficulty counts. Heuristic thresholds, disclosed in UI.

    Returns {hard_share, verdict}. 'Expect Hard' at >=30% Hard, 'Speed round'
    at >=60% Easy, else 'Balanced mix'. No data when total is 0.
    """
    total = sum(difficulty.values())
    if total == 0:
        return {"hard_share": 0.0, "verdict": "No data"}
    hard_share = difficulty.get("Hard", 0) / total
    easy_share = difficulty.get("Easy", 0) / total
    if hard_share >= 0.3:
        verdict = "Expect Hard"
    elif easy_share >= 0.6:
        verdict = "Speed round — accuracy over depth"
    else:
        verdict = "Balanced mix"
    return {"hard_share": round(hard_share, 2), "verdict": verdict}


def company_profile(problems, companies, company, top_n=8, top_k=10):
    """Aggregate ask-pattern for one company (pure function).

    Returns {name, problem_count, difficulty{Easy,Medium,Hard},
    top_topics[(topic, freq-sum)], top_problems[id,title,difficulty,link,freq],
    salary_band} or None when the company has no problems. Name match is
    case-insensitive; aliases resolve via _canonical_company.
    """
    canonical = _canonical_company(company, companies)
    names = {canonical} if canonical else set()
    names |= {c["name"] for c in companies
              if canonical and _squash(c["name"]) == _squash(canonical)}
    lower_names = {n.lower() for n in names if n}
    hits = [p for p in problems
            if lower_names & {k.lower() for k in (p.get("companies") or {})}]
    if not hits:
        return None
    stored = next((c["name"] for c in companies if c["name"] in names), canonical)
    freq_of = lambda p: sum(v for k, v in p["companies"].items()
                            if k.lower() in lower_names)
    topic_freq: dict = {}
    for p in hits:
        for t in p.get("topics", []):
            topic_freq[t] = topic_freq.get(t, 0) + freq_of(p)
    ranked = sorted(hits, key=lambda p: -freq_of(p))
    diff_counts = {
        level: sum(1 for p in hits if p.get("difficulty") == level)
        for level in ("Easy", "Medium", "Hard")
    }
    return {
        "name": stored,
        "problem_count": len(hits),
        "difficulty": diff_counts,
        "meter": difficulty_meter(diff_counts),
        "top_topics": sorted(topic_freq.items(), key=lambda kv: -kv[1])[:top_n],
        "top_problems": [
            {"id": p["id"], "title": p["title"], "difficulty": p.get("difficulty", "?"),
             "link": p.get("link", ""), "frequency": round(freq_of(p), 1)}
            for p in ranked[:top_k]
        ],
        "salary_band": _salary_band(companies, stored),
    }


def roadmap(problems, companies, targets, per_topic=3):
    """Order topics by summed ask-frequency across target companies (pure).

    Returns {resolved[stored names], unresolved[input names],
    topics[{topic, weight, problems[{id,title,difficulty,link,frequency}]}]}.
    Aliases/tiers resolve to base companies; unknown names are listed, not fatal.
    """
    resolved, unresolved = [], []
    for t in targets:
        hit = _canonical_company(t, companies) or ""
        match = next((c["name"] for c in companies
                      if c["name"] == hit or _squash(c["name"]) == _squash(hit)), None)
        (resolved if match else unresolved).append(match if match else t)
    lower = {r.lower() for r in resolved}
    topic_freq: dict = {}
    prob_freq: dict = {}
    for p in problems:
        w = sum(v for k, v in (p.get("companies") or {}).items() if k.lower() in lower)
        if w <= 0:
            continue
        prob_freq[p["id"]] = w
        for topic in p.get("topics", []):
            topic_freq[topic] = topic_freq.get(topic, 0) + w
    by_id = {p["id"]: p for p in problems}
    topics = []
    for topic, weight in sorted(topic_freq.items(), key=lambda kv: -kv[1]):
        members = sorted(
            (pid for pid in by_id
             if topic in by_id[pid].get("topics", []) and pid in prob_freq),
            key=lambda pid: -prob_freq[pid],
        )[:per_topic]
        topics.append({
            "topic": topic,
            "weight": round(weight, 1),
            "problems": [
                {"id": pid, "title": by_id[pid]["title"],
                 "difficulty": by_id[pid].get("difficulty", "?"),
                 "link": by_id[pid].get("link", ""),
                 "frequency": round(prob_freq[pid], 1)}
                for pid in members
            ],
        })
    return {"resolved": resolved, "unresolved": unresolved, "topics": topics}


def do_next(problems, problem_id, top_n=5):
    """Problems co-asked with the given one, by shared-company overlap (pure).

    Score = sum over shared companies of min(freq). Returns
    [{id,title,difficulty,link,score,shared[companies]}] minus self, or [].
    """
    by_id = {p["id"]: p for p in problems}
    if problem_id not in by_id:
        return []
    base = by_id[problem_id].get("companies") or {}
    ranked = []
    for p in problems:
        if p["id"] == problem_id:
            continue
        shared = [c for c in (p.get("companies") or {}) if c in base]
        if not shared:
            continue
        score = sum(min(base[c], p["companies"][c]) for c in shared)
        ranked.append({
            "id": p["id"], "title": p["title"],
            "difficulty": p.get("difficulty", "?"), "link": p.get("link", ""),
            "score": round(score, 1),
            "shared": sorted(shared, key=lambda c: -min(base[c], p["companies"][c])),
        })
    ranked.sort(key=lambda h: -h["score"])
    return ranked[:top_n]


def _band_position(companies, company, ctc):
    """Position a CTC inside the verified band: (percentile|None, verdict)."""
    try:
        ctc = float(ctc)
    except (TypeError, ValueError):
        raise ValueError(f"ctc must be a number, got {ctc!r}")
    canonical = _canonical_company(company, companies) or ""
    row = next((c for c in companies if c["name"].lower() == canonical.lower()), None)
    if row is None or row.get("min_lpa") is None or row.get("max_lpa") is None:
        return None, "No verified band — compare on role and growth, not numbers"
    lo, hi = row["min_lpa"], row["max_lpa"]
    if hi == lo:
        if ctc == lo:
            return 100.0, "At the reported figure"
        return (0.0, "Below the reported figure — negotiate up") if ctc < lo else \
            (100.0, "Above reported range — verify in writing")
    if ctc < lo:
        return 0.0, "Below band — negotiate up or verify components"
    if ctc > hi:
        return 100.0, "Above band — verify in writing"
    return round((ctc - lo) / (hi - lo) * 100, 1), "Inside band"


def compare_offers(companies, offer_a, offer_b):
    """Compare two {company, ctc_lpa} offers against verified bands (pure).

    Returns {a{both with percentile+verdict}, b{...}, better ('A'/'B'/'Tie'/None)}.
    better is None when either side lacks band data — no verdict without evidence.
    """
    out = {}
    for key, offer in (("a", offer_a), ("b", offer_b)):
        pct, verdict = _band_position(companies, offer.get("company"), offer.get("ctc_lpa"))
        out[key] = {"company": offer.get("company"), "ctc_lpa": offer.get("ctc_lpa"),
                    "percentile": pct, "verdict": verdict}
    pa, pb = out["a"]["percentile"], out["b"]["percentile"]
    if pa is None or pb is None:
        better = None
    elif pa > pb:
        better = "A"
    elif pb > pa:
        better = "B"
    else:
        better = "Tie"
    out["better"] = better
    return out


def compare_companies(problems, companies, name_a, name_b, top_n=8):
    """Side-by-side ask-patterns plus shared/unique top topics (pure).

    Returns None when either company has no data.
    """
    pa, pb = company_profile(problems, companies, name_a), company_profile(problems, companies, name_b)
    if pa is None or pb is None:
        return None
    ta = [t for t, _ in pa["top_topics"][:top_n]]
    tb = [t for t, _ in pb["top_topics"][:top_n]]
    return {
        "a": pa, "b": pb,
        "shared_topics": [t for t in ta if t in tb],
        "only_a": [t for t in ta if t not in tb],
        "only_b": [t for t in tb if t not in ta],
    }


def recommend(query, top_k=10, company=None, method="bm25"):
    """Rank problems for a raw query string. Returns ranked hit dicts."""
    if isinstance(method, str) and method.lower() in ("bm25", "tfidf"):
        method = method.lower()
    else:
        raise ValueError(f"unknown method: {method}")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValueError(f"top_k must be an integer, got {top_k!r}")
    if top_k < 1:
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
            "companies": dict(p.get("companies", {})),
            "salary_band": _salary_band(companies, company),
            "detected_company": company,
            "corrections": corrections,
        })
    return hits
