"""JD-resume keyword alignment. Pure functions, zero Streamlit imports.

Method: skill terms = unigrams (stopwords removed) plus a curated multiword
phrase list matched as substrings. Gaps are JD terms absent from the resume,
ranked by JD frequency — the most-mentioned missing skill is the top priority.
Coverage is frequency-weighted, not a raw count ratio.
"""

from recommender import preprocess

TECH_PHRASES = [
    "spring boot", "machine learning", "deep learning", "natural language processing",
    "data structures", "system design", "rest api", "restful", "microservices",
    "distributed systems", "computer vision", "data science", "operating systems",
    "database management", "object oriented",
]

# Unigrams that count as skills. Everything else (developed, team, worked,
# managed, fast, learner) is resume fluff, not matchable signal.
TECH_UNIGRAMS = {
    "python", "java", "javascript", "typescript", "golang", "rust", "kotlin",
    "swift", "scala", "ruby", "php", "csharp", "cpp", "c",
    "react", "angular", "vue", "nextjs", "nodejs", "express", "django", "flask",
    "spring", "boot", "laravel", "rails", "flutter", "reactnative",
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "elasticsearch",
    "nosql", "sqlite", "oracle", "dynamodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "git", "linux", "bash", "nginx",
    "pandas", "numpy", "tensorflow", "pytorch", "scikit", "keras", "spark",
    "hadoop", "tableau", "excel",
    "arrays", "strings", "hashing", "hashmap", "graphs", "trees", "dp",
    "recursion", "oops", "os", "dbms", "cn",
    "api", "apis", "sdk", "graphql", "grpc", "kafka", "rabbitmq",
    "testing", "selenium", "pytest", "junit", "ci", "cd",
    "agile", "scrum", "jira",
}


def skill_terms(text):
    """Map of skill term -> mention count. Empty/non-string input gives {}."""
    if not isinstance(text, str) or not text.strip():
        return {}
    terms = {}
    for tok in preprocess(text):
        if tok in TECH_UNIGRAMS:
            terms[tok] = terms.get(tok, 0) + 1
    lowered = text.lower()
    for phrase in TECH_PHRASES:
        count = lowered.count(phrase)
        if count:
            terms[phrase] = terms.get(phrase, 0) + count
    return terms


def resume_gap(resume_text, jd_text):
    """Compare resume against JD.

    Returns {matched[{term, jd_count}], missing[{term, jd_count}] (freq desc),
    coverage (0.0-1.0, JD-frequency-weighted)}.
    """
    jd_terms = skill_terms(jd_text)
    if not jd_terms:
        return {"matched": [], "missing": [], "coverage": 0.0}
    resume_terms = set(skill_terms(resume_text))
    matched = [{"term": t, "jd_count": c} for t, c in jd_terms.items() if t in resume_terms]
    missing = [{"term": t, "jd_count": c} for t, c in jd_terms.items() if t not in resume_terms]
    missing.sort(key=lambda m: -m["jd_count"])
    total = sum(jd_terms.values())
    hit = sum(m["jd_count"] for m in matched)
    return {"matched": matched, "missing": missing, "coverage": round(hit / total, 3)}
