"""Contract tests for recommender scoring (TDD red phase).

Scoring is not implemented yet: every test here must FAIL on NotImplementedError
until the engine task lands. Properties over hand-computed floats, so the tests
stay valid regardless of k1/b smoothing choices.
"""

import pytest

from recommender import recommend

DOCS = [
    {"id": "d1", "title": "Two Sum", "topics": ["Arrays", "Hashing"],
     "difficulty": "Easy", "link": "https://leetcode.com/problems/two-sum/", "companies": {}},
    {"id": "d2", "title": "Number of Islands", "topics": ["Graphs"],
     "difficulty": "Medium", "link": "https://leetcode.com/problems/number-of-islands/",
     "companies": {"Google": 40}},
]


def _run(monkeypatch, method="bm25", **kwargs):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    return rec.recommend("arrays with hashing", top_k=10, method=method, **kwargs)


def test_exact_title_match_ranks_first(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    hits = rec.recommend("number of islands graphs", top_k=2, method="bm25")
    assert hits[0]["id"] == "d2"


def test_both_rankers_agree_on_clear_winner(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    for method in ("bm25", "tfidf"):
        hits = rec.recommend("number of islands graphs", top_k=2, method=method)
        assert hits[0]["id"] == "d2"


def test_empty_query_returns_empty(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    assert rec.recommend("   ", method="bm25") == []


def test_unknown_terms_return_empty(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    assert rec.recommend("xyzzy quux", method="tfidf") == []


def test_contract_keys(monkeypatch):
    hits = _run(monkeypatch)
    assert hits, "expected hits for a matching query"
    for key in ("id", "title", "topics", "difficulty", "link", "score",
                "matched_terms", "companies", "salary_band", "detected_company"):
        assert key in hits[0], f"missing key {key}"


def test_company_auto_detected_from_query_text(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus",
                        lambda: (DOCS, [{"name": "Google", "band_label": "X"}]))
    hits = rec.recommend("Google backend role with graphs", top_k=2, method="bm25")
    assert hits and hits[0]["detected_company"] == "Google"


def test_company_boost_reorders(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    plain = rec.recommend("arrays graphs hashmap", top_k=2, method="bm25")
    boosted = rec.recommend("arrays graphs hashmap", top_k=2, method="bm25", company="Google")
    assert boosted[0]["id"] == "d2"
    plain_ids = [h["id"] for h in plain]
    assert plain_ids != [h["id"] for h in boosted] or plain[0]["id"] == "d2"


def test_deterministic(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    first = rec.recommend("arrays graphs", method="bm25")
    second = rec.recommend("arrays graphs", method="bm25")
    assert [h["id"] for h in first] == [h["id"] for h in second]


def test_unknown_method_rejected(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    with pytest.raises(ValueError):
        rec.recommend("arrays", method="pagerank")


def test_typo_tolerance(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    hits = rec.recommend("ararys", top_k=5, method="bm25")
    assert hits, "typo 'ararys' should still match"
    assert "arrays" in hits[0]["matched_terms"]


def test_company_alias_resolves(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus",
                        lambda: (DOCS, [{"name": "J.P. Morgan", "band_label": "X"}]))
    hits = rec.recommend("JPMorgan backend role with graphs", top_k=2, method="bm25")
    assert hits and hits[0]["detected_company"] == "J.P. Morgan"


def test_tier_alias_routes_to_base_company(monkeypatch):
    import recommender as rec

    docs = [{**DOCS[0], "companies": {"tcs": 50.0}}]
    corps = (docs, [{"name": "TCS", "band_label": "C1Y entry Rs.3-5L/yr"}])
    monkeypatch.setattr(rec, "_load_corpus", lambda: corps)
    hits = rec.recommend("TCS Digital OA with arrays", top_k=2, method="bm25")
    assert hits and hits[0]["detected_company"] == "TCS"
    assert hits[0]["salary_band"] == "C1Y entry Rs.3-5L/yr"


def test_top_k_guards(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    assert rec.recommend("arrays", top_k=0) == []
    assert rec.recommend("arrays", top_k=-3) == []


def test_empty_corpus(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: ([], []))
    assert rec.recommend("arrays") == []


def test_checklist_format():
    from recommender import to_checklist

    hits = [{
        "id": "d1", "title": "Two Sum", "topics": ["Arrays"], "difficulty": "Easy",
        "link": "https://leetcode.com/problems/two-sum/", "score": 1.5,
        "matched_terms": ["sum"], "companies": {"Google": 100.0},
        "salary_band": None, "detected_company": None, "corrections": {},
    }]
    md = to_checklist("arrays practice", "Google", "bm25", hits)
    assert "- [ ] Two Sum" in md
    assert "https://leetcode.com/problems/two-sum/" in md
    assert "arrays practice" in md
    assert to_checklist("q", None, "bm25", []) == ""


def test_diagnose_flags_unknown_terms(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    diag = rec.diagnose_query("backend sde role")
    assert diag["usable"] == [] and set(diag["unknown"]) == {"backend", "sde", "role"}


def test_diagnose_counts_corrected_as_usable(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, {}))
    diag = rec.diagnose_query("ararys")
    assert diag["usable"] == ["arrays"]
    assert diag["corrections"] == {"ararys": "arrays"}


def test_glossary_covers_leetcode_topic_names():
    from recommender import _topic_phrases, preprocess

    assert "hashmap" in preprocess(_topic_phrases("Hash Table"))
    assert "dfs" in preprocess(_topic_phrases("Depth-First Search"))
    assert "sql" in preprocess(_topic_phrases("Database"))


def test_glossary_fallback_covers_unknown_topics():
    from recommender import _topic_phrases, preprocess

    assert preprocess(_topic_phrases("Boruvka's Algorithm")) != []
