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
