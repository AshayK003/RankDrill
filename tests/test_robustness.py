"""Robustness: weird inputs must fail loudly or degrade gracefully — never crash."""

import pytest

DOCS = [
    {"id": "d1", "title": "Two Sum", "topics": ["Arrays"],
     "difficulty": "Easy", "link": "u1", "companies": {"tcs": 50.0}},
]


@pytest.fixture
def stub(monkeypatch):
    import recommender as rec

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, []))
    return rec


def test_method_case_insensitive(stub):
    assert stub.recommend("arrays", method="BM25")


def test_top_k_string_coerced(stub):
    assert len(stub.recommend("arrays", top_k="1")) == 1


def test_top_k_garbage_raises_loudly(stub):
    with pytest.raises(ValueError):
        stub.recommend("arrays", top_k="many")


def test_non_string_company_ignored(stub):
    hits = stub.recommend("arrays", company=123)
    assert hits and hits[0]["detected_company"] is None


def test_hits_do_not_alias_cache(stub):
    first = stub.recommend("arrays")
    first[0]["companies"]["Evil"] = 999.0
    second = stub.recommend("arrays")
    assert "Evil" not in second[0]["companies"]


def test_query_none_returns_empty(stub):
    assert stub.recommend(None) == []


def test_cli_missing_corpus_message(monkeypatch, capsys):
    import sys

    import recommender as rec
    import run

    def boom(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(rec, "_load_corpus", boom)
    monkeypatch.setattr(sys, "argv", ["run.py", "--query", "arrays"])
    run.main()
    assert "Corpus not built" in capsys.readouterr().out


def test_cli_empty_results_message(monkeypatch, capsys):
    import sys

    import recommender as rec
    import run

    monkeypatch.setattr(rec, "_load_corpus", lambda: (DOCS, []))
    monkeypatch.setattr(sys, "argv", ["run.py", "--query", "xyzzy quux nobody"])
    run.main()
    assert "No matches" in capsys.readouterr().out
