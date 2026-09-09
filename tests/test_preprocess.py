"""Tests for recommender.preprocess and recommender.build_index."""

from recommender import build_index, preprocess


def test_lowercase_and_tokenize():
    assert preprocess("Sum Arrays") == ["sum", "arrays"]


def test_stopwords_and_single_chars_removed():
    tokens = preprocess("the a of arrays")
    assert "the" not in tokens and "a" not in tokens
    assert "arrays" in tokens


def test_empty_and_non_string():
    assert preprocess("") == []
    assert preprocess(None) == []


def test_index_term_frequencies():
    docs = [{"id": "d1", "text": "sum arrays arrays"}]
    index = build_index(docs)
    assert index["arrays"] == {"d1": 2}
    assert index["sum"] == {"d1": 1}


def test_index_spans_documents():
    docs = [{"id": "d1", "text": "graphs"}, {"id": "d2", "text": "graphs trees"}]
    index = build_index(docs)
    assert set(index["graphs"]) == {"d1", "d2"}
