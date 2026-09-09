"""Tests for eval metrics (hand-computed expectations)."""

from eval import precision_at_k, recall_at_k, reciprocal_rank


def test_precision():
    assert precision_at_k(["a", "b", "c"], ["a", "x"], 2) == 0.5
    assert precision_at_k([], ["a"], 5) == 0.0
    assert precision_at_k(["a"], ["a"], 0) == 0.0


def test_recall():
    assert recall_at_k(["a", "b"], ["a", "x", "y", "z"], 2) == 0.25
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0
