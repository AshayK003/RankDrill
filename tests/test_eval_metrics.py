"""Tests for eval metrics (hand-computed expectations)."""

import math

import pytest

from eval import dcg_at_k, mean_ci, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


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


def test_mean_ci():
    mean, ci = mean_ci([1.0, 1.0, 1.0, 1.0])
    assert (mean, ci) == (1.0, 0.0)
    mean, ci = mean_ci([0.0, 1.0])
    assert mean == 0.5 and 0.6 < ci < 1.1
    assert mean_ci([]) == (0.0, 0.0)


def test_ndcg():
    assert ndcg_at_k(["a", "b"], ["a"], 2) == 1.0
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x"], ["a"], 5) == 0.0
    assert ndcg_at_k(["x"], [], 5) == 0.0
    assert dcg_at_k(["a"], {"a": 2}, 1) == 3.0
