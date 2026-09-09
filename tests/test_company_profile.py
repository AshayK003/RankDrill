"""Tests for recommender.company_profile (TDD red phase)."""

import pytest

from recommender import company_profile

PROBLEMS = [
    {"id": "d1", "title": "Two Sum", "topics": ["Arrays", "Hashing"],
     "difficulty": "Easy", "link": "u1", "companies": {"Google": 100.0, "Amazon": 20.0}},
    {"id": "d2", "title": "Number of Islands", "topics": ["Graphs"],
     "difficulty": "Medium", "link": "u2", "companies": {"Google": 40.0}},
    {"id": "d3", "title": "Median Streams", "topics": ["Heaps"],
     "difficulty": "Hard", "link": "u3", "companies": {"Amazon": 60.0}},
]

COMPANIES = [{"name": "Google", "band_label": "L3 total Rs.35-50L/yr"}]


def test_profile_counts_and_splits():
    prof = company_profile(PROBLEMS, COMPANIES, "Google")
    assert prof["problem_count"] == 2
    assert prof["difficulty"] == {"Easy": 1, "Medium": 1, "Hard": 0}


def test_top_topics_weighted_by_frequency():
    prof = company_profile(PROBLEMS, COMPANIES, "google")
    topics = [t for t, _ in prof["top_topics"]]
    assert topics[0] in ("Arrays", "Hashing")
    assert "Graphs" in topics


def test_top_problems_ranked_by_frequency():
    prof = company_profile(PROBLEMS, COMPANIES, "Google")
    assert [p["id"] for p in prof["top_problems"]] == ["d1", "d2"]


def test_salary_band_attached():
    prof = company_profile(PROBLEMS, COMPANIES, "Google")
    assert prof["salary_band"] == "L3 total Rs.35-50L/yr"


def test_unknown_company_returns_none():
    assert company_profile(PROBLEMS, COMPANIES, "NoSuchCo") is None


def test_difficulty_meter_thresholds():
    from recommender import difficulty_meter

    assert difficulty_meter({"Easy": 1, "Medium": 2, "Hard": 7})["verdict"] == "Expect Hard"
    assert difficulty_meter({"Easy": 7, "Medium": 2, "Hard": 1})["verdict"].startswith("Speed")
    assert difficulty_meter({"Easy": 4, "Medium": 4, "Hard": 2})["verdict"] == "Balanced mix"
    assert difficulty_meter({"Easy": 0, "Medium": 0, "Hard": 0})["verdict"] == "No data"


def test_case_mismatch_between_corpus_and_registry():
    docs = [{**PROBLEMS[0], "companies": {"tcs": 50.0}}]
    corps = [{"name": "TCS", "band_label": "B"}]
    prof = company_profile(docs, corps, "TCS")
    assert prof is not None and prof["problem_count"] == 1
