"""Tests for recommender.roadmap (TDD red phase)."""

from recommender import roadmap

PROBLEMS = [
    {"id": "d1", "title": "Two Sum", "topics": ["Arrays", "Hashing"],
     "difficulty": "Easy", "link": "u1", "companies": {"Google": 100.0, "tcs": 90.0}},
    {"id": "d2", "title": "Number of Islands", "topics": ["Graphs"],
     "difficulty": "Medium", "link": "u2", "companies": {"Google": 40.0}},
    {"id": "d3", "title": "Median Streams", "topics": ["Heaps", "Arrays"],
     "difficulty": "Hard", "link": "u3", "companies": {"tcs": 10.0}},
]

COMPANIES = [{"name": "Google", "band_label": "G"}, {"name": "tcs", "band_label": "T"}]


def test_topics_ordered_by_weight():
    rm = roadmap(PROBLEMS, COMPANIES, ["Google", "tcs"])
    topics = [t["topic"] for t in rm["topics"]]
    assert topics[0] == "Arrays"
    assert set(rm["resolved"]) == {"Google", "tcs"}


def test_single_company_weights():
    rm = roadmap(PROBLEMS, COMPANIES, ["tcs"])
    topics = [t["topic"] for t in rm["topics"]]
    assert topics[0] == "Arrays"
    assert "Graphs" not in topics


def test_top_problems_per_topic_ranked():
    rm = roadmap(PROBLEMS, COMPANIES, ["Google", "tcs"], per_topic=1)
    arrays = next(t for t in rm["topics"] if t["topic"] == "Arrays")
    assert [p["id"] for p in arrays["problems"]] == ["d1"]
    assert arrays["problems"][0]["frequency"] == 190.0


def test_alias_and_unknown_handling():
    rm = roadmap(PROBLEMS, COMPANIES, ["TCS Digital", "NoSuchCo"])
    assert rm["resolved"] == ["TCS"] or "tcs" in [r.lower() for r in rm["resolved"]]
    assert "NoSuchCo" in rm["unresolved"]


def test_empty_targets():
    rm = roadmap(PROBLEMS, COMPANIES, [])
    assert rm["topics"] == [] and rm["resolved"] == []
