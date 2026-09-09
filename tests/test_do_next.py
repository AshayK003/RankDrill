"""Tests for recommender.do_next (TDD red phase)."""

from recommender import do_next

PROBLEMS = [
    {"id": "d1", "title": "Two Sum", "topics": ["Arrays"],
     "difficulty": "Easy", "link": "u1",
     "companies": {"Google": 100.0, "Amazon": 80.0}},
    {"id": "d2", "title": "3Sum", "topics": ["Arrays"],
     "difficulty": "Medium", "link": "u2",
     "companies": {"Google": 60.0, "Meta": 50.0}},
    {"id": "d3", "title": "Islands", "topics": ["Graphs"],
     "difficulty": "Medium", "link": "u3", "companies": {"Meta": 90.0}},
]


def test_ranks_by_shared_company_overlap():
    nxt = do_next(PROBLEMS, "d1", top_n=5)
    assert [p["id"] for p in nxt] == ["d2"]
    assert nxt[0]["shared"] == ["Google"]
    assert nxt[0]["score"] == 60.0


def test_self_excluded():
    assert all(p["id"] != "d1" for p in do_next(PROBLEMS, "d1"))


def test_unknown_id_returns_empty():
    assert do_next(PROBLEMS, "nope") == []


def test_no_shared_companies_returns_empty():
    lone = [{"id": "x", "title": "X", "topics": [], "difficulty": "Easy",
             "link": "u", "companies": {"Only": 1.0}},
            {"id": "y", "title": "Y", "topics": [], "difficulty": "Easy",
             "link": "u", "companies": {"Other": 1.0}}]
    assert do_next(lone, "x") == []
