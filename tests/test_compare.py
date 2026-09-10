"""Tests for compare_offers / compare_companies (TDD red phase)."""

from recommender import compare_companies, compare_offers

COMPANIES = [
    {"name": "Flipkart", "band_label": "F", "min_lpa": 18.0, "max_lpa": 28.0},
    {"name": "tcs", "band_label": "T", "min_lpa": 3.0, "max_lpa": 5.5},
    {"name": "Flat", "band_label": "X", "min_lpa": 30.0, "max_lpa": 30.0},
    {"name": "NoBand", "band_label": "?", "min_lpa": None, "max_lpa": None},
]

PROBLEMS = [
    {"id": "d1", "title": "Two Sum", "topics": ["Arrays"],
     "difficulty": "Easy", "link": "u1", "companies": {"Flipkart": 50.0, "tcs": 90.0}},
    {"id": "d2", "title": "Islands", "topics": ["Graphs"],
     "difficulty": "Medium", "link": "u2", "companies": {"Flipkart": 40.0}},
]


def _offer(company, ctc):
    return {"company": company, "ctc_lpa": ctc}


def test_inside_band_percentile():
    r = compare_offers(COMPANIES, _offer("Flipkart", 23.0), _offer("tcs", 4.0))
    assert r["a"]["percentile"] == 50.0
    assert r["better"] == "A"


def test_above_range_flagged():
    r = compare_offers(COMPANIES, _offer("Flipkart", 40.0), _offer("tcs", 4.0))
    assert "verify in writing" in r["a"]["verdict"]
    assert r["a"]["percentile"] == 100.0
    assert r["better"] == "A"


def test_below_range_flagged():
    r = compare_offers(COMPANIES, _offer("tcs", 2.0), _offer("Flipkart", 23.0))
    assert "Below" in r["a"]["verdict"]


def test_flat_band_no_division_crash():
    r = compare_offers(COMPANIES, _offer("Flat", 30.0), _offer("Flat", 35.0))
    assert r["a"]["percentile"] == 100.0
    assert "Above" in r["b"]["verdict"] or "verify" in r["b"]["verdict"]


def test_unknown_band_is_honest():
    r = compare_offers(COMPANIES, _offer("NoBand", 10.0), _offer("tcs", 4.0))
    assert r["a"]["percentile"] is None
    assert r["better"] is None
    assert r["b"]["verdict"] == "Inside band"


def test_company_compare_shared_and_unique():
    r = compare_companies(PROBLEMS, COMPANIES, "Flipkart", "tcs")
    assert "Arrays" in r["shared_topics"]
    assert "Graphs" in r["only_a"]
    assert r["b"]["name"] == "tcs"


def test_company_compare_unknown_returns_none():
    assert compare_companies(PROBLEMS, COMPANIES, "Flipkart", "Nope") is None
