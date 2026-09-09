"""Tests for data.build_corpus merge logic (TDD red phase)."""

from data.build_corpus import merge_company_rows


def _row(company, title, freq, topics="Array", difficulty="MEDIUM", link="u"):
    return {"company": company, "Title": title, "Frequency": str(freq),
            "Difficulty": difficulty, "Link": link, "Topics": topics}


def test_same_title_merges_companies():
    rows = [_row("Google", "Two Sum", 100.0), _row("Amazon", "Two Sum", 80.0)]
    problems, stats = merge_company_rows(rows)
    assert len(problems) == 1
    assert problems[0]["companies"] == {"Google": 100.0, "Amazon": 80.0}
    assert stats["matched_titles"] == 1


def test_topics_union_and_first_link_kept():
    rows = [_row("Google", "Two Sum", 100.0, topics="Array, Hash Table"),
            _row("Amazon", "Two Sum", 80.0, topics="Array")]
    problems, _ = merge_company_rows(rows)
    assert sorted(problems[0]["topics"]) == ["Array", "Hash Table"]
    assert problems[0]["link"] == "u"


def test_distinct_titles_stay_distinct():
    rows = [_row("Google", "Two Sum", 100.0), _row("Google", "3Sum", 70.0)]
    problems, stats = merge_company_rows(rows)
    assert len(problems) == 2
    assert stats["total_rows"] == 2


def test_empty_input():
    problems, stats = merge_company_rows([])
    assert problems == [] and stats["total_rows"] == 0
