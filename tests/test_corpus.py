"""Tests for data.build_corpus merge logic (TDD red phase)."""

from data.build_corpus import WINDOW_STATS, fetch_company_best, merge_company_rows


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


def test_fallback_picks_first_window_with_rows():
    calls = []

    def stub(company, window):
        calls.append(window)
        if window == "3. Six Months.csv":
            return []
        return [{"Title": "X", "company": company}]

    rows = fetch_company_best("Acme", _fetch=stub)
    assert len(rows) == 1
    assert WINDOW_STATS["Acme"] == "2. Three Months.csv"
    assert calls[0] == "3. Six Months.csv"


def test_fallback_records_none_when_all_empty():
    rows = fetch_company_best("Ghost", _fetch=lambda c, w: [])
    assert rows == [] and WINDOW_STATS["Ghost"] is None


def test_fetcher_errors_are_skipped():
    def boom(company, window):
        raise IOError("down")

    assert fetch_company_best("Down", _fetch=boom) == []
