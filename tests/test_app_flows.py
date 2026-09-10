"""Headless UI regression flows via Streamlit AppTest (M18 evidence).

These drive the real app script: boot, rank flow, empty-query warning,
resume flow, and static renders of the company/roadmap tabs. Any uncaught
exception in the app fails the run via at.exception.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TIMEOUT = 120
APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _boot():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"boot exception: {at.exception}"
    return at


def _click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            return at.run(timeout=TIMEOUT)
    raise AssertionError(f"button not found: {label}")


def test_boot_clean_and_tabs_present():
    at = _boot()
    assert len(at.tabs) == 4
    assert len(at.text_area) == 3
    assert at.selectbox[0].value == "Google"


def test_rank_flow_produces_results():
    at = _boot()
    at.text_area(key="jd_input").set_value("Backend role: arrays, graphs, SQL, caching")
    _click(at, "Rank problems")
    assert not at.exception, f"rank exception: {at.exception}"
    assert len(at.subheader) > 0
    assert any(b.label == "Download prep checklist" for b in at.download_button)


def test_empty_query_warns_instead_of_crashing():
    at = _boot()
    _click(at, "Rank problems")
    assert not at.exception
    assert len(at.warning) == 1


def test_resume_flow_reports_coverage():
    at = _boot()
    at.text_area(key="rc_resume").set_value("Python and Django projects with MySQL.")
    at.text_area(key="rc_jd").set_value("Backend SDE-1: Python, Django, MySQL, Redis, Docker.")
    _click(at, "Check alignment")
    assert not at.exception, f"resume exception: {at.exception}"
    assert any("%" in m.value for m in at.metric)


def test_company_and_roadmap_tabs_render():
    at = _boot()
    assert len(at.metric) >= 4
    assert len(at.multiselect) == 1
    assert len(at.subheader) > 2
