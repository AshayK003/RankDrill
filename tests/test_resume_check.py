"""Tests for resume_check.resume_gap (TDD red phase)."""

from resume_check import resume_gap, skill_terms


def test_phrase_and_unigram_extraction():
    terms = skill_terms("Spring Boot and machine learning with Redis caching")
    assert terms["spring boot"] == 1
    assert terms["machine learning"] == 1
    assert "redis" in terms


def test_missing_sorted_by_jd_frequency():
    jd = "Python Python Python SQL Docker"
    resume = "Python and teamwork"
    gap = resume_gap(resume, jd)
    assert [m["term"] for m in gap["missing"]] == ["sql", "docker"]
    assert gap["matched"] and gap["matched"][0]["term"] == "python"


def test_coverage_weighted():
    gap = resume_gap("Python", "Python Python SQL")
    assert gap["coverage"] == round(2 / 3, 3)


def test_empty_inputs():
    assert resume_gap("", "Python SQL")["coverage"] == 0.0
    assert resume_gap("Python", "") == {"matched": [], "missing": [], "coverage": 0.0}


def test_stopwords_not_skills():
    gap = resume_gap("the and of", "the and of Python")
    assert [m["term"] for m in gap["missing"]] == ["python"]


def test_realistic_texts_keep_only_skills():
    jd = ("Backend SDE-1: developed REST APIs with Python and Django. Worked with "
          "the team on MySQL, Redis caching, Docker deployments. Fast learner, "
          "strong communication, managed sprint deliverables.")
    resume = ("Developed web apps with Python. Team player, quick learner, "
              "managed college fest logistics.")
    gap = resume_gap(resume, jd)
    missing = [m["term"] for m in gap["missing"]]
    for junk in ("developed", "team", "managed", "learner", "communication", "sprint"):
        assert junk not in missing, f"fluff leaked: {junk}"
    for skill in ("django", "mysql", "redis", "docker"):
        assert skill in missing, f"skill missed: {skill}"
