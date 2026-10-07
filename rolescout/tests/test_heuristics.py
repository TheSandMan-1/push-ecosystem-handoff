import pytest

from rolescout.config import get_settings
from rolescout.evaluate import load_cases, run_eval
from rolescout.extract.heuristics import extract, extract_salary, extract_years, title_seniority


def test_rules_meet_bar_on_labeled_set():
    report = run_eval(load_cases(), None, get_settings())
    assert report.cases >= 20
    assert report.accuracy >= 0.95, report.failures


@pytest.mark.parametrize("title,expected", [
    ("Senior Mechanical Engineer", "senior"),
    ("Sr. Quality Engineer", "senior"),
    ("Staff Software Engineer", "senior"),
    ("Engineer I/II, Quality", "entry"),
    ("Manufacturing Engineer II", "mid"),
    ("Manufacturing Engineer III", "senior"),
    ("Test Engineer, V&V", "unknown"),
    ("Associate R&D Engineer", "entry"),
    ("Associate Director, Engineering", "lead_manager"),
    ("Engineering Manager", "lead_manager"),
    ("Mechanical Engineering Intern", "intern"),
    ("New Grad Software Engineer", "entry"),
    ("Engineer 2, Systems", "mid"),
])
def test_title_seniority(title, expected):
    assert title_seniority(title)[0] == expected


def test_years_ignores_company_history_and_preferred():
    text = "For over 50 years we have led the industry.\nRequired\n- 2+ years of experience in design\nPreferred\n- 7+ years of experience"
    req, pref, _ = extract_years(text)
    assert req == 2 and pref == 7


def test_years_word_numbers_and_ranges():
    assert extract_years("Minimum of three years of experience required.")[0] == 3
    assert extract_years("3-5 years of relevant experience")[0] == 3
    assert extract_years("Five (5) years of experience")[0] == 5


def test_entry_title_with_senior_requirements_is_flagged():
    f = extract("Entry Level Design Engineer", "Requires 5+ years of experience in SolidWorks.")
    assert f.seniority == "senior"
    assert any("Titled entry-level" in r for r in f.red_flags)


def test_hotel_vs_factory_maintenance():
    hotel = extract("Maintenance Engineer", "Maintain guest rooms and pools at our resort hotel.")
    factory = extract("Maintenance Engineer", "Bachelor's degree in Mechanical Engineering required. Maintain automated lines.")
    assert hotel.is_engineering is False
    assert factory.is_engineering is True


def test_clearance_and_itar():
    f = extract("Systems Engineer", "Must be able to obtain a Secret clearance. U.S. citizenship required.")
    assert f.requires_clearance
    assert any("ITAR" in r for r in f.red_flags)


@pytest.mark.parametrize("text,lo,hi", [
    ("Salary: $85,000 - $105,000 per year", 85000, 105000),
    ("$36.00 - $44.00 per hour", 74880, 91520),
    ("$90k-$120k", 90000, 120000),
    ("We have 3 offices", None, None),
])
def test_salary(text, lo, hi):
    assert extract_salary(text) == (lo, hi)


def test_work_mode_from_location_and_text():
    assert extract("Engineer", "x", "Remote, US").work_mode == "remote"
    assert extract("Engineer", "This is a hybrid role in Irvine.", "Irvine, CA").work_mode == "hybrid"
