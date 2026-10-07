from datetime import timedelta

from rolescout.matching import evaluate
from rolescout.models import Job, JobFacts, Profile, utcnow


def make(title="Design Engineer I", location="San Diego, CA", closed=False, **f):
    job = Job(id=1, source_id=1, external_id="1", title=title, company_name="Acme", location=location, url="u",
              description="SolidWorks daily.", posted_at=utcnow() - timedelta(days=2), first_seen_at=utcnow(),
              last_seen_at=utcnow(), closed_at=utcnow() if closed else None)
    base = dict(seniority="entry", min_years=1, is_engineering=True, role_family="design_rd",
                disciplines=["mechanical"], requires_clearance=False, work_mode="onsite", red_flags=[],
                verified_status="none", salary_min=None, salary_max=None, summary=None)
    base.update(f)
    return job, JobFacts(job_id=1, **base)


def profile(**over):
    p = Profile(target_seniorities=["entry"], max_years=2, disciplines=["mechanical"], role_families=["design_rd"],
                excluded_role_families=["facilities_maintenance"], locations=["San Diego"], remote_ok=True,
                exclude_clearance=True, require_engineering=True, salary_floor=None, include_keywords=["SolidWorks"],
                exclude_keywords=[], muted_companies=[])
    for k, v in over.items():
        setattr(p, k, v)
    return p


def test_good_match_scores_high_with_reasons():
    m = evaluate(*make(), profile())
    assert m.excluded is None and m.score >= 85
    assert "Mechanical" in m.reasons and "Mentions SolidWorks" in m.reasons


def test_exclusions():
    p = profile()
    assert evaluate(*make(closed=True), p).excluded == "closed"
    assert evaluate(*make(seniority="senior", min_years=8), p).excluded == "too_senior"
    assert evaluate(*make(min_years=4, seniority="mid"), p).excluded == "too_senior"
    assert evaluate(*make(is_engineering=False), p).excluded == "not_engineering"
    assert evaluate(*make(requires_clearance=True), p).excluded == "clearance"
    assert evaluate(*make(location="Austin, TX"), p).excluded == "location"
    assert evaluate(*make(role_family="facilities_maintenance"), p).excluded == "excluded_family"
    assert evaluate(*make(), p, user_status="hidden").excluded == "hidden"
    assert evaluate(*make(), profile(muted_companies=["acme"])).excluded == "muted_company"
    assert evaluate(*make(title="Design Engineer, Night Shift"), profile(exclude_keywords=["night shift"])).excluded == "keyword"
    assert evaluate(*make(salary_min=50000, salary_max=60000), profile(salary_floor=70000)).excluded == "salary"


def test_mid_title_with_low_years_passes_for_entry_seeker():
    m = evaluate(*make(seniority="mid", min_years=2), profile())
    assert m.excluded is None and any("only 2 yrs" in r for r in m.reasons)


def test_remote_passes_location_filter_when_remote_ok():
    assert evaluate(*make(location="Remote, US", work_mode="remote"), profile()).excluded is None
    assert evaluate(*make(location="Remote, US", work_mode="remote"), profile(remote_ok=False)).excluded == "location"


def test_state_abbreviation_matches_full_name():
    assert evaluate(*make(location="Irvine, California"), profile(locations=["CA"])).excluded is None
    assert evaluate(*make(location="Toronto, Canada"), profile(locations=["CA"])).excluded == "location"


def test_discipline_mismatch_is_penalized_not_excluded():
    good = evaluate(*make(), profile())
    off = evaluate(*make(disciplines=["electrical"]), profile())
    assert off.excluded is None and off.score < good.score - 15
    assert any("Asks for Electrical" in c for c in off.cautions)


def test_red_flags_surface_as_cautions():
    m = evaluate(*make(red_flags=["Travel up to 75%"]), profile())
    assert "Travel up to 75%" in m.cautions
