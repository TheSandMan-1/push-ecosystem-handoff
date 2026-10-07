"""One test per finding from the first independent code review."""

import sqlite3
import threading

import httpx
import pytest
from sqlalchemy import select

from rolescout.ats import detect_source
from rolescout.config import get_settings
from rolescout.extract.heuristics import extract, extract_years
from rolescout.extract.pipeline import apply_guardrails
from rolescout.extract.heuristics import Facts
from rolescout.extract.providers import LLMError, LocalProvider, cost_usd
from rolescout.ingest import run_ingest, upsert_source
from rolescout.matching import evaluate
from rolescout.models import Job, Profile, utcnow

from .fakeboards import Boards
from .test_matching import profile


def test_unanalyzed_job_is_never_shown():
    job = Job(id=1, source_id=1, external_id="x", title="Senior Director of Sales", company_name="Acme",
              url="u", first_seen_at=utcnow(), last_seen_at=utcnow())
    assert evaluate(job, None, profile()).excluded == "pending"


@pytest.mark.parametrize("text,required", [
    ("Qualifications\n- Experience with SolidWorks preferred\n- 5+ years of design experience required", 5),
    ("Bachelor's degree plus 5 years of experience in product design.", 5),
    ("Requirements: 6+ years of experience in quality; ISO 13485 is a plus.", 6),
])
def test_required_years_not_misfiled_as_preferred(text, required):
    assert extract_years(text)[0] == required


def test_only_preferred_high_years_is_flagged():
    f = extract("Mechanical Engineer", "Preferred:\n- 7+ years of experience in design")
    assert f.min_years is None and any("Prefers 7+" in r for r in f.red_flags)


def test_support_engineer_is_not_engineering():
    assert extract("IT Support Engineer", "Help desk tickets.").is_engineering is False


def test_guardrails_keep_hospitality_and_explicit_mid():
    rules = extract("Engineer", "Maintain guest rooms and pools at our hotel resort. Hotel housekeeping support.")
    assert rules.signals["role_rule"] == "hospitality"
    model = Facts(seniority="entry", is_engineering=True, role_family="other_engineering")
    out = apply_guardrails(rules, model)
    assert out.is_engineering is False

    rules = extract("Mechanical Engineer II", "Design parts.")
    out = apply_guardrails(rules, Facts(seniority="entry", is_engineering=True, role_family="design_rd"))
    assert out.seniority == "mid"


def test_capped_workday_fetch_does_not_close_jobs(session, monkeypatch):
    monkeypatch.setenv("ROLESCOUT_WORKDAY_MAX_JOBS", "3")
    monkeypatch.setenv("ROLESCOUT_WORKDAY_PAGE_LIMIT", "3")
    get_settings.cache_clear()
    boards = Boards()
    src, _ = upsert_source(session, detect_source("https://acme.wd1.myworkdayjobs.com/External"), "Acme")
    session.commit()
    with boards.client() as c:
        run_ingest(session, client=c)
        # Three new postings appear at the top; the cap now hides the original three.
        new = [dict(boards.workday[0], title=f"Engineer {i}", externalPath=f"/job/x/N{i}", bulletFields=[f"N{i}"])
               for i in range(3)]
        boards.workday = new + boards.workday
        for _ in range(3):
            run_ingest(session, client=c)
    session.expire_all()
    originals = session.scalars(select(Job).where(Job.external_id.in_(["R1001", "R1002", "R1003"]))).all()
    assert originals and all(j.closed_at is None for j in originals)


def test_web_writes_are_not_blocked_while_details_download(session, tmp_path):
    """Reviewer repro: a write during the lazy detail phase used to fail with 'database is locked'."""
    boards = Boards()
    upsert_source(session, detect_source("https://acme.wd1.myworkdayjobs.com/External"), "Acme")
    session.commit()
    outcome = {}
    original = boards.handler

    def handler(request):
        if request.method == "GET" and "outcome" not in outcome:
            con = sqlite3.connect(f"{tmp_path}/test.db", timeout=1)
            try:
                con.execute("CREATE TABLE IF NOT EXISTS probe (x INTEGER)")
                con.execute("INSERT INTO probe VALUES (1)")
                con.commit()
                outcome["ok"] = True
            except sqlite3.OperationalError as exc:
                outcome["ok"] = str(exc)
            finally:
                con.close()
        return original(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        run_ingest(session, client=c)
    assert outcome.get("ok") is True


def test_ats_workplace_type_reaches_extraction(session):
    from rolescout.extract.pipeline import Extractor

    boards = Boards()
    boards.lever[0]["workplaceType"] = "remote"
    boards.lever[0]["categories"]["location"] = "San Francisco, CA"
    upsert_source(session, detect_source("https://jobs.lever.co/acme"), "Acme")
    session.commit()
    with boards.client() as c:
        run_ingest(session, client=c)
    Extractor(get_settings(), use_defaults=False).run(session)
    job = session.scalar(select(Job))
    assert job.extra["workplace_type"] == "remote"
    assert job.facts.work_mode == "remote"
    assert job.facts.salary_min == 80000


def test_cost_uses_prefix_for_snapshot_and_fallback_models():
    assert cost_usd("claude-sonnet-5-5-20261001", 1_000_000, 0) == 2.0
    assert cost_usd("claude-opus-4-8", 0, 1_000_000) == 25.0
    assert cost_usd("qwen2.5:7b", 1000, 1000) == 0.0


def test_local_non_json_response_is_llm_error():
    p = LocalProvider("http://local/v1", "qwen", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html>proxy error</html>"))))
    with pytest.raises(LLMError, match="non-JSON"):
        p.complete_json("s", "u", {}, "x")
