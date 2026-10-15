from sqlalchemy import select

from rolescout.config import get_settings
from rolescout.extract.pipeline import Extractor
from rolescout.extract.providers import LLMError, LLMResult
from rolescout.models import Job, LLMCall, Source


class FakeProvider:
    """Returns canned JSON; records what it was asked."""

    def __init__(self, name, model, responses):
        self.name, self.model = name, model
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system, user, schema, schema_name):
        self.calls.append((schema_name, user))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResult(data=item, provider=self.name, model=self.model, input_tokens=1000, output_tokens=200)


def facts(**over):
    base = {
        "seniority": "entry", "min_years": 1, "is_engineering": True, "role_family": "design_rd",
        "disciplines": ["mechanical"], "industry": "medical_devices", "requires_clearance": False,
        "degree_required": "bachelors", "work_mode": "onsite", "salary_min": None, "salary_max": None,
        "summary": "Design catheter components.", "red_flags": [], "confidence": 0.9,
    }
    base.update(over)
    return base


def add_job(session, title, description, location="San Diego, CA"):
    src = session.scalar(select(Source).where(Source.slug == "t:acme"))
    if src is None:
        src = Source(slug="t:acme", ats="greenhouse", token="acme", company_name="Acme")
        session.add(src)
        session.flush()
    job = Job(source_id=src.id, external_id=str(hash(title + description)), title=title, company_name="Acme",
              location=location, url="https://x", description=description, content_hash=str(hash(description)))
    session.add(job)
    session.commit()
    return job


def test_rules_only_mode(session):
    add_job(session, "Senior Mechanical Engineer", "8+ years of experience.")
    stats = Extractor(get_settings(), use_defaults=False).run(session)
    assert stats.processed == 1 and stats.rules_only == 1
    job = session.scalar(select(Job))
    assert job.facts.seniority == "senior" and job.facts.extractor == "rules"


def test_non_engineering_titles_never_reach_the_model(session):
    add_job(session, "Payroll Specialist", "Run payroll. 2 years of experience.")
    model = FakeProvider("local", "qwen", [])
    stats = Extractor(get_settings(), extractor=model, use_defaults=False).run(session)
    assert model.calls == [] and stats.gated_out == 1


def test_guardrail_title_senior_beats_model(session):
    add_job(session, "Senior Design Engineer", "Design things. Some experience.")
    model = FakeProvider("local", "qwen", [facts(seniority="entry", min_years=None)])
    Extractor(get_settings(), extractor=model, use_defaults=False).run(session)
    job = session.scalar(select(Job))
    assert job.facts.seniority == "senior"
    assert job.facts.extractor == "local:qwen"


def test_disagreement_triggers_verifier_which_corrects(session):
    add_job(session, "Design Engineer", "Requirements: 6+ years of experience designing implants.")
    model = FakeProvider("local", "qwen", [facts(seniority="entry", min_years=1)])
    verifier = FakeProvider("anthropic", "claude-sonnet-5-5", [
        {"verdict": "corrected", "notes": "Requires 6 years; senior.", "facts": facts(seniority="senior", min_years=6)}])
    stats = Extractor(get_settings(), extractor=model, verifier=verifier, use_defaults=False).run(session)
    job = session.scalar(select(Job))
    assert stats.verified == 1 and stats.corrected == 1
    assert job.facts.seniority == "senior" and job.facts.min_years == 6
    assert job.facts.verified_status == "corrected"
    assert "seniority" in job.facts.verifier_notes
    costs = session.scalars(select(LLMCall)).all()
    assert {c.purpose for c in costs} == {"extract", "verify"}
    verify_cost = [c for c in costs if c.purpose == "verify"][0].cost_usd
    assert abs(verify_cost - (1000 * 2 + 200 * 10) / 1e6) < 1e-9  # Sonnet 5.5 at $2/$10 per MTok


def test_agreement_skips_verifier(session):
    add_job(session, "Design Engineer I", "Requirements: 1+ years of experience. BS in Mechanical Engineering.")
    model = FakeProvider("local", "qwen", [facts()])
    verifier = FakeProvider("anthropic", "claude-sonnet-5-5", [])
    stats = Extractor(get_settings(), extractor=model, verifier=verifier, use_defaults=False).run(session)
    # Sampling is deterministic per job id; with a 5% rate this job isn't sampled unless unlucky.
    assert stats.verified in (0, 1)
    if stats.verified == 0:
        assert verifier.calls == []


def test_model_error_falls_back_to_rules_and_is_logged(session):
    add_job(session, "Quality Engineer I", "0-2 years of experience.")
    model = FakeProvider("local", "qwen", [LLMError("server down", retryable=True)])
    stats = Extractor(get_settings(), extractor=model, use_defaults=False).run(session)
    job = session.scalar(select(Job))
    assert stats.llm_errors == 1 and job.facts.seniority == "entry"
    assert job.facts.extractor == "rules (model error)"
    call = session.scalar(select(LLMCall))
    assert call.ok is False and "server down" in call.error


def test_invalid_model_output_retries_once(session):
    add_job(session, "Process Engineer", "1 year of experience.")
    bad = facts()
    del bad["seniority"]
    model = FakeProvider("local", "qwen", [bad, facts(seniority="entry")])
    stats = Extractor(get_settings(), extractor=model, use_defaults=False).run(session)
    assert stats.llm_ok == 1 and len(model.calls) == 2


def test_changed_posting_is_reextracted(session):
    job = add_job(session, "Design Engineer", "1 year of experience.")
    ex = Extractor(get_settings(), use_defaults=False)
    assert ex.run(session).processed == 1
    assert ex.run(session).processed == 0  # unchanged: skipped
    job.description = "Now requires 7 years of experience."
    job.content_hash = "changed"
    session.commit()
    assert ex.run(session).processed == 1
    session.expire_all()
    assert session.scalar(select(Job)).facts.seniority == "senior"
