"""Tiered extraction.

  1. Rules (heuristics.py) run on every posting. Free and instant.
  2. The bulk model (local LLM or Claude) runs only on postings whose title could
     plausibly be engineering. At most companies that is a small minority of jobs,
     so this gate is the biggest cost lever in the system.
  3. The verifier (default Claude Sonnet 5.5) reviews only the postings where the
     rules and the bulk model disagree, the bulk model is unsure, or a small random
     QA sample. The agreement rate is tracked on the admin dashboard.

Guardrails then re-apply what rules see with certainty (an explicit "Senior" in
the title always wins).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from ..config import Settings
from ..models import Job, JobFacts, LLMCall, utcnow
from . import heuristics
from .heuristics import Facts
from .prompts import EXTRACT_SYSTEM, VERIFY_SYSTEM, job_block, verify_user
from .providers import LLMError, LLMResult, Provider, make_provider
from .schema import ExtractedFacts, VerifyResult, facts_json_schema, verify_json_schema

log = logging.getLogger(__name__)

FACT_FIELDS = [
    "seniority", "min_years", "is_engineering", "role_family", "disciplines", "industry",
    "requires_clearance", "degree_required", "work_mode", "salary_min", "salary_max",
    "summary", "red_flags", "confidence",
]
EXPLICIT_TITLE_LEVELS = {"intern", "senior", "lead_manager"}
NOT_ENGINEERING_RULES = {"facilities", "hospitality", "recruiter", "technician", "sales", "support"}


@dataclass
class ExtractStats:
    processed: int = 0
    rules_only: int = 0
    gated_out: int = 0
    llm_ok: int = 0
    llm_errors: int = 0
    verified: int = 0
    corrected: int = 0
    verify_errors: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["cost_usd"] = round(self.cost_usd, 4)
        d["errors"] = self.errors[:20]
        return d


def facts_from_model(data: ExtractedFacts) -> Facts:
    return Facts(**{k: getattr(data, k) for k in FACT_FIELDS})


def apply_guardrails(rules: Facts, facts: Facts) -> Facts:
    """Re-apply what rules know for certain, so a model can't override them."""
    title_level = rules.signals.get("title_seniority")
    if title_level in EXPLICIT_TITLE_LEVELS:
        facts.seniority = title_level
    if title_level == "mid" and facts.seniority in {"entry", "intern", "unknown"}:
        facts.seniority = "mid"  # an explicit "II" never reads as entry level
    if rules.signals.get("role_rule") in NOT_ENGINEERING_RULES:
        facts.is_engineering = False
        facts.role_family = rules.role_family
    facts.requires_clearance = facts.requires_clearance or rules.requires_clearance
    if rules.signals.get("work_mode_source") == "ats" and rules.work_mode != "unknown":
        facts.work_mode = rules.work_mode
    if rules.salary_min and rules.salary_max:
        facts.salary_min, facts.salary_max = rules.salary_min, rules.salary_max
    if title_level == "entry" and facts.min_years is not None and facts.min_years >= 3:
        facts.seniority = "mid" if facts.min_years <= 4 else "senior"
        flag = f"Titled entry-level but asks for {facts.min_years}+ years"
        if flag not in facts.red_flags:
            facts.red_flags.append(flag)
    seen: set[str] = set()
    merged: list[str] = []
    for flag in [*facts.red_flags, *rules.red_flags]:
        key = flag.lower()[:40]
        if key not in seen:
            seen.add(key)
            merged.append(flag)
    facts.red_flags = merged[:6]
    facts.signals = rules.signals
    return facts


def _years_bucket(years: int | None) -> str:
    if years is None:
        return "?"
    return "0-1" if years <= 1 else "2-4" if years <= 4 else "5+"


def disagreement(rules: Facts, model: Facts) -> list[str]:
    reasons = []
    if rules.seniority != "unknown" and model.seniority != "unknown" and rules.seniority != model.seniority:
        reasons.append(f"seniority rules={rules.seniority} model={model.seniority}")
    if rules.is_engineering != model.is_engineering:
        reasons.append(f"is_engineering rules={rules.is_engineering} model={model.is_engineering}")
    if rules.min_years is not None and model.min_years is not None and _years_bucket(rules.min_years) != _years_bucket(model.min_years):
        reasons.append(f"min_years rules={rules.min_years} model={model.min_years}")
    return reasons


def _sampled(job_id: int, rate: float) -> bool:
    if rate <= 0:
        return False
    digest = hashlib.sha256(f"verify-sample:{job_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < rate


def verify_reasons(job: Job, rules: Facts, facts: Facts, settings: Settings, model_used: bool) -> list[str]:
    reasons: list[str] = []
    if model_used:
        reasons += disagreement(rules, facts)
    if facts.confidence < settings.verify_confidence_below:
        reasons.append(f"low confidence {facts.confidence:.2f}")
    if facts.is_engineering and facts.seniority in {"entry", "intern", "unknown"} and facts.min_years is None:
        reasons.append("entry-level claim with no stated years")
    if not reasons and _sampled(job.id, settings.verify_sample_rate):
        reasons.append("QA sample")
    return reasons


def _log_call(session: Session, job: Job | None, purpose: str, provider: Provider,
              result: LLMResult | None, error: str | None) -> float:
    cost = result.cost_usd if result else 0.0
    session.add(LLMCall(
        job_id=job.id if job else None,
        purpose=purpose,
        provider=provider.name,
        model=result.model if result else provider.model,
        input_tokens=result.input_tokens if result else 0,
        output_tokens=result.output_tokens if result else 0,
        cost_usd=cost,
        latency_ms=result.latency_ms if result else 0,
        ok=error is None,
        error=error,
    ))
    return cost


def _call_validated(provider: Provider, system: str, user: str, schema: dict, name: str, model_cls):
    """Call a provider and validate; one retry on malformed output."""
    result = provider.complete_json(system, user, schema, name)
    try:
        return result, model_cls.model_validate(result.data)
    except ValidationError as exc:
        retry_user = f"{user}\n\nYour previous answer failed validation: {exc.errors()[:3]}. Return corrected JSON only."
        result2 = provider.complete_json(system, retry_user, schema, name)
        result2.input_tokens += result.input_tokens
        result2.output_tokens += result.output_tokens
        try:
            return result2, model_cls.model_validate(result2.data)
        except ValidationError as exc2:
            raise LLMError(f"Model output failed validation twice: {exc2.errors()[:2]}") from exc2


def store_facts(session: Session, job: Job, facts: Facts, rules: Facts, extractor: str) -> JobFacts:
    row = job.facts or JobFacts(job_id=job.id)
    for name in FACT_FIELDS:
        setattr(row, name, getattr(facts, name))
    row.heuristic = rules.to_dict()
    row.extractor = extractor
    row.content_hash = job.content_hash
    row.extracted_at = utcnow()
    row.verified_status = "none"
    row.verifier = None
    row.verifier_notes = None
    row.verified_at = None
    if job.facts is None:
        session.add(row)
        job.facts = row
    return row


class Extractor:
    def __init__(self, settings: Settings, extractor: Provider | None = None,
                 verifier: Provider | None = None, *, use_defaults: bool = True):
        self.settings = settings
        if use_defaults:
            extractor = extractor or make_provider(settings.extractor, settings.extractor_model, settings, effort="low")
            verifier = verifier or make_provider(settings.verifier, settings.verifier_model, settings, effort="medium")
        self.extractor = extractor
        self.verifier = verifier

    # -------------------------------------------------------------- one job
    def process(self, session: Session, job: Job, stats: ExtractStats, *, force_verify: bool = False) -> JobFacts:
        industry_hint = job.source.industry if job.source else None
        rules = heuristics.extract(job.title, job.description, job.location, industry_hint, job.extra or {})
        facts = Facts(**{k: getattr(rules, k) for k in FACT_FIELDS})
        facts.signals = rules.signals
        extractor_label = "rules"
        model_used = False
        text = job_block(job.title, job.company_name, job.location, job.description,
                         self.settings.llm_max_description_chars)

        worth_model = heuristics.title_worth_reading(job.title) and bool(job.description)
        if self.extractor is not None and worth_model:
            try:
                result, parsed = _call_validated(
                    self.extractor, EXTRACT_SYSTEM, text, facts_json_schema(), "job_facts", ExtractedFacts)
                stats.cost_usd += _log_call(session, job, "extract", self.extractor, result, None)
                facts = apply_guardrails(rules, facts_from_model(parsed))
                extractor_label = f"{self.extractor.name}:{self.extractor.model}"
                model_used = True
                stats.llm_ok += 1
            except LLMError as exc:
                _log_call(session, job, "extract", self.extractor, None, str(exc))
                stats.llm_errors += 1
                stats.errors.append(f"extract job {job.id}: {exc}")
                extractor_label = "rules (model error)"
        elif self.extractor is not None:
            stats.gated_out += 1
        else:
            stats.rules_only += 1

        row = store_facts(session, job, facts, rules, extractor_label)

        if self.verifier is not None and (worth_model or force_verify):
            reasons = ["requested"] if force_verify else verify_reasons(job, rules, facts, self.settings, model_used)
            if reasons:
                self._verify(session, job, row, facts, text, reasons, stats)
        stats.processed += 1
        return row

    def _verify(self, session: Session, job: Job, row: JobFacts, facts: Facts, text: str,
                reasons: list[str], stats: ExtractStats) -> None:
        assert self.verifier is not None
        proposed = {k: getattr(facts, k) for k in FACT_FIELDS}
        try:
            result, parsed = _call_validated(
                self.verifier, VERIFY_SYSTEM, verify_user(text, proposed), verify_json_schema(),
                "verification", VerifyResult)
        except LLMError as exc:
            _log_call(session, job, "verify", self.verifier, None, str(exc))
            row.verified_status = "error"
            row.verifier = f"{self.verifier.name}:{self.verifier.model}"
            row.verifier_notes = str(exc)[:500]
            stats.verify_errors += 1
            stats.errors.append(f"verify job {job.id}: {exc}")
            return
        stats.cost_usd += _log_call(session, job, "verify", self.verifier, result, None)
        stats.verified += 1
        row.verifier = f"{self.verifier.name}:{result.model}"
        row.verified_at = utcnow()
        trigger = "; ".join(reasons)
        if parsed.verdict == "corrected":
            rules = Facts(**{k: v for k, v in (row.heuristic or {}).items() if k in Facts.__dataclass_fields__})
            corrected = apply_guardrails(rules, facts_from_model(parsed.facts))
            for name in FACT_FIELDS:
                setattr(row, name, getattr(corrected, name))
            row.verified_status = "corrected"
            stats.corrected += 1
        else:
            if parsed.facts.summary and not row.summary:
                row.summary = parsed.facts.summary
            row.verified_status = "agree"
        row.verifier_notes = f"[{trigger}] {parsed.notes}".strip()[:1000]

    # -------------------------------------------------------------- batch
    def run(self, session: Session, limit: int | None = None) -> ExtractStats:
        stats = ExtractStats()
        limit = limit or self.settings.extract_batch_limit
        stmt = (
            select(Job)
            .outerjoin(JobFacts, JobFacts.job_id == Job.id)
            .options(joinedload(Job.source), joinedload(Job.facts))
            .where(Job.closed_at.is_(None))
            .where(or_(JobFacts.job_id.is_(None), JobFacts.content_hash.is_(None),
                       JobFacts.content_hash != Job.content_hash))
            .order_by(Job.first_seen_at.desc())
            .limit(limit)
        )
        jobs = session.scalars(stmt).unique().all()
        for job in jobs:
            try:
                self.process(session, job, stats)
                session.commit()
            except Exception as exc:  # keep the batch going; record the failure
                session.rollback()
                log.exception("extraction failed for job %s", job.id)
                stats.errors.append(f"job {job.id}: {exc}")
        return stats
