"""Per-user matching. Deterministic, explainable, and free: no model calls.

Every job either matches (with a score and the reasons) or is excluded with
exactly one reason, so the UI can say "we hid 412 jobs: 230 too senior, ...".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .models import Job, JobFacts, Profile, UserJob, utcnow
from .taxonomy import DISCIPLINES, ROLE_FAMILIES, SENIORITIES

EXCLUSION_LABELS = {
    "closed": "Posting closed",
    "muted_company": "Company muted",
    "not_engineering": "Not a real engineering job",
    "excluded_family": "Role type you excluded",
    "too_senior": "Too senior",
    "level_mismatch": "Wrong level",
    "too_many_years": "Asks for more experience than you have",
    "clearance": "Needs a security clearance",
    "location": "Outside your locations",
    "keyword": "Has a keyword you blocked",
    "salary": "Pays below your floor",
    "hidden": "You hid it",
}

STATE_NAMES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia",
}
_STATE_ABBR = {v: k for k, v in STATE_NAMES.items()}


@dataclass
class Match:
    job: Job
    facts: JobFacts | None
    score: int = 0
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    excluded: str | None = None
    user_status: str | None = None

    @property
    def excluded_label(self) -> str | None:
        return EXCLUSION_LABELS.get(self.excluded) if self.excluded else None


def _location_terms(locations: list[str]) -> list[re.Pattern[str]]:
    patterns = []
    for loc in locations:
        term = loc.strip().lower()
        if not term:
            continue
        variants = {term}
        if term in STATE_NAMES:
            variants.add(STATE_NAMES[term])
        if term in _STATE_ABBR:
            variants.add(_STATE_ABBR[term])
        for v in variants:
            patterns.append(re.compile(rf"(?<![a-z]){re.escape(v)}(?![a-z])", re.I))
    return patterns


def location_ok(job: Job, facts: JobFacts | None, profile: Profile) -> tuple[bool, str | None]:
    if not profile.locations:
        return True, None
    mode = facts.work_mode if facts else "unknown"
    text = job.location or ""
    if profile.remote_ok and (mode == "remote" or re.search(r"\bremote\b", text, re.I)):
        return True, "Remote"
    if not text:
        return True, None
    for pat in _location_terms(profile.locations):
        if pat.search(text):
            return True, None
    return False, None


def evaluate(job: Job, facts: JobFacts | None, profile: Profile, *, user_status: str | None = None,
             now: datetime | None = None) -> Match:
    now = now or utcnow()
    m = Match(job=job, facts=facts, user_status=user_status)

    def exclude(code: str) -> Match:
        m.excluded = code
        return m

    if job.closed_at is not None:
        return exclude("closed")
    if user_status == "hidden":
        return exclude("hidden")
    if job.company_name.lower() in {c.lower() for c in profile.muted_companies}:
        return exclude("muted_company")
    if facts is None:
        # Not extracted yet: show conservatively only if the title is clearly fine.
        m.cautions.append("Details still being analyzed")
        m.score = 40
        return m

    if profile.require_engineering and not facts.is_engineering:
        return exclude("not_engineering")
    if facts.role_family in set(profile.excluded_role_families):
        return exclude("excluded_family")

    targets = set(profile.target_seniorities or ["entry"])
    years = facts.min_years
    level_reason = None
    if facts.seniority in targets:
        level_reason = SENIORITIES.get(facts.seniority, facts.seniority)
    elif facts.seniority in {"mid", "unknown"} and years is not None and years <= profile.max_years:
        level_reason = f"Asks for only {years} yr{'s' if years != 1 else ''}"
    elif facts.seniority == "unknown" and years is None:
        m.cautions.append("Level not stated")
    elif facts.seniority in {"senior", "lead_manager"} or (facts.seniority == "mid" and "mid" not in targets):
        return exclude("too_senior")
    else:
        return exclude("level_mismatch")
    if years is not None and years > profile.max_years:
        return exclude("too_many_years")
    if profile.exclude_clearance and facts.requires_clearance:
        return exclude("clearance")
    loc_ok, loc_reason = location_ok(job, facts, profile)
    if not loc_ok:
        return exclude("location")
    haystack = f"{job.title}\n{facts.summary or ''}".lower()
    for kw in profile.exclude_keywords:
        if kw.strip() and re.search(rf"(?<![a-z]){re.escape(kw.strip().lower())}(?![a-z])", haystack):
            return exclude("keyword")
    if profile.salary_floor and facts.salary_max and facts.salary_max < profile.salary_floor:
        return exclude("salary")

    # Weights sum to ~100 for a perfect fit, so scores spread out instead of all capping.
    score = 40
    if level_reason:
        m.reasons.append(level_reason)
        score += 15 if facts.seniority in targets else 9
    else:
        score += 4
    if years is not None:
        m.reasons.append(f"{years}+ yrs required" if years else "New grads OK")
        score += 5
    if profile.disciplines and facts.disciplines:
        overlap = set(profile.disciplines) & set(facts.disciplines)
        if overlap:
            m.reasons.append(" / ".join(DISCIPLINES[d] for d in sorted(overlap)))
            score += 12
        else:
            m.cautions.append("Asks for " + ", ".join(DISCIPLINES.get(d, d) for d in facts.disciplines))
            score -= 12
    elif profile.disciplines:
        score += 4  # field not stated: neutral-ish
    if profile.role_families and facts.role_family in profile.role_families:
        m.reasons.append(ROLE_FAMILIES[facts.role_family].split(" /")[0])
        score += 8
    hits = 0
    desc_low = (job.description or "").lower()
    for kw in profile.include_keywords:
        k = kw.strip().lower()
        if k and (k in haystack or k in desc_low):
            hits += 1
            if hits <= 2:
                m.reasons.append(f"Mentions {kw.strip()}")
    score += min(hits, 2) * 4
    if loc_reason:
        m.reasons.append(loc_reason)
    posted = job.posted_at or job.first_seen_at
    age = now - posted if posted else timedelta(days=999)
    if age <= timedelta(days=7):
        score += 6
        m.reasons.append("New this week")
    elif age <= timedelta(days=30):
        score += 3
    elif age > timedelta(days=60):
        score -= 6
        m.cautions.append("Posted over 60 days ago")
    if facts.salary_min and profile.salary_floor and facts.salary_min >= profile.salary_floor:
        score += 3
    if facts.verified_status in {"agree", "corrected"}:
        score += 3
    for flag in facts.red_flags or []:
        m.cautions.append(flag)
        score -= 4
    m.score = max(0, min(100, score))
    return m


@dataclass
class Feed:
    matches: list[Match]
    excluded: list[Match]
    exclusion_counts: Counter

    @property
    def hidden_total(self) -> int:
        return sum(self.exclusion_counts.values())


def load_candidates(session: Session, *, include_closed: bool = False) -> list[Job]:
    stmt = select(Job).options(joinedload(Job.facts))
    if not include_closed:
        stmt = stmt.where(Job.closed_at.is_(None))
    return list(session.scalars(stmt).unique().all())


def user_states(session: Session, user_id: int) -> dict[int, UserJob]:
    return {uj.job_id: uj for uj in session.scalars(select(UserJob).where(UserJob.user_id == user_id))}


def build_feed(session: Session, user_id: int, profile: Profile, *, now: datetime | None = None) -> Feed:
    states = user_states(session, user_id)
    matches: list[Match] = []
    excluded: list[Match] = []
    counts: Counter = Counter()
    for job in load_candidates(session):
        st = states.get(job.id)
        m = evaluate(job, job.facts, profile, user_status=st.status if st else None, now=now)
        if m.excluded:
            excluded.append(m)
            if m.excluded != "hidden":
                counts[m.excluded] += 1
        else:
            matches.append(m)
    matches.sort(key=lambda x: (x.score, x.job.posted_at or x.job.first_seen_at), reverse=True)
    return Feed(matches=matches, excluded=excluded, exclusion_counts=counts)
