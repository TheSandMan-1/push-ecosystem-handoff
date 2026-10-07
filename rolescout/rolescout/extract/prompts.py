"""Prompts for the extraction and verification tiers."""

from __future__ import annotations

import json

from ..taxonomy import DISCIPLINES, ROLE_FAMILIES, SENIORITIES

_FAMILIES = "\n".join(f"- {k}: {v}" for k, v in ROLE_FAMILIES.items())
_SENIORITIES = "\n".join(f"- {k}: {v}" for k, v in SENIORITIES.items())

EXTRACT_SYSTEM = f"""You read job postings for early-career engineers and record facts about them.
The people using these facts are new and recent engineering graduates. They are hurt by two
mistakes: being shown roles that are actually senior, and being shown "engineer" jobs that are
not engineering (hotel building maintenance, sales, IT helpdesk). Be precise on both.

Fields:
- seniority, one of:
{_SENIORITIES}
  Judge by the title first, then by the REQUIRED years. A title with no level that requires 5+
  years is senior. "Associate" or "I" usually means entry. "II" usually means mid.
- min_years: the minimum years of experience the posting REQUIRES (not "preferred"). If the
  posting gives different minimums by degree, use the one for a bachelor's degree. null if none.
- is_engineering: true only if the job is professional engineering work that an engineering
  degree prepares you for. Building/facilities maintenance, technicians, recruiters and pure
  sales are false.
- role_family, one of:
{_FAMILIES}
- disciplines: engineering degrees the posting asks for, from: {", ".join(DISCIPLINES)}.
- industry: short lowercase label (medical_devices, defense, aerospace, semiconductor,
  automotive, hospitality, energy, software, consumer, other) or null.
- requires_clearance: true if a government security clearance is required or must be obtained.
- degree_required: none, bachelors, masters, phd, or unknown.
- work_mode: onsite, hybrid, remote, or unknown.
- salary_min / salary_max: annual USD if stated (convert hourly x 2080), else null.
- summary: one plain sentence on what the person will actually do day to day.
- red_flags: short phrases a new grad would want to know (title says entry but requires 5
  years, 75% travel, contract role, ITAR / U.S. person, staffing agency). Empty if none.
- confidence: 0 to 1, how sure you are about seniority and is_engineering together.

Return only the JSON object."""

VERIFY_SYSTEM = f"""You are the reviewer in a two-tier pipeline. A smaller model (or rule
engine) extracted facts from a job posting for early-career engineers. Check those facts
against the posting. Most important, in order: seniority and min_years (never let a senior
role pass as entry), is_engineering and role_family (never let hotel maintenance or sales pass
as engineering), requires_clearance.

If every important field is right, return verdict "agree" and the facts unchanged (you may
tidy the summary). If anything important is wrong, return verdict "corrected" with the fixed
facts and say what you changed in notes, in one or two short sentences.

Field definitions are the same as the extractor's:
seniority values: {", ".join(SENIORITIES)}
role_family values: {", ".join(ROLE_FAMILIES)}
disciplines values: {", ".join(DISCIPLINES)}"""


def job_block(title: str, company: str, location: str | None, description: str | None, max_chars: int) -> str:
    desc = (description or "(no description provided)").strip()
    if len(desc) > max_chars:
        # Requirements usually sit near the end; keep the head and the tail.
        head = int(max_chars * 0.55)
        desc = desc[:head] + "\n...\n" + desc[-(max_chars - head):]
    return f"Title: {title}\nCompany: {company}\nLocation: {location or 'unknown'}\n\nPosting:\n{desc}"


def verify_user(job_text: str, proposed: dict) -> str:
    clean = {k: v for k, v in proposed.items() if k != "signals"}
    return f"{job_text}\n\nProposed facts:\n{json.dumps(clean, indent=2)}"
