"""Deterministic fact extraction. Free, instant, and the ground truth for
anything a rule can see reliably (an explicit "Senior" in the title beats any
model's opinion). Models only fill in what rules can't.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15,
}


@dataclass
class Facts:
    seniority: str = "unknown"
    min_years: int | None = None
    is_engineering: bool = False
    role_family: str = "non_engineering"
    disciplines: list[str] = field(default_factory=list)
    industry: str | None = None
    requires_clearance: bool = False
    degree_required: str = "unknown"
    work_mode: str = "unknown"
    salary_min: int | None = None
    salary_max: int | None = None
    summary: str | None = None
    red_flags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    # Provenance for the rules that fired; useful for debugging and the admin view.
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- seniority

_RE = lambda p: re.compile(p, re.I)  # noqa: E731

T_INTERN = _RE(r"\b(intern|internship|co-?op|summer student)\b")
T_LEAD = _RE(
    r"\b(manager|director|head of|vp|vice president|chief|supervisor|team lead|tech(?:nical)? lead|lead)\b"
)
T_SENIOR = _RE(r"\b(senior|sr\.?|staff|principal|distinguished|fellow|expert|architect)\b")
T_ENTRY = _RE(
    r"\b(entry[- ]level|entry|junior|jr\.?|new grad(uate)?|graduate|early[- ]career|"
    r"university|rotational|rotation program|apprentice|recent grad(uate)?)\b"
)
T_ASSOCIATE = _RE(r"\bassociate\b")
_ROMAN_TOKENS = {"I", "II", "III", "IV", "V"}
ARABIC_LEVEL = _RE(r"\b(?:engineer|scientist|developer|technologist|level)\s*(?:-\s*)?([1-5])\b")
_ROMAN_VAL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


def title_seniority(title: str) -> tuple[str, str | None]:
    """Seniority stated by the title itself, plus the token that decided it."""
    t = title or ""
    if m := T_INTERN.search(t):
        return "intern", m.group(0)
    # "Associate Director" is management, "Associate Engineer" is entry.
    if m := T_LEAD.search(t):
        word = m.group(0).lower()
        # "Lead-free", "Leadership Development Program" are not management.
        if not (word == "lead" and re.search(r"lead[- ]free|leadership development", t, re.I)):
            return "lead_manager", m.group(0)
    if m := T_SENIOR.search(t):
        return "senior", m.group(0)
    # Split on separators but keep "&" so "V&V" never reads as level V.
    tokens = re.split(r"[\s,()/\-:]+", t)
    levels = [_ROMAN_VAL[x] for x in tokens if x in _ROMAN_TOKENS]
    levels += [int(x) for x in ARABIC_LEVEL.findall(t)]
    if levels:
        level = min(levels)  # "Engineer I/II" -> I
        return ("entry" if level <= 1 else "mid" if level == 2 else "senior"), f"level {level}"
    if m := T_ENTRY.search(t):
        return "entry", m.group(0)
    if m := T_ASSOCIATE.search(t):
        return "entry", m.group(0)
    if re.search(r"\b(intermediate|mid[- ]level)\b", t, re.I):
        return "mid", "mid-level"
    return "unknown", None


# --------------------------------------------------------------------------- years of experience

_NUM = r"(\d{1,2}|" + "|".join(WORD_NUMBERS) + r")"
YEARS_PATTERNS = [
    # "3-5 years", "3 to 5 years", "3 – 5+ years"
    re.compile(_NUM + r"\s*\+?\s*(?:-|–|—|to)\s*" + _NUM + r"\s*\+?\s*(?:years?|yrs?)", re.I),
    # "minimum of 3 years", "at least three years"
    re.compile(r"(?:minimum|at least|min\.?)\s*(?:of\s*)?" + _NUM + r"\s*\+?\s*(?:years?|yrs?)", re.I),
    # "5+ years", "5 years", "five (5) years"
    re.compile(_NUM + r"\s*(?:\(\s*\d{1,2}\s*\))?\s*\+?\s*(?:years?|yrs?)", re.I),
]
EXPERIENCE_CTX = re.compile(r"experience|exp\b|working in|in a (?:regulated|medical|manufacturing)|industry", re.I)
NOT_EXPERIENCE_CTX = re.compile(
    r"years? (?:old|of age)|founded|history|since \d{4}|legacy|in business|serving|anniversary|"
    r"warranty|retention|record retention|over the (?:past|last)|for (?:more than|over) \d+ years|"
    r"\d+\s*\+?\s*years? (?:of (?:innovation|service|history|excellence))|tenure|vest",
    re.I,
)
PREFERRED_CTX = re.compile(r"prefer|nice to have|bonus|desired|plus\b|ideal|advantage", re.I)
REQUIRED_HEAD = re.compile(r"(required|minimum|basic|must have|requirements|qualifications|what you.ll need|you have)", re.I)
NO_EXPERIENCE = re.compile(
    r"\b(no (?:prior )?experience (?:is )?(?:required|necessary)|new (?:college )?grad(?:uate)?s? (?:are )?(?:welcome|encouraged)|"
    r"0\s*(?:-|–|to)\s*[12]\s*years?|recent (?:college )?graduates?)\b",
    re.I,
)


def _num(token: str) -> int:
    token = token.lower()
    return WORD_NUMBERS[token] if token in WORD_NUMBERS else int(token)


def _sentences(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = line.strip(" -•*\t")
        if not line:
            continue
        lines.extend(s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", line) if s.strip())
    return lines


def extract_years(text: str | None) -> tuple[int | None, int | None, bool]:
    """Return (min required years, min preferred years, explicit no-experience signal)."""
    if not text:
        return None, None, False
    required: list[int] = []
    preferred: list[int] = []
    in_preferred_section = False
    for sent in _sentences(text):
        low = sent.lower()
        is_heading = len(sent) < 60 and not re.search(r"\d|\byears?\b|\byrs?\b", low)
        if is_heading:
            if PREFERRED_CTX.search(low):
                in_preferred_section = True
            elif REQUIRED_HEAD.search(low):
                in_preferred_section = False
            continue
        if NOT_EXPERIENCE_CTX.search(low) or not EXPERIENCE_CTX.search(low):
            continue
        found: list[int] = []
        scope = sent
        # "Bachelor's with 2 years, or Master's with 0": use the bachelor's requirement.
        if re.search(r"bachelor|\bb\.?s\.?\b", low) and re.search(r"master|\bm\.?s\.?\b|ph\.?d", low):
            bs = re.search(r"(bachelor|\bb\.?s\.?\b)", sent, re.I)
            nxt = re.search(r"(master|\bm\.?s\.?\b|ph\.?d)", sent[bs.end():], re.I) if bs else None
            if bs:
                scope = sent[bs.start(): bs.end() + nxt.start()] if nxt else sent[bs.start():]
        for pat in YEARS_PATTERNS:
            for m in pat.finditer(scope):
                found.append(_num(m.group(1)))
            if found:
                break  # first (most specific) pattern wins for this sentence
        found = [n for n in found if 0 <= n <= 25]
        if not found:
            continue
        bucket = preferred if (in_preferred_section or PREFERRED_CTX.search(low)) else required
        bucket.append(min(found))
    no_exp = bool(NO_EXPERIENCE.search(text))
    return (min(required) if required else None, min(preferred) if preferred else None, no_exp)


# --------------------------------------------------------------------------- role family & engineering

ENGINEERING_TITLE = _RE(r"\b(engineer|engineering|developer|programmer|sre)\b")
FAMILY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("facilities_maintenance", _RE(r"\b(building|stationary|chief|facilit(?:y|ies)|hvac|plant operations|property)\s+engineer|engineer\s*[-,(]\s*(?:hotel|resort|building|property)|\bmaintenance (?:engineer|mechanic|technician)")),
    ("sales_engineering", _RE(r"\b(sales|solutions?|pre-?sales|presales|business development|account)\s+engineer")),
    ("support", _RE(r"\b(support|customer success|help ?desk|desktop|it support|technical support)\s+engineer")),
    ("technician", _RE(r"\btechnician\b|\btech\b")),
    ("applications_field", _RE(r"\b(field service|field|service|applications?|application|clinical|installation|commissioning)\s+engineer")),
    ("software", _RE(r"\b(software|firmware|embedded software|full[- ]?stack|front[- ]?end|back[- ]?end|devops|data|machine learning|ml|ai|cloud|platform|site reliability|security|web|mobile|ios|android|qa automation|sdet)\b|\bdeveloper\b|\bprogrammer\b|\bsre\b")),
    ("quality_reliability", _RE(r"\b(quality|reliability|supplier quality|design quality|capa|complaint|post[- ]market|compliance)\b")),
    ("test_verification", _RE(r"\b(test|testing|verification|validation|v&v)\b")),
    ("systems", _RE(r"\bsystems?\s+engineer")),
    ("manufacturing_process", _RE(r"\b(manufacturing|process|production|industrialization|npi|operations|packaging|sustaining|automation|industrial|tooling|equipment|maintenance)\b")),
    ("electrical", _RE(r"\b(electrical|electronics?|hardware|pcb|power|rf|analog|fpga|asic)\b")),
    ("regulatory", _RE(r"\bregulatory\b")),
    ("design_rd", _RE(r"\b(design|r&d|research and development|product development|development|mechanical|biomedical|new product|mechatronics|materials)\b")),
]
HOSPITALITY = _RE(r"\b(hotel|resort|guest rooms?|hospitality|casino|front desk|housekeeping|property management)\b")

DISCIPLINE_RULES: dict[str, re.Pattern[str]] = {
    "mechanical": _RE(r"\bmechanical\b|\bmechatronics\b"),
    "electrical": _RE(r"\belectrical\b|\belectronics?\b|\bEE\b"),
    "biomedical": _RE(r"\bbiomedical\b|\bbioengineering\b|\bBME\b"),
    "software": _RE(r"\bsoftware\b|\bcomputer science\b|\bcomputer engineering\b|\bfirmware\b"),
    "chemical": _RE(r"\bchemical engineering\b|\bchemical engineer\b"),
    "industrial": _RE(r"\bindustrial engineering\b|\bindustrial engineer\b"),
    "materials": _RE(r"\bmaterials? (?:science|engineering)\b|\bmaterials engineer\b"),
    "aerospace": _RE(r"\baerospace\b|\baeronautical\b"),
    "civil": _RE(r"\bcivil engineering\b|\bcivil engineer\b|\bstructural engineer\b"),
    "systems": _RE(r"\bsystems engineering\b|\bsystems engineer\b"),
}
FAMILY_DISCIPLINE = {
    "software": "software",
    "electrical": "electrical",
    "systems": "systems",
}

# --------------------------------------------------------------------------- other facts

CLEARANCE = _RE(
    r"(security clearance|secret clearance|top secret|ts/sci|\bts\b/|clearance (?:is )?required|"
    r"(?:ability|able|eligible) to obtain (?:and maintain )?(?:a |an )?(?:u\.?s\.? )?(?:government |dod |secret |active )*clearance|"
    r"active (?:secret |dod )?clearance|dod clearance|polygraph)"
)
ITAR = _RE(r"(\bitar\b|export control|\bu\.?s\.? persons?\b|\bu\.?s\.? citizenship (?:is )?required|must be a u\.?s\.? citizen)")
INDUSTRY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("defense", _RE(r"\b(defense|defence|dod|department of defense|missile|munitions|weapon|warfighter|national security|military)\b")),
    ("medical_devices", _RE(r"\b(medical device|iso 13485|21 cfr 820|fda|patients?|clinical|surgical|implant|diagnostic|healthcare)\b")),
    ("aerospace", _RE(r"\b(aircraft|aerospace|satellite|spacecraft|launch vehicle|faa|avionics)\b")),
    ("semiconductor", _RE(r"\b(semiconductor|wafer|fab\b|lithography)\b")),
    ("automotive", _RE(r"\b(automotive|vehicle|ev battery|powertrain)\b")),
    ("hospitality", HOSPITALITY),
    ("energy", _RE(r"\b(solar|wind turbine|oil and gas|utility|grid|energy storage)\b")),
    ("software", _RE(r"\b(saas|software company|cloud platform)\b")),
]
DEGREE_PHD = _RE(r"\bph\.?d\.? (?:is )?required|\bdoctorate required")
DEGREE_MS = _RE(r"\bmaster'?s degree (?:is )?required|\bms required|\bm\.s\. required")
DEGREE_BS = _RE(r"\b(bachelor'?s?|b\.?s\.?|b\.?a\.?|bsme|bsee|bse|undergraduate degree|4[- ]year degree)\b")
DEGREE_NONE = _RE(r"\b(high school diploma|ged)\b")

REMOTE = _RE(r"\b(fully remote|100% remote|remote[- ]first|work from home|remote\b)")
HYBRID = _RE(r"\bhybrid\b")
ONSITE = _RE(r"\b(on-?site|in[- ]office|in[- ]person)\b")

_MONEY = r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?\s?[kK]?)"
SALARY_RANGE = re.compile(_MONEY + r"\s*(?:-|–|—|to)\s*" + _MONEY + r"(\s*(?:/|per)\s*(?:hr|hour|year|yr|annum|annually))?", re.I)
HOURLY_HINT = re.compile(r"(/|per)\s*(hr|hour)|hourly", re.I)

TRAVEL = _RE(r"travel(?:ing)?\s+(?:up to\s+)?(\d{2})\s*%|(\d{2})\s*%\s+travel")
CONTRACT = _RE(r"\b(contract(?:or)?[- ]to[- ]hire|temporary|temp\b|1099|w2 contract|contract position|contract role|\d+[- ]month contract)\b")
STAFFING = _RE(r"\b(our client|on behalf of (?:our|a) client|staffing agency)\b")


def _money(token: str) -> float:
    token = token.replace(",", "").replace(" ", "").lower()
    mult = 1000 if token.endswith("k") else 1
    return float(token.rstrip("k")) * mult


def extract_salary(text: str) -> tuple[int | None, int | None]:
    m = SALARY_RANGE.search(text)
    if not m:
        return None, None
    lo, hi = _money(m.group(1)), _money(m.group(2))
    tail = m.group(3) or text[m.end(): m.end() + 30]
    if HOURLY_HINT.search(tail) or (hi < 300):
        lo, hi = lo * 2080, hi * 2080
    if lo > hi:
        lo, hi = hi, lo
    if not (15_000 <= lo <= 1_000_000 and 15_000 <= hi <= 1_500_000):
        return None, None
    return int(lo), int(hi)


def classify_role(title: str, description: str, industry_hint: str | None) -> tuple[bool, str, str | None]:
    """Return (is_engineering, role_family, rule_name)."""
    t = title or ""
    if not ENGINEERING_TITLE.search(t):
        if re.search(r"\bscientist\b", t, re.I) and re.search(r"\b(r&d|research|materials|polymer|biomaterials)\b", t + " " + description[:500], re.I):
            return False, "research_science", "scientist"
        if re.search(r"\btechnician\b", t, re.I):
            return False, "technician", "technician"
        if re.search(r"\bregulatory\b", t, re.I):
            return False, "regulatory", "regulatory"
        return False, "non_engineering", None
    if re.search(r"\brecruit|\btalent\b|\bsourcer\b", t, re.I):
        return False, "non_engineering", "recruiter"
    hospitality = industry_hint == "hospitality" or bool(HOSPITALITY.search(t)) or len(HOSPITALITY.findall(description[:3000])) >= 2
    for family, pattern in FAMILY_RULES:
        if pattern.search(t):
            if family == "facilities_maintenance":
                # "Maintenance Engineer" at a factory with an engineering degree
                # requirement is real engineering; at a hotel it is not.
                if (not hospitality and re.search(r"\bmaintenance\b", t, re.I)
                        and DEGREE_BS.search(description) and re.search(r"engineering", description, re.I)):
                    return True, "manufacturing_process", "maintenance_engineering"
                return False, family, "facilities"
            if family == "technician":
                return False, family, "technician"
            if family == "sales_engineering":
                return False, family, "sales"
            if hospitality and family in {"manufacturing_process", "other_engineering"}:
                return False, "facilities_maintenance", "hospitality"
            return True, family, family
    if hospitality:
        return False, "facilities_maintenance", "hospitality"
    return True, "other_engineering", "engineer"


def extract(title: str, description: str | None, location: str | None = None,
            industry_hint: str | None = None, extra: dict[str, Any] | None = None) -> Facts:
    desc = description or ""
    extra = extra or {}
    facts = Facts()
    sig: dict[str, Any] = {}
    confidence = 0.4

    # Role family / engineering
    is_eng, family, rule = classify_role(title, desc, industry_hint)
    facts.is_engineering, facts.role_family = is_eng, family
    sig["role_rule"] = rule
    if rule and rule not in {"engineer"}:
        confidence += 0.15

    # Years
    req_years, pref_years, no_exp = extract_years(desc)
    facts.min_years = req_years
    if req_years is None and no_exp:
        facts.min_years = 0
    sig["years_required"], sig["years_preferred"], sig["no_experience_signal"] = req_years, pref_years, no_exp
    if facts.min_years is not None:
        confidence += 0.15

    # Seniority: explicit title wins; years correct an unclear or mislabeled title.
    title_level, token = title_seniority(title)
    sig["title_seniority"], sig["title_token"] = title_level, token
    seniority = title_level
    years = facts.min_years
    if title_level in {"unknown", "entry", "mid"} and years is not None:
        by_years = "entry" if years <= 1 else "mid" if years <= 4 else "senior"
        if title_level == "unknown":
            seniority = by_years
        elif title_level == "entry" and years >= 3:
            facts.red_flags.append(f"Titled entry-level but asks for {years}+ years")
            seniority = by_years
        elif title_level == "mid" and years >= 5:
            seniority = "senior"
    elif title_level == "unknown" and no_exp:
        seniority = "entry"
    if title_level != "unknown":
        confidence += 0.2
    facts.seniority = seniority

    # Disciplines
    disc = {name for name, pat in DISCIPLINE_RULES.items() if pat.search(title)}
    if not disc and is_eng:
        head = desc[:6000]
        disc = {name for name, pat in DISCIPLINE_RULES.items() if pat.search(head)}
    if family in FAMILY_DISCIPLINE:
        disc.add(FAMILY_DISCIPLINE[family])
    facts.disciplines = sorted(disc)

    # Industry
    if industry_hint:
        facts.industry = industry_hint
    else:
        scores = {name: len(pat.findall(desc[:8000])) for name, pat in INDUSTRY_RULES}
        best = max(scores.items(), key=lambda kv: kv[1])
        facts.industry = best[0] if best[1] >= 2 else None

    # Clearance / export control
    facts.requires_clearance = bool(CLEARANCE.search(desc))
    if ITAR.search(desc):
        facts.red_flags.append("Export-controlled (ITAR / U.S. person required)")

    # Degree
    if DEGREE_PHD.search(desc):
        facts.degree_required = "phd"
    elif DEGREE_MS.search(desc):
        facts.degree_required = "masters"
    elif DEGREE_BS.search(desc):
        facts.degree_required = "bachelors"
    elif DEGREE_NONE.search(desc):
        facts.degree_required = "none"

    # Work mode
    loc = location or ""
    wt = (extra.get("workplace_type") or extra.get("remote_type") or "").lower()
    if "remote" in wt or re.search(r"\bremote\b", loc, re.I):
        facts.work_mode = "remote"
    elif "hybrid" in wt or re.search(r"\bhybrid\b", loc, re.I):
        facts.work_mode = "hybrid"
    elif "onsite" in wt or "on-site" in wt or "on site" in wt:
        facts.work_mode = "onsite"
    elif HYBRID.search(desc[:5000]):
        facts.work_mode = "hybrid"
    elif re.search(r"\b(fully remote|100% remote|remote[- ]first|this (?:is a )?remote (?:role|position))\b", desc, re.I):
        facts.work_mode = "remote"
    elif ONSITE.search(desc[:5000]):
        facts.work_mode = "onsite"

    # Salary
    salary = extra.get("salary") or {}
    if isinstance(salary, dict) and salary.get("min") and salary.get("max"):
        lo, hi = float(salary["min"]), float(salary["max"])
        if str(salary.get("interval", "")).lower().startswith("per-hour") or hi < 300:
            lo, hi = lo * 2080, hi * 2080
        facts.salary_min, facts.salary_max = int(lo), int(hi)
    else:
        comp = extra.get("compensation") or ""
        facts.salary_min, facts.salary_max = extract_salary(f"{comp}\n{desc}")

    # Red flags
    if m := TRAVEL.search(desc):
        pct = int(m.group(1) or m.group(2))
        if pct >= 50:
            facts.red_flags.append(f"Travel up to {pct}%")
    if CONTRACT.search(title + "\n" + desc[:4000]):
        facts.red_flags.append("Contract / temporary role")
    if STAFFING.search(desc[:3000]):
        facts.red_flags.append("Posted by a staffing agency")

    facts.confidence = round(min(confidence, 0.95), 2)
    facts.signals = sig
    return facts


def title_worth_reading(title: str) -> bool:
    """Cheap gate: is this title worth fetching details / spending model tokens on?"""
    t = title or ""
    return bool(ENGINEERING_TITLE.search(t) or re.search(r"\b(scientist|technician|regulatory|intern)\b", t, re.I))
