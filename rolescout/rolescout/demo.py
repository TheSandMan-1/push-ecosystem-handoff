"""Fictional demo data so the app is explorable before any real boards are added.

Companies and postings are invented. Demo sources are disabled so ingest never
tries to fetch them.
"""

from __future__ import annotations

import random
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import create_user
from .models import Job, Profile, Source, User, UserJob, utcnow

COMPANIES = {
    "northwind": ("Northwind Medical", "medical_devices", "Carlsbad, CA"),
    "helix": ("Helix Surgical Robotics", "medical_devices", "San Diego, CA"),
    "coastline": ("Coastline Diagnostics", "medical_devices", "Irvine, CA"),
    "pacific": ("Pacific Orthopedics", "medical_devices", "Lake Forest, CA"),
    "summit": ("Summit Biologics", "biotech", "Oceanside, CA"),
    "ironclad": ("Ironclad Defense Systems", "defense", "San Diego, CA"),
    "vantage": ("Vantage Aerospace", "aerospace", "Long Beach, CA"),
    "brightline": ("Brightline Robotics", "robotics", "Austin, TX"),
    "lumen": ("Lumen Semiconductor", "semiconductor", "San Jose, CA"),
    "tidepool": ("Tidepool Software", "software", "Remote, US"),
    "seabreeze": ("Seabreeze Resort & Spa", "hospitality", "La Jolla, CA"),
}

DUTIES = {
    "design": "Design and develop mechanical components and assemblies for next-generation products. Build CAD models in SolidWorks, run tolerance stack-ups, and support prototype builds and design reviews.",
    "mfg": "Own manufacturing processes on the production floor: write work instructions, run process validations (IQ/OQ/PQ), drive yield improvements, and support new product introduction.",
    "quality": "Support design controls and quality systems: CAPA investigations, complaint analysis, risk management (FMEA), and supplier quality activities.",
    "test": "Plan and execute verification and validation testing, write protocols and reports, and maintain test fixtures and traceability.",
    "systems": "Define system requirements, manage interfaces between subsystems, and lead integration testing across mechanical, electrical and software teams.",
    "software": "Build and maintain production software services. Write clean, tested code, participate in code reviews and on-call, and ship features end to end.",
    "electrical": "Design analog and digital circuits, lay out PCBs, and bring up and debug hardware in the lab.",
    "field": "Install, maintain and troubleshoot equipment at customer sites. Provide technical support and training to customers.",
    "sales": "Partner with account executives to run technical demos, answer customer questions, and close deals.",
    "facilities": "Maintain guest rooms, HVAC, plumbing, pools and kitchen equipment. Respond to guest service calls and perform preventive maintenance.",
    "recruit": "Recruit top engineering talent: source candidates, run phone screens and manage the interview loop.",
    "tech": "Assemble and test products per work instructions, perform rework and maintain accurate production records.",
}

# title, company, years phrase, discipline, duties, extras
POSTINGS = [
    ("Associate Mechanical Engineer", "northwind", "0-2 years of experience in mechanical design, internships count.", "Mechanical or Biomedical Engineering", "design", {"salary": "$78,000 - $96,000", "days": 2}),
    ("R&D Engineer I", "helix", "New graduates are encouraged to apply.", "Mechanical Engineering", "design", {"salary": "$85,000 - $105,000", "days": 1}),
    ("Manufacturing Engineer I", "coastline", "1+ years of experience in a manufacturing environment (co-ops count).", "Mechanical, Industrial or Biomedical Engineering", "mfg", {"days": 4, "mode": "On-site"}),
    ("Quality Engineer I", "pacific", "0-2 years of experience in a regulated industry.", "an engineering discipline", "quality", {"days": 6}),
    ("Engineer I, Verification & Validation", "northwind", "Minimum of one year of experience with test methods.", "Biomedical or Electrical Engineering", "test", {"days": 9, "salary": "$80,000 - $98,000"}),
    ("Process Development Engineer", "summit", "1-3 years of experience in process development.", "Chemical or Biomedical Engineering", "mfg", {"days": 3, "mode": "Hybrid"}),
    ("Systems Engineer, New Grad", "helix", "Recent graduates welcome.", "Systems, Mechanical or Electrical Engineering", "systems", {"days": 12}),
    ("Associate Quality Engineer", "coastline", "1+ year of experience in quality engineering or a related internship.", "Biomedical Engineering", "quality", {"days": 15, "salary": "$75,000 - $90,000"}),
    ("Design Engineer", "pacific", "1-2 years of experience with SolidWorks and GD&T.", "Mechanical Engineering", "design", {"days": 5}),
    ("Manufacturing Engineer", "helix", "2+ years of experience supporting production of electromechanical devices.", "Mechanical Engineering", "mfg", {"days": 8}),
    ("Test Engineer", "vantage", "0-2 years of experience. Experience with LabVIEW or Python is a plus.", "Aerospace, Mechanical or Electrical Engineering", "test", {"days": 10, "salary": "$82,000 - $100,000"}),
    ("Electrical Engineer I", "lumen", "0-2 years of experience with circuit design.", "Electrical Engineering", "electrical", {"days": 7, "salary": "$110,000 - $135,000"}),
    ("Software Engineer, Early Career", "tidepool", "0-2 years of professional software experience.", "Computer Science or Software Engineering", "software", {"days": 3, "salary": "$105,000 - $130,000", "mode": "This is a fully remote role."}),
    ("Junior Robotics Engineer", "brightline", "1+ years of experience with ROS or embedded systems.", "Mechanical or Electrical Engineering", "design", {"days": 11}),
    ("Sustaining Engineer I", "northwind", "Minimum 1 year of experience in sustaining engineering or a related co-op.", "Mechanical or Biomedical Engineering", "mfg", {"days": 20}),
    ("Engineering Rotational Program - Mechanical", "pacific", "Open to graduating seniors and recent graduates.", "Mechanical Engineering", "design", {"days": 18, "salary": "$80,000 - $92,000"}),
    # Mid level that still fits a profile with max_years 2-3
    ("Quality Engineer II", "coastline", "2-4 years of experience in medical device quality.", "an engineering discipline", "quality", {"days": 6}),
    ("Mechanical Engineer II", "helix", "3+ years of experience in medical device design.", "Mechanical Engineering", "design", {"days": 9}),
    # Too senior
    ("Senior Mechanical Engineer", "northwind", "7+ years of experience in product development.", "Mechanical Engineering", "design", {"days": 2}),
    ("Staff Systems Engineer", "helix", "10+ years of experience in systems engineering.", "Systems Engineering", "systems", {"days": 5}),
    ("Principal Quality Engineer", "pacific", "12+ years of experience in quality.", "an engineering discipline", "quality", {"days": 14}),
    ("Engineering Manager, R&D", "coastline", "8+ years of experience including 3 years managing engineers.", "an engineering discipline", "design", {"days": 4}),
    ("Lead Manufacturing Engineer", "summit", "6+ years of experience in biologics manufacturing.", "Chemical Engineering", "mfg", {"days": 22}),
    ("Sr. Test Engineer", "vantage", "5+ years of experience in flight hardware test.", "Aerospace Engineering", "test", {"days": 3}),
    # Mislabeled: entry title, senior requirements
    ("Entry Level Design Engineer", "brightline", "Requirements: 5+ years of experience in product design.", "Mechanical Engineering", "design", {"days": 6}),
    ("Mechanical Engineer", "lumen", "Minimum of six years of experience in semiconductor equipment design.", "Mechanical Engineering", "design", {"days": 8}),
    # Defense / clearance
    ("Manufacturing Engineer I", "ironclad", "0-2 years of experience. Must be able to obtain and maintain a Secret security clearance. U.S. citizenship required (ITAR).", "Mechanical or Industrial Engineering", "mfg", {"days": 2}),
    ("Systems Engineer I", "ironclad", "1+ years of experience on DoD programs. Active Secret clearance required.", "Systems or Electrical Engineering", "systems", {"days": 5}),
    ("Test Engineer", "vantage", "1-2 years of experience. Ability to obtain a DoD security clearance. ITAR: U.S. person required.", "Aerospace Engineering", "test", {"days": 12}),
    # Not real engineering
    ("Maintenance Engineer", "seabreeze", "1+ years of hotel or resort maintenance experience. High school diploma or GED.", "", "facilities", {"days": 1}),
    ("Building Engineer", "seabreeze", "Experience with HVAC and plumbing preferred.", "", "facilities", {"days": 3}),
    ("Technical Recruiter, Engineering", "helix", "2+ years of experience recruiting engineers.", "", "recruit", {"days": 4}),
    ("Manufacturing Technician", "northwind", "1+ years of experience in assembly.", "", "tech", {"days": 2}),
    ("Sales Engineer", "lumen", "1+ years of experience in technical sales.", "Electrical Engineering", "sales", {"days": 7}),
    # Field service with heavy travel (eng, but flagged)
    ("Field Service Engineer", "coastline", "0-2 years of experience. Travel up to 70% within the territory.", "Biomedical or Electrical Engineering", "field", {"days": 5}),
    # Out-of-area entry roles (location filter)
    ("Associate Process Engineer", "lumen", "0-2 years of experience in a fab or lab.", "Chemical or Materials Engineering", "mfg", {"days": 9}),
    ("Mechanical Engineer I", "brightline", "0-2 years of experience with CAD.", "Mechanical Engineering", "design", {"days": 13}),
    # Older posting
    ("Associate R&D Engineer", "pacific", "0-2 years of experience.", "Biomedical Engineering", "design", {"days": 70}),
    # Intern
    ("Mechanical Engineering Intern - Summer", "helix", "Currently pursuing a BS in Mechanical Engineering.", "Mechanical Engineering", "design", {"days": 3}),
]


def _description(company: str, industry: str, years: str, discipline: str, duties: str, extras: dict) -> str:
    parts = [
        f"About {company}\n{company} builds products that matter. We are an equal opportunity employer.",
        f"What you'll do\n{DUTIES[duties]}",
        "Required qualifications",
    ]
    if discipline:
        parts.append(f"- Bachelor's degree in {discipline}.")
    elif duties in ("facilities", "tech"):
        parts.append("- High school diploma or GED.")
    parts.append(f"- {years}")
    if industry == "medical_devices":
        parts.append("- Familiarity with FDA 21 CFR 820 and ISO 13485 is a plus for patient-facing devices.")
    parts.append("Preferred qualifications\n- Experience with Minitab or Python.\n- 5+ years of experience in the industry is a bonus, not a requirement.")
    if extras.get("mode"):
        parts.append(extras["mode"] if "." in extras["mode"] else f"This role is {extras['mode'].lower()}.")
    if extras.get("salary"):
        parts.append(f"The base salary range for this position is {extras['salary']} per year.")
    return "\n\n".join(parts)


DEMO_EMAIL = "demo@rolescout.local"
DEMO_PASSWORD = "demo12345"


def load_demo(session: Session, *, seed: int = 7) -> dict:
    rng = random.Random(seed)
    now = utcnow()
    sources: dict[str, Source] = {}
    for key, (name, industry, _loc) in COMPANIES.items():
        slug = f"demo:{key}"
        src = session.scalar(select(Source).where(Source.slug == slug))
        if src is None:
            src = Source(slug=slug, ats="demo", token=key, company_name=name, industry=industry,
                         enabled=False, last_status="ok", last_fetched_at=now)
            session.add(src)
            session.flush()
        sources[key] = src
    created = 0
    for i, (title, key, years, discipline, duties, extras) in enumerate(POSTINGS):
        src = sources[key]
        ext_id = f"demo-{i:03d}"
        if session.scalar(select(Job).where(Job.source_id == src.id, Job.external_id == ext_id)):
            continue
        name, industry, loc = COMPANIES[key]
        days = extras.get("days", rng.randint(1, 30))
        desc = _description(name, industry, years, discipline, duties, extras)
        job = Job(source_id=src.id, external_id=ext_id, title=title, company_name=name, location=loc,
                  url=f"https://example.com/careers/{key}/{ext_id}", description=desc,
                  posted_at=now - timedelta(days=days, hours=rng.randint(0, 20)),
                  first_seen_at=now - timedelta(days=min(days, 14)),
                  last_seen_at=now - timedelta(minutes=rng.randint(5, 240)))
        from .ats.base import RawJob

        job.content_hash = RawJob(external_id=ext_id, title=title, url=job.url, location=loc, description=desc).content_hash()
        session.add(job)
        created += 1
    # A closed posting, to show liveness tracking.
    src = sources["coastline"]
    if not session.scalar(select(Job).where(Job.external_id == "demo-closed")):
        session.add(Job(source_id=src.id, external_id="demo-closed", title="Associate Design Engineer",
                        company_name=src.company_name, location="Irvine, CA", url="https://example.com/closed",
                        description="0-2 years of experience.", posted_at=now - timedelta(days=40),
                        first_seen_at=now - timedelta(days=40), last_seen_at=now - timedelta(days=3),
                        closed_at=now - timedelta(days=2), content_hash="closed"))
    user = session.scalar(select(User).where(User.email == DEMO_EMAIL))
    if user is None:
        user = create_user(session, DEMO_EMAIL, DEMO_PASSWORD, "Alex", is_admin=True)
        p: Profile = user.profile
        p.target_seniorities = ["entry"]
        p.max_years = 2
        p.disciplines = ["mechanical", "biomedical"]
        p.role_families = ["design_rd", "manufacturing_process", "quality_reliability"]
        p.locations = ["San Diego", "Carlsbad", "Oceanside", "Irvine", "Lake Forest", "Long Beach"]
        p.include_keywords = ["SolidWorks", "GD&T"]
    session.flush()
    first = session.scalar(select(Job).where(Job.title == "R&D Engineer I"))
    if first and not session.get(UserJob, (user.id, first.id)):
        session.add(UserJob(user_id=user.id, job_id=first.id, status="saved"))
    return {"jobs_created": created, "user": DEMO_EMAIL, "password": DEMO_PASSWORD}
