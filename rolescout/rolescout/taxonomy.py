"""Shared vocabulary for seniority, role families and disciplines."""

SENIORITIES: dict[str, str] = {
    "intern": "Intern / co-op",
    "entry": "Entry level (0-2 yrs)",
    "mid": "Mid level (2-5 yrs)",
    "senior": "Senior / staff / principal",
    "lead_manager": "Lead / manager / director",
    "unknown": "Unclear",
}

ROLE_FAMILIES: dict[str, str] = {
    "design_rd": "Design / R&D / product development",
    "manufacturing_process": "Manufacturing / process / production",
    "quality_reliability": "Quality / reliability",
    "test_verification": "Test / verification & validation",
    "systems": "Systems engineering",
    "software": "Software / firmware / data",
    "electrical": "Electrical / electronics / hardware",
    "applications_field": "Applications / field / service engineering",
    "support": "Technical support",
    "sales_engineering": "Sales / solutions engineering",
    "regulatory": "Regulatory affairs",
    "research_science": "Research science",
    "technician": "Technician",
    "facilities_maintenance": "Facilities / building maintenance",
    "other_engineering": "Other engineering",
    "non_engineering": "Not an engineering role",
}

DISCIPLINES: dict[str, str] = {
    "mechanical": "Mechanical",
    "electrical": "Electrical",
    "biomedical": "Biomedical",
    "software": "Software / computer",
    "chemical": "Chemical",
    "industrial": "Industrial",
    "materials": "Materials",
    "aerospace": "Aerospace",
    "civil": "Civil",
    "systems": "Systems",
}

DEGREES = ["none", "bachelors", "masters", "phd", "unknown"]
WORK_MODES = ["onsite", "hybrid", "remote", "unknown"]

HIDE_REASONS: dict[str, str] = {
    "too_senior": "Too senior",
    "not_engineering": "Not a real engineering job",
    "wrong_field": "Wrong field / discipline",
    "wrong_location": "Wrong location",
    "wrong_industry": "Industry I don't want",
    "closed": "Posting was closed",
    "not_interested": "Just not interested",
}

SENIORITY_RANK = {"intern": 0, "entry": 1, "mid": 2, "senior": 3, "lead_manager": 4}
