"""Structured-output schemas shared by every model provider."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..taxonomy import DEGREES, DISCIPLINES, ROLE_FAMILIES, SENIORITIES, WORK_MODES

Seniority = Literal["intern", "entry", "mid", "senior", "lead_manager", "unknown"]


class ExtractedFacts(BaseModel):
    seniority: Seniority
    min_years: int | None = Field(description="Minimum REQUIRED years of experience; null if not stated")
    is_engineering: bool
    role_family: str
    disciplines: list[str]
    industry: str | None
    requires_clearance: bool
    degree_required: str
    work_mode: str
    salary_min: int | None
    salary_max: int | None
    summary: str
    red_flags: list[str]
    confidence: float

    @field_validator("role_family")
    @classmethod
    def _family(cls, v: str) -> str:
        return v if v in ROLE_FAMILIES else "other_engineering"

    @field_validator("disciplines")
    @classmethod
    def _disc(cls, v: list[str]) -> list[str]:
        return sorted({d for d in v if d in DISCIPLINES})

    @field_validator("degree_required")
    @classmethod
    def _degree(cls, v: str) -> str:
        return v if v in DEGREES else "unknown"

    @field_validator("work_mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return v if v in WORK_MODES else "unknown"

    @field_validator("min_years")
    @classmethod
    def _years(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return v if 0 <= v <= 30 else None

    @field_validator("confidence")
    @classmethod
    def _conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("summary")
    @classmethod
    def _summary(cls, v: str) -> str:
        return (v or "").strip()[:300]

    @field_validator("red_flags")
    @classmethod
    def _flags(cls, v: list[str]) -> list[str]:
        return [f.strip()[:120] for f in v if f and f.strip()][:6]


class VerifyResult(BaseModel):
    verdict: Literal["agree", "corrected"]
    notes: str
    facts: ExtractedFacts


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def facts_json_schema() -> dict:
    """Hand-written JSON schema (portable across Claude structured outputs,
    Ollama, LM Studio and llama.cpp; avoids numeric constraints some reject)."""
    return {
        "type": "object",
        "properties": {
            "seniority": {"type": "string", "enum": list(SENIORITIES)},
            "min_years": _nullable({"type": "integer"}),
            "is_engineering": {"type": "boolean"},
            "role_family": {"type": "string", "enum": list(ROLE_FAMILIES)},
            "disciplines": {"type": "array", "items": {"type": "string", "enum": list(DISCIPLINES)}},
            "industry": _nullable({"type": "string"}),
            "requires_clearance": {"type": "boolean"},
            "degree_required": {"type": "string", "enum": DEGREES},
            "work_mode": {"type": "string", "enum": WORK_MODES},
            "salary_min": _nullable({"type": "integer"}),
            "salary_max": _nullable({"type": "integer"}),
            "summary": {"type": "string"},
            "red_flags": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": [
            "seniority", "min_years", "is_engineering", "role_family", "disciplines", "industry",
            "requires_clearance", "degree_required", "work_mode", "salary_min", "salary_max",
            "summary", "red_flags", "confidence",
        ],
        "additionalProperties": False,
    }


def verify_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["agree", "corrected"]},
            "notes": {"type": "string"},
            "facts": facts_json_schema(),
        },
        "required": ["verdict", "notes", "facts"],
        "additionalProperties": False,
    }
