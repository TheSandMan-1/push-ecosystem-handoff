"""ORM models.

Design rule that makes the business work: everything expensive happens once
per *job* (fetching, extraction, verification) and is shared by every user.
Per-user work (matching) is cheap, deterministic filtering over stored facts.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Source(Base):
    """One company careers board on one applicant tracking system."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    ats: Mapped[str] = mapped_column(String(32))
    company_name: Mapped[str] = mapped_column(String(200))
    token: Mapped[str] = mapped_column(String(200))
    workday_host: Mapped[str | None] = mapped_column(String(200), nullable=True)
    workday_site: Mapped[str | None] = mapped_column(String(200), nullable=True)
    careers_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(80), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_job_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    jobs: Mapped[list["Job"]] = relationship(back_populates="source")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_job_source_external"),
        Index("ix_jobs_open", "closed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(400))
    company_name: Mapped[str] = mapped_column(String(200))
    location: Mapped[str | None] = mapped_column(String(400), nullable=True)
    department: Mapped[str | None] = mapped_column(String(200), nullable=True)
    url: Mapped[str] = mapped_column(String(1000))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    missed_runs: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    source: Mapped[Source] = relationship(back_populates="jobs")
    facts: Mapped["JobFacts | None"] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


class JobFacts(Base):
    """Structured facts extracted from one posting (once, shared by all users)."""

    __tablename__ = "job_facts"

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    seniority: Mapped[str] = mapped_column(String(20), default="unknown", index=True)
    min_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_engineering: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    role_family: Mapped[str] = mapped_column(String(40), default="other")
    disciplines: Mapped[list[str]] = mapped_column(JSON, default=list)
    industry: Mapped[str | None] = mapped_column(String(80), nullable=True)
    requires_clearance: Mapped[bool] = mapped_column(Boolean, default=False)
    degree_required: Mapped[str] = mapped_column(String(20), default="unknown")
    work_mode: Mapped[str] = mapped_column(String(20), default="unknown")
    salary_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    red_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    extractor: Mapped[str] = mapped_column(String(120), default="heuristic")
    heuristic: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    verified_status: Mapped[str] = mapped_column(String(20), default="none", index=True)
    verifier: Mapped[str | None] = mapped_column(String(120), nullable=True)
    verifier_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    job: Mapped[Job] = relationship(back_populates="facts")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    unsubscribe_token: Mapped[str] = mapped_column(
        String(64), default=lambda: secrets.token_urlsafe(24), unique=True
    )

    profile: Mapped["Profile | None"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    target_seniorities: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["entry"])
    max_years: Mapped[int] = mapped_column(Integer, default=2)
    disciplines: Mapped[list[str]] = mapped_column(JSON, default=list)
    role_families: Mapped[list[str]] = mapped_column(JSON, default=list)
    excluded_role_families: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: ["facilities_maintenance", "sales_engineering"]
    )
    locations: Mapped[list[str]] = mapped_column(JSON, default=list)
    remote_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    exclude_clearance: Mapped[bool] = mapped_column(Boolean, default=True)
    require_engineering: Mapped[bool] = mapped_column(Boolean, default=True)
    salary_floor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    include_keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    exclude_keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    muted_companies: Mapped[list[str]] = mapped_column(JSON, default=list)
    digest_frequency: Mapped[str] = mapped_column(String(10), default="weekly")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="profile")


class UserJob(Base):
    """A user's action on a job: saved, applied, or hidden (with a reason)."""

    __tablename__ = "user_jobs"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    hide_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Run(Base):
    """One pipeline run (ingest, extract, digest) for the admin dashboard."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class LLMCall(Base):
    """Every model call, with tokens and cost, so unit economics are measured, not guessed."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    purpose: Mapped[str] = mapped_column(String(20))
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(120))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class DigestSend(Base):
    __tablename__ = "digest_sends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    job_count: Mapped[int] = mapped_column(Integer, default=0)
    transport: Mapped[str] = mapped_column(String(20))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    job_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
