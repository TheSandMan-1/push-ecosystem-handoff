"""Fetch every enabled board, upsert jobs, and track whether each job is still open.

Liveness rule: a job is closed only after it is missing from a *successful*
fetch of its board `close_after_missed_runs` times in a row. A failed fetch, or
a fetch that suddenly returns zero jobs for a board that had many, never closes
anything (that is an outage, not a hiring freeze).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .ats import LAZY_DETAIL, DetectedSource, FetchError, RawJob, SourceSpec, detect_source, get_connector
from .config import Settings, get_settings
from .extract.heuristics import title_worth_reading
from .models import Job, Run, Source, utcnow

log = logging.getLogger(__name__)
SUSPICIOUS_EMPTY_MIN = 5


@dataclass
class SourceResult:
    slug: str
    ok: bool
    fetched: int = 0
    new: int = 0
    updated: int = 0
    reopened: int = 0
    closed: int = 0
    details_fetched: int = 0
    detail_errors: int = 0
    error: str | None = None


@dataclass
class IngestStats:
    sources_ok: int = 0
    sources_failed: int = 0
    fetched: int = 0
    new: int = 0
    updated: int = 0
    reopened: int = 0
    closed: int = 0
    details_fetched: int = 0
    per_source: list[SourceResult] = field(default_factory=list)

    def add(self, r: SourceResult) -> None:
        self.per_source.append(r)
        if r.ok:
            self.sources_ok += 1
        else:
            self.sources_failed += 1
        for name in ("fetched", "new", "updated", "reopened", "closed", "details_fetched"):
            setattr(self, name, getattr(self, name) + getattr(r, name))

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "per_source"}
        d["failures"] = [{"source": r.slug, "error": r.error} for r in self.per_source if not r.ok][:30]
        return d


def make_client(settings: Settings | None = None) -> httpx.Client:
    settings = settings or get_settings()
    return httpx.Client(
        timeout=settings.http_timeout,
        headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
        follow_redirects=True,
    )


def spec_for(source: Source) -> SourceSpec:
    return SourceSpec(
        ats=source.ats,
        token=source.token,
        company_name=source.company_name,
        workday_host=source.workday_host,
        workday_site=source.workday_site,
    )


def _fetch_details(source: Source, raws: list[RawJob], client: httpx.Client, result: SourceResult) -> None:
    if not raws:
        return
    connector = get_connector(source.ats)
    spec = spec_for(source)

    def one(raw: RawJob) -> RawJob | None:
        try:
            return connector.fetch_detail(spec, raw, client)
        except FetchError as exc:
            log.warning("detail fetch failed for %s %s: %s", source.slug, raw.external_id, exc)
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for out in pool.map(one, raws):
            if out is None:
                result.detail_errors += 1
            else:
                result.details_fetched += 1


def _clean_extra(extra: dict | None) -> dict | None:
    cleaned = {k: v for k, v in (extra or {}).items() if v not in (None, "", {}, [])}
    return cleaned or None


def ingest_source(session: Session, source: Source, client: httpx.Client, settings: Settings) -> SourceResult:
    result = SourceResult(slug=source.slug, ok=False)
    try:
        connector = get_connector(source.ats)
    except ValueError as exc:
        source.last_status, source.last_error = "error", str(exc)
        result.error = str(exc)
        return result
    now = utcnow()
    # All network I/O happens before this source's rows are touched, so no SQLite
    # write lock is held while we wait on HTTP (logins and saves stay responsive).
    try:
        raws = connector.fetch(spec_for(source), client)
    except (FetchError, ValueError) as exc:
        source.last_fetched_at = now
        source.last_status, source.last_error = "error", str(exc)[:1000]
        result.error = str(exc)
        return result
    complete = getattr(raws, "complete", True)

    with session.no_autoflush:
        existing: dict[str, Job] = {
            j.external_id: j for j in session.scalars(select(Job).where(Job.source_id == source.id))
        }
    open_before = sum(1 for j in existing.values() if j.closed_at is None)
    if not raws and open_before >= SUSPICIOUS_EMPTY_MIN:
        source.last_fetched_at = now
        msg = f"Board returned 0 jobs but had {open_before} open; treating as an outage, nothing closed"
        source.last_status, source.last_error = "error", msg
        result.error = msg
        return result

    # De-duplicate within the feed (some boards list one req in several locations).
    unique: dict[str, RawJob] = {}
    for raw in raws:
        if raw.external_id and raw.title and raw.url:
            unique.setdefault(raw.external_id, raw)
    result.fetched = len(unique)

    if source.ats in LAZY_DETAIL:
        need = [
            raw for ext_id, raw in unique.items()
            if title_worth_reading(raw.title)
            and (ext_id not in existing or not existing[ext_id].description
                 or existing[ext_id].title != raw.title)
        ]
        # End the read transaction before the slow HTTP phase.
        session.commit()
        _fetch_details(source, need, client, result)
    source.last_fetched_at = now

    for ext_id, raw in unique.items():
        job = existing.get(ext_id)
        if job is None:
            job = Job(
                source_id=source.id,
                external_id=ext_id,
                title=raw.title[:400],
                company_name=source.company_name,
                location=(raw.location or None) and raw.location[:400],
                department=(raw.department or None) and raw.department[:200],
                url=raw.url[:1000],
                description=raw.description,
                posted_at=raw.posted_at,
                first_seen_at=now,
                last_seen_at=now,
                content_hash=raw.content_hash(),
                extra=_clean_extra(raw.extra),
            )
            session.add(job)
            result.new += 1
            continue
        if raw.description is None and job.description:
            raw.description = job.description  # lazy-detail board, details not refetched
        new_hash = raw.content_hash()
        if job.closed_at is not None:
            job.closed_at = None
            result.reopened += 1
        job.missed_runs = 0
        job.last_seen_at = now
        if new_hash != job.content_hash:
            job.title = raw.title[:400]
            job.location = (raw.location or None) and raw.location[:400]
            job.department = (raw.department or None) and raw.department[:200]
            job.description = raw.description
            job.content_hash = new_hash
            result.updated += 1
        job.url = raw.url[:1000]
        job.company_name = source.company_name
        if raw.extra:
            job.extra = _clean_extra(raw.extra)
        if raw.posted_at and not job.posted_at:
            job.posted_at = raw.posted_at

    for ext_id, job in existing.items():
        if not complete:
            break  # partial read: absence proves nothing
        if ext_id in unique or job.closed_at is not None:
            continue
        job.missed_runs = (job.missed_runs or 0) + 1
        if job.missed_runs >= settings.close_after_missed_runs:
            job.closed_at = now
            result.closed += 1

    source.last_status = "ok"
    source.last_error = None if complete else "Board only partly read (page cap); closures skipped this run"
    source.last_job_count = result.fetched
    result.ok = True
    return result


def run_ingest(session: Session, settings: Settings | None = None, *, source_ids: list[int] | None = None,
               client: httpx.Client | None = None) -> IngestStats:
    settings = settings or get_settings()
    stats = IngestStats()
    run = Run(kind="ingest")
    session.add(run)
    session.commit()
    own_client = client is None
    client = client or make_client(settings)
    try:
        stmt = select(Source).where(Source.enabled.is_(True)).order_by(Source.id)
        if source_ids:
            stmt = stmt.where(Source.id.in_(source_ids))
        for source in session.scalars(stmt).all():
            try:
                result = ingest_source(session, source, client, settings)
                session.commit()
            except Exception as exc:  # one bad board must not stop the rest
                session.rollback()
                log.exception("ingest failed for %s", source.slug)
                result = SourceResult(slug=source.slug, ok=False, error=f"internal error: {exc}")
            stats.add(result)
            log.info("ingest %s: %s", source.slug, result)
    finally:
        if own_client:
            client.close()
    run.finished_at = utcnow()
    run.stats = stats.as_dict()
    run.status = "ok" if stats.sources_failed == 0 else ("partial" if stats.sources_ok else "error")
    session.commit()
    return stats


def upsert_source(session: Session, detected: DetectedSource, company_name: str,
                  industry: str | None = None, careers_url: str | None = None) -> tuple[Source, bool]:
    source = session.scalar(select(Source).where(Source.slug == detected.slug))
    created = source is None
    if source is None:
        source = Source(slug=detected.slug, ats=detected.ats, token=detected.token,
                        workday_host=detected.workday_host, workday_site=detected.workday_site,
                        company_name=company_name)
        session.add(source)
    source.company_name = company_name or source.company_name
    source.industry = industry or source.industry
    source.careers_url = careers_url or source.careers_url
    source.enabled = True
    session.flush()
    return source, created


def add_source_from_url(session: Session, url: str, company_name: str, industry: str | None = None) -> tuple[Source, bool]:
    return upsert_source(session, detect_source(url), company_name, industry, careers_url=url)


def probe_url(url: str, client: httpx.Client | None = None) -> tuple[DetectedSource, int]:
    """Detect and fetch a board without saving anything. Returns (source, job count)."""
    detected = detect_source(url)
    own = client is None
    client = client or make_client()
    try:
        spec = SourceSpec(ats=detected.ats, token=detected.token, company_name=detected.token,
                          workday_host=detected.workday_host, workday_site=detected.workday_site)
        jobs = get_connector(detected.ats).fetch(spec, client)
        return detected, len(jobs)
    finally:
        if own:
            client.close()
