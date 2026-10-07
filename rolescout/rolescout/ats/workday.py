"""Workday candidate-experience API (the JSON behind *.myworkdayjobs.com).

List:   POST https://{host}/wday/cxs/{tenant}/{site}/jobs
Detail: GET  https://{host}/wday/cxs/{tenant}/{site}{externalPath}

The list endpoint has no descriptions, so details are fetched lazily by ingest,
only for new jobs whose title is worth reading. That keeps request volume low.
"""

from __future__ import annotations

import httpx

from ..config import get_settings
from .base import FetchError, JobList, RawJob, SourceSpec, get_json, html_to_text, parse_iso, parse_relative_posted, post_json


def _tenant(host: str) -> str:
    return host.split(".")[0]


class WorkdayConnector:
    name = "workday"

    def _base(self, spec: SourceSpec) -> str:
        if not spec.workday_host or not spec.workday_site:
            raise FetchError("Workday source needs workday_host and workday_site")
        return f"https://{spec.workday_host}/wday/cxs/{_tenant(spec.workday_host)}/{spec.workday_site}"

    def fetch(self, spec: SourceSpec, client: httpx.Client) -> list[RawJob]:
        settings = get_settings()
        base = self._base(spec)
        limit = settings.workday_page_limit
        offset = 0
        total: int | None = None
        jobs = JobList()
        seen: set[str] = set()
        while offset < settings.workday_max_jobs:
            data = post_json(
                client,
                f"{base}/jobs",
                {"appliedFacets": {}, "limit": limit, "offset": offset, "searchText": settings.workday_search_text},
            )
            postings = data.get("jobPostings")
            if postings is None:
                raise FetchError("Workday response missing 'jobPostings'")
            if total is None:
                # Workday only reports a reliable total on the first page.
                total = int(data.get("total") or 0)
            for item in postings:
                path = item.get("externalPath")
                if not path or path in seen:
                    continue
                seen.add(path)
                bullets = item.get("bulletFields") or []
                jobs.append(
                    RawJob(
                        external_id=bullets[0] if bullets else path,
                        title=(item.get("title") or "").strip(),
                        url=f"https://{spec.workday_host}/en-US/{spec.workday_site}{path}",
                        location=item.get("locationsText") or None,
                        posted_at=parse_relative_posted(item.get("postedOn")),
                        detail_ref=path,
                    )
                )
            offset += limit
            if not postings or offset >= (total or 0):
                break
        else:
            jobs.complete = False  # hit workday_max_jobs before the end of the board
        if total and len(seen) < total and offset < total:
            jobs.complete = False
        return jobs

    def fetch_detail(self, spec: SourceSpec, job: RawJob, client: httpx.Client) -> RawJob:
        data = get_json(client, f"{self._base(spec)}{job.detail_ref}")
        info = data.get("jobPostingInfo") or {}
        job.description = html_to_text(info.get("jobDescription")) or job.description
        if info.get("location"):
            extra_locs = [loc for loc in (info.get("additionalLocations") or []) if loc]
            job.location = "; ".join([info["location"], *extra_locs])
        if info.get("startDate"):
            job.posted_at = parse_iso(info["startDate"]) or job.posted_at
        if info.get("externalUrl"):
            job.url = info["externalUrl"]
        job.extra["time_type"] = info.get("timeType")
        job.extra["remote_type"] = info.get("remoteType")
        return job
