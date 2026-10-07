"""Ashby public job board API: api.ashbyhq.com/posting-api/job-board/{token}"""

from __future__ import annotations

import httpx

from .base import FetchError, RawJob, SourceSpec, get_json, html_to_text, parse_iso

API = "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"


class AshbyConnector:
    name = "ashby"

    def fetch(self, spec: SourceSpec, client: httpx.Client) -> list[RawJob]:
        data = get_json(client, API.format(token=spec.token))
        if not isinstance(data, dict) or "jobs" not in data:
            raise FetchError("Ashby response missing 'jobs'")
        jobs: list[RawJob] = []
        for item in data["jobs"]:
            if item.get("isListed") is False:
                continue
            location = item.get("location")
            if item.get("isRemote") and location and "remote" not in location.lower():
                location = f"{location} (Remote)"
            jobs.append(
                RawJob(
                    external_id=str(item["id"]),
                    title=(item.get("title") or "").strip(),
                    url=item.get("jobUrl") or item.get("applyUrl") or "",
                    location=location or None,
                    department=item.get("department") or item.get("team") or None,
                    description=item.get("descriptionPlain") or html_to_text(item.get("descriptionHtml")),
                    posted_at=parse_iso(item.get("publishedAt")),
                    extra={
                        "workplace_type": item.get("workplaceType"),
                        "compensation": (item.get("compensation") or {}).get("compensationTierSummary"),
                    },
                )
            )
        return jobs

    def fetch_detail(self, spec: SourceSpec, job: RawJob, client: httpx.Client) -> RawJob:
        return job
