"""Greenhouse public job board API: boards-api.greenhouse.io/v1/boards/{token}/jobs"""

from __future__ import annotations

import httpx

from .base import FetchError, RawJob, SourceSpec, get_json, html_to_text, parse_iso

API = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"


class GreenhouseConnector:
    name = "greenhouse"

    def fetch(self, spec: SourceSpec, client: httpx.Client) -> list[RawJob]:
        data = get_json(client, API.format(token=spec.token))
        if not isinstance(data, dict) or "jobs" not in data:
            raise FetchError("Greenhouse response missing 'jobs'")
        jobs: list[RawJob] = []
        for item in data["jobs"]:
            departments = item.get("departments") or []
            jobs.append(
                RawJob(
                    external_id=str(item["id"]),
                    title=(item.get("title") or "").strip(),
                    url=item.get("absolute_url") or "",
                    location=((item.get("location") or {}).get("name") or None),
                    department=departments[0].get("name") if departments else None,
                    description=html_to_text(item.get("content")),
                    posted_at=parse_iso(item.get("first_published") or item.get("updated_at")),
                )
            )
        return jobs

    def fetch_detail(self, spec: SourceSpec, job: RawJob, client: httpx.Client) -> RawJob:
        return job
