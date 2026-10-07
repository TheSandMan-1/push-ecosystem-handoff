"""Lever public postings API: api.lever.co/v0/postings/{token}?mode=json"""

from __future__ import annotations

import httpx

from .base import FetchError, RawJob, SourceSpec, get_json, html_to_text, parse_iso

API = "https://api.lever.co/v0/postings/{token}?mode=json"


def _description(item: dict) -> str | None:
    parts: list[str] = []
    if item.get("descriptionPlain"):
        parts.append(item["descriptionPlain"].strip())
    for block in item.get("lists") or []:
        heading = (block.get("text") or "").strip()
        body = html_to_text(block.get("content")) or ""
        parts.append(f"{heading}\n{body}".strip())
    if item.get("additionalPlain"):
        parts.append(item["additionalPlain"].strip())
    text = "\n\n".join(p for p in parts if p)
    return text or None


class LeverConnector:
    name = "lever"

    def fetch(self, spec: SourceSpec, client: httpx.Client) -> list[RawJob]:
        data = get_json(client, API.format(token=spec.token))
        if not isinstance(data, list):
            raise FetchError("Lever response was not a list")
        jobs: list[RawJob] = []
        for item in data:
            cats = item.get("categories") or {}
            salary = item.get("salaryRange") or None
            jobs.append(
                RawJob(
                    external_id=str(item["id"]),
                    title=(item.get("text") or "").strip(),
                    url=item.get("hostedUrl") or item.get("applyUrl") or "",
                    location=cats.get("location") or None,
                    department=cats.get("team") or cats.get("department") or None,
                    description=_description(item),
                    posted_at=parse_iso(item.get("createdAt")),
                    extra={
                        "workplace_type": item.get("workplaceType"),
                        "commitment": cats.get("commitment"),
                        "salary": salary,
                    },
                )
            )
        return jobs

    def fetch_detail(self, spec: SourceSpec, job: RawJob, client: httpx.Client) -> RawJob:
        return job
