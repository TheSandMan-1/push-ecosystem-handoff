"""SmartRecruiters public postings API: api.smartrecruiters.com/v1/companies/{token}/postings"""

from __future__ import annotations

import httpx

from .base import FetchError, RawJob, SourceSpec, get_json, html_to_text, parse_iso

LIST_API = "https://api.smartrecruiters.com/v1/companies/{token}/postings"
DETAIL_API = "https://api.smartrecruiters.com/v1/companies/{token}/postings/{id}"
PUBLIC_URL = "https://jobs.smartrecruiters.com/{token}/{id}"
PAGE = 100
MAX_JOBS = 5000


class SmartRecruitersConnector:
    name = "smartrecruiters"

    def fetch(self, spec: SourceSpec, client: httpx.Client) -> list[RawJob]:
        jobs: list[RawJob] = []
        offset = 0
        while offset < MAX_JOBS:
            data = get_json(
                client, LIST_API.format(token=spec.token), params={"limit": PAGE, "offset": offset}
            )
            if not isinstance(data, dict) or "content" not in data:
                raise FetchError("SmartRecruiters response missing 'content'")
            page = data["content"] or []
            for item in page:
                loc = item.get("location") or {}
                parts = [loc.get("city"), loc.get("region"), loc.get("country")]
                location = ", ".join(p for p in parts if p) or None
                if loc.get("remote"):
                    location = f"{location} (Remote)" if location else "Remote"
                jobs.append(
                    RawJob(
                        external_id=str(item["id"]),
                        title=(item.get("name") or "").strip(),
                        url=PUBLIC_URL.format(token=spec.token, id=item["id"]),
                        location=location,
                        department=(item.get("department") or {}).get("label"),
                        posted_at=parse_iso(item.get("releasedDate")),
                        detail_ref=str(item["id"]),
                    )
                )
            offset += len(page)
            if not page or offset >= int(data.get("totalFound") or 0):
                break
        return jobs

    def fetch_detail(self, spec: SourceSpec, job: RawJob, client: httpx.Client) -> RawJob:
        data = get_json(client, DETAIL_API.format(token=spec.token, id=job.detail_ref or job.external_id))
        sections = ((data.get("jobAd") or {}).get("sections")) or {}
        chunks = []
        for key in ("jobDescription", "qualifications", "additionalInformation"):
            sec = sections.get(key) or {}
            text = html_to_text(sec.get("text"))
            if text:
                title = sec.get("title") or ""
                chunks.append(f"{title}\n{text}".strip())
        job.description = "\n\n".join(chunks) or job.description
        return job
