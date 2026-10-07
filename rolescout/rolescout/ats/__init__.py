"""ATS connector registry and careers-URL detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from .ashby import AshbyConnector
from .base import Connector, FetchError, RawJob, SourceSpec, html_to_text
from .greenhouse import GreenhouseConnector
from .lever import LeverConnector
from .smartrecruiters import SmartRecruitersConnector
from .workday import WorkdayConnector

CONNECTORS: dict[str, Connector] = {
    c.name: c
    for c in (
        GreenhouseConnector(),
        LeverConnector(),
        AshbyConnector(),
        WorkdayConnector(),
        SmartRecruitersConnector(),
    )
}

# Connectors whose list endpoint lacks descriptions.
LAZY_DETAIL = {"workday", "smartrecruiters"}


def get_connector(ats: str) -> Connector:
    try:
        return CONNECTORS[ats]
    except KeyError:
        raise ValueError(f"Unsupported ATS '{ats}'. Supported: {', '.join(sorted(CONNECTORS))}")


@dataclass
class DetectedSource:
    ats: str
    token: str
    workday_host: str | None = None
    workday_site: str | None = None

    @property
    def slug(self) -> str:
        if self.ats == "workday":
            return f"workday:{self.workday_host}/{self.workday_site}".lower()
        return f"{self.ats}:{self.token}".lower()


_WORKDAY_HOST = re.compile(r"^[a-z0-9-]+\.wd\d+\.myworkdayjobs\.com$", re.I)
_LOCALE = re.compile(r"^[a-z]{2}-[A-Z]{2}$")


def detect_source(url: str) -> DetectedSource:
    """Turn a pasted careers URL into a connector + board token.

    Raises ValueError with a helpful message when the ATS is not supported.
    """
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]

    if host in {"boards.greenhouse.io", "job-boards.greenhouse.io", "boards.eu.greenhouse.io", "job-boards.eu.greenhouse.io"}:
        qs = parse_qs(parsed.query)
        if "for" in qs:
            return DetectedSource("greenhouse", qs["for"][0])
        if parts and parts[0] != "embed":
            return DetectedSource("greenhouse", parts[0])
    if host == "boards-api.greenhouse.io" and len(parts) >= 3 and parts[1] == "boards":
        return DetectedSource("greenhouse", parts[2])
    if host in {"jobs.lever.co", "jobs.eu.lever.co"} and parts:
        return DetectedSource("lever", parts[0])
    if host == "api.lever.co" and len(parts) >= 3:
        return DetectedSource("lever", parts[2])
    if host == "jobs.ashbyhq.com" and parts:
        return DetectedSource("ashby", parts[0])
    if host in {"careers.smartrecruiters.com", "jobs.smartrecruiters.com"} and parts:
        return DetectedSource("smartrecruiters", parts[0])
    if _WORKDAY_HOST.match(host):
        site_parts = [p for p in parts if not _LOCALE.match(p)]
        if site_parts and site_parts[0] not in {"wday", "job"}:
            return DetectedSource("workday", host.split(".")[0], workday_host=host, workday_site=site_parts[0])
        raise ValueError(
            "That Workday URL has no site name. Open the careers page and copy the URL "
            "that looks like https://company.wd1.myworkdayjobs.com/SiteName"
        )
    raise ValueError(
        f"Can't detect the job system for '{url}'. Supported: Greenhouse, Lever, Ashby, "
        "Workday (*.myworkdayjobs.com), SmartRecruiters. iCIMS, Taleo and SuccessFactors are not supported yet."
    )


__all__ = [
    "CONNECTORS",
    "LAZY_DETAIL",
    "Connector",
    "DetectedSource",
    "FetchError",
    "RawJob",
    "SourceSpec",
    "detect_source",
    "get_connector",
    "html_to_text",
]
