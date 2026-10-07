"""Shared types and helpers for applicant-tracking-system (ATS) connectors."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Protocol

import httpx


class FetchError(RuntimeError):
    """A source could not be fetched. Jobs from it must NOT be marked closed."""


@dataclass
class RawJob:
    external_id: str
    title: str
    url: str
    location: str | None = None
    department: str | None = None
    description: str | None = None
    posted_at: datetime | None = None
    # Connectors whose list endpoint has no description set this so ingest can
    # fetch details lazily, only for new jobs worth reading.
    detail_ref: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        basis = "\x1f".join([self.title or "", self.location or "", self.description or ""])
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()


@dataclass
class SourceSpec:
    """The minimum needed to fetch a board. Mirrors models.Source."""

    ats: str
    token: str
    company_name: str
    workday_host: str | None = None
    workday_site: str | None = None


class Connector(Protocol):
    name: str

    def fetch(self, spec: SourceSpec, client: httpx.Client) -> list[RawJob]: ...

    def fetch_detail(self, spec: SourceSpec, job: RawJob, client: httpx.Client) -> RawJob: ...


_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "table", "section", "article", "header", "footer",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(value: str | None) -> str | None:
    """Convert (possibly entity-escaped) HTML into readable plain text."""
    if not value:
        return None
    text = value
    # Greenhouse double-escapes: "&lt;p&gt;" -> "<p>"
    if "&lt;" in text and "<" not in text:
        text = html.unescape(text)
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    out = "".join(parser.parts).replace("\xa0", " ")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() or None


def parse_iso(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        # epoch milliseconds (Lever) or seconds
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


_POSTED_RE = re.compile(r"posted\s+(\d+)\+?\s+days?\s+ago", re.I)


def parse_relative_posted(value: str | None, now: datetime | None = None) -> datetime | None:
    """Workday-style 'Posted 3 Days Ago' / 'Posted Today' / 'Posted 30+ Days Ago'."""
    if not value:
        return None
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    low = value.strip().lower()
    if "today" in low:
        return now
    if "yesterday" in low:
        return now - timedelta(days=1)
    m = _POSTED_RE.search(low)
    if m:
        return now - timedelta(days=int(m.group(1)))
    return None


def get_json(client: httpx.Client, url: str, **kwargs) -> Any:
    try:
        resp = client.get(url, **kwargs)
    except httpx.HTTPError as exc:
        raise FetchError(f"GET {url} failed: {exc}") from exc
    if resp.status_code != 200:
        raise FetchError(f"GET {url} returned HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise FetchError(f"GET {url} returned non-JSON") from exc


def post_json(client: httpx.Client, url: str, payload: dict, **kwargs) -> Any:
    try:
        resp = client.post(url, json=payload, **kwargs)
    except httpx.HTTPError as exc:
        raise FetchError(f"POST {url} failed: {exc}") from exc
    if resp.status_code != 200:
        raise FetchError(f"POST {url} returned HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise FetchError(f"POST {url} returned non-JSON") from exc
