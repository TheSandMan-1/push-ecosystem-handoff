"""Jinja filters for friendly dates, money and labels."""

from __future__ import annotations

from datetime import datetime

from ..models import utcnow
from ..taxonomy import DISCIPLINES, ROLE_FAMILIES, SENIORITIES


def timeago(value: datetime | None) -> str:
    if not value:
        return "unknown"
    seconds = (utcnow() - value).total_seconds()
    if seconds < 0:
        seconds = 0
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}h ago"
    days = hours / 24
    if days < 30:
        return f"{int(days)}d ago"
    months = days / 30
    if months < 12:
        return f"{int(months)}mo ago"
    return f"{int(days / 365)}y ago"


def money(value: int | float | None) -> str:
    if value is None:
        return ""
    if value >= 1000:
        return f"${value / 1000:.0f}k"
    return f"${value:,.0f}"


def usd(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if 0 < value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.{digits}f}"


def number(value: int | float | None) -> str:
    if value is None:
        return "0"
    return f"{value:,}"


def salary_range(lo: int | None, hi: int | None) -> str:
    if lo and hi:
        return f"{money(lo)}–{money(hi)}" if lo != hi else money(lo)
    return money(lo or hi)


def hue(value: str | None) -> int:
    import hashlib

    digest = hashlib.md5((value or "").encode()).digest()
    return int.from_bytes(digest[:2], "big") % 360


def initials(value: str | None) -> str:
    words = [w for w in (value or "?").replace("&", " ").split() if w[:1].isalnum()]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def register_filters(env) -> None:
    env.filters["hue"] = hue
    env.filters["initials"] = initials
    env.filters["timeago"] = timeago
    env.filters["money"] = money
    env.filters["usd"] = usd
    env.filters["number"] = number
    env.globals["salary_range"] = salary_range
    env.globals["SENIORITIES"] = SENIORITIES
    env.globals["ROLE_FAMILIES"] = ROLE_FAMILIES
    env.globals["DISCIPLINES"] = DISCIPLINES
