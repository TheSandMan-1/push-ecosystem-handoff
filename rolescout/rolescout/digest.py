"""Email digests: the cheapest way to test demand before building more product.

Each digest lists new matches the user has not been sent before. With no SMTP
configured, digests are written to OUTBOX_DIR as .html files you can open.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from datetime import timedelta
from email.message import EmailMessage
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .matching import Match, build_feed
from .models import DigestSend, Run, User, utcnow
from .web.filters import register_filters

log = logging.getLogger(__name__)
MAX_ITEMS = 15
FREQUENCY_DAYS = {"daily": 1, "weekly": 7}

_env = Environment(loader=PackageLoader("rolescout", "web/templates"), autoescape=select_autoescape(["html"]))
register_filters(_env)


@dataclass
class Digest:
    user: User
    subject: str
    html: str
    text: str
    matches: list[Match]
    hidden_total: int


def _already_sent(session: Session, user_id: int) -> set[int]:
    sent: set[int] = set()
    for row in session.scalars(select(DigestSend).where(DigestSend.user_id == user_id)):
        sent.update(row.job_ids or [])
    return sent


def build_digest(session: Session, user: User, settings: Settings | None = None, *, preview: bool = False) -> Digest:
    settings = settings or get_settings()
    feed = build_feed(session, user.id, user.profile)
    sent = set() if preview else _already_sent(session, user.id)
    fresh = [m for m in feed.matches
             if m.facts is not None and m.job.id not in sent and m.user_status is None][:MAX_ITEMS]
    count = len(fresh)
    subject = (
        f"{count} new {'role' if count == 1 else 'roles'} that fit you" if count else "No new matches this time"
    )
    ctx = {
        "user": user,
        "matches": fresh,
        "hidden_total": feed.hidden_total,
        "exclusion_counts": feed.exclusion_counts.most_common(4),
        "base_url": settings.base_url,
        "subject": subject,
    }
    html = _env.get_template("email/digest.html").render(**ctx)
    lines = [subject, ""]
    for m in fresh:
        lines.append(f"- {m.job.title} at {m.job.company_name} ({m.job.location or 'location n/a'})")
        if m.reasons:
            lines.append(f"  Why: {', '.join(m.reasons[:4])}")
        lines.append(f"  {m.job.url}")
    lines += ["", f"We filtered out {feed.hidden_total} postings that didn't fit.",
              f"Manage your settings: {settings.base_url}/app/profile",
              f"Unsubscribe: {settings.base_url}/unsubscribe/{user.unsubscribe_token}"]
    return Digest(user=user, subject=subject, html=html, text="\n".join(lines), matches=fresh,
                  hidden_total=feed.hidden_total)


def deliver(digest: Digest, settings: Settings) -> tuple[str, str]:
    if settings.smtp_host:
        msg = EmailMessage()
        msg["Subject"] = digest.subject
        msg["From"] = settings.smtp_from
        msg["To"] = digest.user.email
        msg["List-Unsubscribe"] = f"<{settings.base_url}/unsubscribe/{digest.user.unsubscribe_token}>"
        msg.set_content(digest.text)
        msg.add_alternative(digest.html, subtype="html")
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            if settings.smtp_user:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
        return "smtp", digest.user.email
    outbox = Path(settings.outbox_dir)
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%d-%H%M%S")
    path = outbox / f"{stamp}-user{digest.user.id}.html"
    path.write_text(digest.html)
    return "outbox", str(path)


def due(session: Session, user: User) -> bool:
    days = FREQUENCY_DAYS.get(user.profile.digest_frequency if user.profile else "off")
    if not days:
        return False
    last = session.scalar(
        select(DigestSend.sent_at).where(DigestSend.user_id == user.id).order_by(DigestSend.sent_at.desc()).limit(1)
    )
    return last is None or utcnow() - last >= timedelta(days=days) - timedelta(hours=2)


def run_digests(session: Session, settings: Settings | None = None, *, force: bool = False,
                user_ids: list[int] | None = None) -> dict:
    settings = settings or get_settings()
    run = Run(kind="digest")
    session.add(run)
    session.commit()
    stats = {"considered": 0, "sent": 0, "skipped_empty": 0, "errors": []}
    stmt = select(User)
    if user_ids:
        stmt = stmt.where(User.id.in_(user_ids))
    for user in session.scalars(stmt).all():
        if user.profile is None:
            continue
        stats["considered"] += 1
        if not force and not due(session, user):
            continue
        try:
            digest = build_digest(session, user, settings)
            if not digest.matches:
                stats["skipped_empty"] += 1
                continue
            transport, detail = deliver(digest, settings)
            session.add(DigestSend(user_id=user.id, job_count=len(digest.matches), transport=transport,
                                   detail=detail, job_ids=[m.job.id for m in digest.matches]))
            session.commit()
            stats["sent"] += 1
        except Exception as exc:  # one bad address must not stop the batch
            session.rollback()
            log.exception("digest failed for user %s", user.id)
            stats["errors"].append(f"user {user.id}: {exc}")
    run.finished_at = utcnow()
    run.stats = stats
    run.status = "ok" if not stats["errors"] else "partial"
    session.commit()
    return stats
