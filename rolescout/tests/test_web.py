import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rolescout.db import session_scope
from rolescout.demo import DEMO_EMAIL, DEMO_PASSWORD, load_demo
from rolescout.extract.pipeline import Extractor
from rolescout.config import get_settings
from rolescout.models import DigestSend, Profile, User, UserJob


def csrf_of(html: str) -> str:
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


@pytest.fixture
def app_client():
    from rolescout.web.app import create_app

    with session_scope() as s:
        load_demo(s)
    with session_scope() as s:
        Extractor(get_settings(), use_defaults=False).run(s)
    with TestClient(create_app()) as c:
        yield c


def login(c, email=DEMO_EMAIL, password=DEMO_PASSWORD):
    token = csrf_of(c.get("/login").text)
    r = c.post("/login", data={"email": email, "password": password, "csrf_token": token}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return csrf_of(c.get("/app").text)


def test_landing_and_health(app_client):
    assert app_client.get("/healthz").json()["ok"]
    r = app_client.get("/")
    assert r.status_code == 200 and "at your level" in r.text


def test_app_requires_login(app_client):
    r = app_client.get("/app", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")


def test_post_without_csrf_is_rejected(app_client):
    r = app_client.post("/login", data={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    assert r.status_code == 403


def test_bad_password(app_client):
    token = csrf_of(app_client.get("/login").text)
    r = app_client.post("/login", data={"email": DEMO_EMAIL, "password": "wrong-password", "csrf_token": token})
    assert r.status_code == 400 and "Wrong email or password" in r.text


def test_feed_shows_entry_roles_and_hides_senior(app_client):
    login(app_client)
    html = app_client.get("/app").text
    assert "R&amp;D Engineer I" in html or "R&D Engineer I" in html
    assert "Senior Mechanical Engineer" not in html
    assert "Maintenance Engineer" not in html
    filtered = app_client.get("/app?tab=filtered&reason=too_senior").text
    assert "Senior Mechanical Engineer" in filtered


def test_status_api_save_hide_and_tabs(app_client):
    token = login(app_client)
    api = app_client.get("/api/feed").json()
    first, second = api["matches"][0]["id"], api["matches"][1]["id"]
    h = {"X-CSRF-Token": token}
    assert app_client.post(f"/app/jobs/{first}/status", json={"status": "applied"}, headers=h).json()["ok"]
    assert app_client.post(f"/app/jobs/{second}/status", json={"status": "hidden", "reason": "too_senior"}, headers=h).json()["ok"]
    assert app_client.post(f"/app/jobs/{second}/status", json={"status": "bogus"}, headers=h).status_code == 400
    assert app_client.post(f"/app/jobs/{first}/status", json={"status": "saved"}, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    ids = [m["id"] for m in app_client.get("/api/feed").json()["matches"]]
    assert second not in ids
    with session_scope() as s:
        user = s.scalar(select(User).where(User.email == DEMO_EMAIL))
        rows = {r.job_id: r for r in s.scalars(select(UserJob).where(UserJob.user_id == user.id))}
        assert rows[first].status == "applied" and rows[second].hide_reason == "too_senior"
    assert "data-job=\"%d\"" % first in app_client.get("/app?tab=applied").text


def test_profile_update_changes_matches(app_client):
    token = login(app_client)
    before = len(app_client.get("/api/feed").json()["matches"])
    r = app_client.post("/app/profile", data={
        "csrf_token": token, "target_seniorities": ["entry", "mid"], "max_years": "4", "disciplines": ["mechanical"],
        "locations": "San Diego, Irvine, Carlsbad, Lake Forest, Oceanside, Long Beach", "remote_ok": "on",
        "exclude_clearance": "on", "require_engineering": "on", "digest_frequency": "daily", "salary_floor": "$70,000",
    }, follow_redirects=False)
    assert r.status_code == 303
    after = len(app_client.get("/api/feed").json()["matches"])
    assert after > before
    with session_scope() as s:
        p = s.scalar(select(Profile))
        assert p.max_years == 4 and p.salary_floor == 70000 and p.digest_frequency == "daily"


def test_signup_creates_account_and_non_admin_cannot_see_admin(app_client):
    token = csrf_of(app_client.get("/signup").text)
    r = app_client.post("/signup", data={"email": "new@example.com", "password": "longenough1", "csrf_token": token},
                        follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/app/profile")
    assert app_client.get("/admin").status_code == 403
    token = csrf_of(app_client.get("/signup").text)
    dup = app_client.post("/signup", data={"email": "new@example.com", "password": "longenough1", "csrf_token": token})
    assert dup.status_code in (303, 400)


def test_signup_validation(app_client):
    token = csrf_of(app_client.get("/signup").text)
    r = app_client.post("/signup", data={"email": "nope", "password": "short", "csrf_token": token})
    assert r.status_code == 400 and "valid email" in r.text


def test_open_redirect_blocked(app_client):
    token = csrf_of(app_client.get("/login").text)
    r = app_client.post("/login", data={"email": DEMO_EMAIL, "password": DEMO_PASSWORD, "csrf_token": token,
                                         "next": "//evil.example.com"}, follow_redirects=False)
    assert r.headers["location"] == "/app"


def test_admin_pages_and_add_source(app_client):
    token = login(app_client)
    assert "Operations" in app_client.get("/admin").text
    assert app_client.get("/admin/review").status_code == 200
    r = app_client.post("/admin/sources", data={"csrf_token": token, "url": "https://jobs.lever.co/acme",
                                                "company_name": "Acme"}, follow_redirects=True)
    assert "Added Acme" in r.text
    r = app_client.post("/admin/sources", data={"csrf_token": token, "url": "https://example.com/jobs",
                                                "company_name": "Nope"}, follow_redirects=True)
    assert "Can&#39;t detect" in r.text or "Can't detect" in r.text


def test_job_detail_and_digest_preview(app_client):
    login(app_client)
    jid = app_client.get("/api/feed").json()["matches"][0]["id"]
    detail = app_client.get(f"/app/jobs/{jid}")
    assert detail.status_code == 200 and "What the posting actually asks for" in detail.text
    assert app_client.get("/app/jobs/999999").status_code == 404
    raw = app_client.get("/app/digest/raw").text
    assert "Unsubscribe" in raw and "View &amp; apply" in raw


def test_digest_run_writes_outbox_and_does_not_repeat(app_client):
    from rolescout.digest import MAX_ITEMS, run_digests

    with session_scope() as s:
        results = [run_digests(s, force=True) for _ in range(4)]
        sends = s.scalars(select(DigestSend).order_by(DigestSend.id)).all()
    assert results[0]["sent"] == 1 and sends[0].job_count == MAX_ITEMS
    assert results[-1]["sent"] == 0 and results[-1]["skipped_empty"] == 1  # backlog drained
    all_ids = [jid for send in sends for jid in send.job_ids]
    assert len(all_ids) == len(set(all_ids))  # no job is ever emailed twice
    assert Path(sends[0].detail).exists()


def test_unsubscribe(app_client):
    with session_scope() as s:
        token = s.scalar(select(User)).unsubscribe_token
    assert "unsubscribed" in app_client.get(f"/unsubscribe/{token}").text
    with session_scope() as s:
        assert s.scalar(select(Profile)).digest_frequency == "off"
