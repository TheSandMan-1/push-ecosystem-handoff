"""FastAPI web app: landing page, auth, the matches feed, profile, admin."""

from __future__ import annotations

import logging
import secrets
from collections import Counter
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, joinedload
from starlette.middleware.sessions import SessionMiddleware

from .. import __version__
from ..auth import AuthError, authenticate, create_user
from ..config import get_settings
from ..db import init_db, new_session
from ..digest import build_digest
from ..extract.pipeline import ExtractStats, Extractor
from ..ingest import add_source_from_url
from ..jobs import runner
from ..matching import EXCLUSION_LABELS, build_feed, evaluate, user_states
from ..models import DigestSend, Job, JobFacts, LLMCall, Profile, Run, Source, User, UserJob, utcnow
from ..taxonomy import DISCIPLINES, HIDE_REASONS, ROLE_FAMILIES, SENIORITIES
from .filters import register_filters

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
register_filters(templates.env)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


# --------------------------------------------------------------------------- plumbing

def get_db():
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


async def check_csrf(request: Request) -> None:
    if request.method in SAFE_METHODS:
        return
    expected = request.session.get("csrf")
    sent = request.headers.get("x-csrf-token")
    if not sent:
        form = await request.form()
        sent = form.get("csrf_token")
    if not expected or not sent or not secrets.compare_digest(str(sent), str(expected)):
        raise HTTPException(status_code=403, detail="Your session expired. Reload the page and try again.")


def current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    uid = request.session.get("uid")
    if not uid:
        return None
    user = db.get(User, uid)
    if user is None:
        request.session.pop("uid", None)
    return user


class LoginRequired(Exception):
    pass


def require_user(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise LoginRequired()
    if user.profile is None:
        user.profile = Profile()
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admins only.")
    return user


def flash(request: Request, message: str, kind: str = "info") -> None:
    request.session.setdefault("flash", []).append({"message": message, "kind": kind})


def render(request: Request, name: str, ctx: dict[str, Any] | None = None, status_code: int = 200) -> HTMLResponse:
    ctx = dict(ctx or {})
    ctx.setdefault("user", None)
    ctx["request"] = request
    ctx["csrf"] = csrf_token(request)
    ctx["flashes"] = request.session.pop("flash", [])
    ctx["version"] = __version__
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _list_field(form, name: str, allowed: dict | list | None = None) -> list[str]:
    values = [v.strip() for v in form.getlist(name) if v and v.strip()]
    if allowed is not None:
        values = [v for v in values if v in allowed]
    return values


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").replace("\n", ",").split(",") if v.strip()][:30]


# --------------------------------------------------------------------------- app

def create_app() -> FastAPI:
    settings = get_settings()
    init_db()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runner.start_scheduler(settings.schedule_minutes)
        yield
        runner.stop()

    app = FastAPI(title="RoleScout", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie="rs_session",
                       max_age=60 * 60 * 24 * 30, same_site="lax",
                       https_only=settings.base_url.startswith("https"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired):
        if request.url.path.startswith("/api/") or request.headers.get("x-csrf-token"):
            return JSONResponse({"error": "login required"}, status_code=401)
        return redirect(f"/login?next={request.url.path}")

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/") or request.headers.get("x-csrf-token"):
            return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
        return render(request, "error.html", {"status": exc.status_code, "detail": exc.detail},
                      status_code=exc.status_code)

    # ------------------------------------------------------------------ public

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/", response_class=HTMLResponse)
    def landing(request: Request, user: User | None = Depends(current_user), db: Session = Depends(get_db)):
        if user:
            return redirect("/app")
        week_ago = utcnow() - timedelta(days=7)
        open_jobs = db.scalar(select(func.count(Job.id)).where(Job.closed_at.is_(None))) or 0
        eng = db.scalar(select(func.count(JobFacts.job_id)).join(Job).where(
            Job.closed_at.is_(None), JobFacts.is_engineering.is_(True))) or 0
        entry = db.scalar(select(func.count(JobFacts.job_id)).join(Job).where(
            Job.closed_at.is_(None), JobFacts.is_engineering.is_(True),
            JobFacts.seniority.in_(["entry", "intern"]))) or 0
        closed_week = db.scalar(select(func.count(Job.id)).where(Job.closed_at >= week_ago)) or 0
        companies = db.scalar(select(func.count(func.distinct(Job.company_name))).where(Job.closed_at.is_(None))) or 0
        return render(request, "landing.html", {
            "stats": {"open_jobs": open_jobs, "engineering": eng, "entry": entry,
                      "closed_week": closed_week, "companies": companies},
            "signups_open": settings.signups_open,
        })

    @app.get("/signup", response_class=HTMLResponse)
    def signup_form(request: Request, user: User | None = Depends(current_user)):
        if user:
            return redirect("/app")
        return render(request, "auth/signup.html", {"signups_open": settings.signups_open})

    @app.post("/signup", dependencies=[Depends(check_csrf)])
    def signup(request: Request, email: str = Form(""), password: str = Form(""), name: str = Form(""),
               db: Session = Depends(get_db)):
        if not settings.signups_open:
            flash(request, "Signups are closed right now.", "error")
            return redirect("/signup")
        try:
            user = create_user(db, email, password, name)
        except AuthError as exc:
            return render(request, "auth/signup.html", {"error": str(exc), "email": email, "name": name,
                                                        "signups_open": True}, status_code=400)
        db.commit()
        request.session.clear()
        request.session["uid"] = user.id
        flash(request, "Account created. Tell us what you're looking for and we'll do the filtering.", "success")
        return redirect("/app/profile?welcome=1")

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/app", user: User | None = Depends(current_user)):
        if user:
            return redirect("/app")
        return render(request, "auth/login.html", {"next": next})

    @app.post("/login", dependencies=[Depends(check_csrf)])
    def login(request: Request, email: str = Form(""), password: str = Form(""), next: str = Form("/app"),
              db: Session = Depends(get_db)):
        try:
            user = authenticate(db, email, password)
        except AuthError as exc:
            return render(request, "auth/login.html", {"error": str(exc), "email": email, "next": next},
                          status_code=400)
        request.session.clear()
        request.session["uid"] = user.id
        target = next if next.startswith("/") and not next.startswith("//") else "/app"
        return redirect(target)

    @app.post("/logout", dependencies=[Depends(check_csrf)])
    def logout(request: Request):
        request.session.clear()
        return redirect("/")

    @app.get("/unsubscribe/{token}", response_class=HTMLResponse)
    def unsubscribe(request: Request, token: str, db: Session = Depends(get_db)):
        user = db.scalar(select(User).where(User.unsubscribe_token == token))
        if user and user.profile:
            user.profile.digest_frequency = "off"
        return render(request, "unsubscribed.html", {"ok": user is not None})

    # ------------------------------------------------------------------ app

    @app.get("/app", response_class=HTMLResponse)
    def feed(request: Request, tab: str = "matches", q: str = "", user: User = Depends(require_user),
             db: Session = Depends(get_db)):
        profile = user.profile
        result = build_feed(db, user.id, profile)
        states = user_states(db, user.id)
        query = q.strip().lower()

        def matches_query(m) -> bool:
            if not query:
                return True
            blob = f"{m.job.title} {m.job.company_name} {m.job.location or ''}".lower()
            return all(part in blob for part in query.split())

        if tab in ("saved", "applied", "hidden"):
            ids = [jid for jid, st in states.items() if st.status == tab]
            jobs = db.scalars(select(Job).options(joinedload(Job.facts)).where(Job.id.in_(ids))).unique().all() if ids else []
            items = [evaluate(j, j.facts, profile, user_status=None) for j in jobs]
            for it in items:
                it.user_status = states[it.job.id].status
            items.sort(key=lambda m: states[m.job.id].updated_at, reverse=True)
        elif tab == "filtered":
            items = [m for m in result.excluded if m.excluded != "hidden"]
            reason = request.query_params.get("reason")
            if reason:
                items = [m for m in items if m.excluded == reason]
            items.sort(key=lambda m: m.job.first_seen_at, reverse=True)
            items = items[:300]
        else:
            tab = "matches"
            items = result.matches
        items = [m for m in items if matches_query(m)]
        counts = Counter(st.status for st in states.values())
        last_ingest = db.scalar(select(Run.finished_at).where(Run.kind == "ingest", Run.status != "error")
                                .order_by(Run.id.desc()).limit(1))
        return render(request, "app/feed.html", {
            "user": user, "tab": tab, "q": q, "items": items,
            "match_count": len(result.matches), "hidden_total": result.hidden_total,
            "exclusions": result.exclusion_counts.most_common(), "exclusion_labels": EXCLUSION_LABELS,
            "status_counts": counts, "hide_reasons": HIDE_REASONS, "last_ingest": last_ingest,
            "profile_incomplete": not (profile.disciplines or profile.locations),
            "active_reason": request.query_params.get("reason"),
        })

    @app.get("/app/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
        job = db.scalar(select(Job).options(joinedload(Job.facts), joinedload(Job.source)).where(Job.id == job_id))
        if job is None:
            raise HTTPException(404, "That job doesn't exist.")
        st = db.get(UserJob, (user.id, job.id))
        m = evaluate(job, job.facts, user.profile, user_status=st.status if st else None)
        return render(request, "app/job.html", {
            "user": user, "m": m, "job": job, "facts": job.facts, "hide_reasons": HIDE_REASONS,
            "seniorities": SENIORITIES, "families": ROLE_FAMILIES, "disciplines": DISCIPLINES,
        })

    @app.post("/app/jobs/{job_id}/status", dependencies=[Depends(check_csrf)])
    async def set_status(request: Request, job_id: int, user: User = Depends(require_user),
                         db: Session = Depends(get_db)):
        if request.headers.get("content-type", "").startswith("application/json"):
            payload = await request.json()
        else:
            payload = dict(await request.form())
        status = str(payload.get("status") or "")
        reason = payload.get("reason") or None
        if status not in {"saved", "applied", "hidden", "clear"}:
            raise HTTPException(400, "Unknown status.")
        if reason is not None and reason not in HIDE_REASONS:
            reason = "not_interested"
        if db.get(Job, job_id) is None:
            raise HTTPException(404, "That job doesn't exist.")
        row = db.get(UserJob, (user.id, job_id))
        if status == "clear":
            if row:
                db.delete(row)
        else:
            if row is None:
                row = UserJob(user_id=user.id, job_id=job_id, status=status)
                db.add(row)
            row.status = status
            row.hide_reason = reason if status == "hidden" else None
            row.updated_at = utcnow()
        db.commit()
        if request.headers.get("x-csrf-token"):
            return JSONResponse({"ok": True, "status": status})
        back = request.headers.get("referer") or "/app"
        return redirect(back)

    @app.get("/app/profile", response_class=HTMLResponse)
    def profile_form(request: Request, welcome: int = 0, user: User = Depends(require_user)):
        return render(request, "app/profile.html", {
            "user": user, "p": user.profile, "welcome": welcome,
            "seniorities": {k: v for k, v in SENIORITIES.items() if k != "unknown"},
            "families": {k: v for k, v in ROLE_FAMILIES.items() if k != "non_engineering"},
            "disciplines": DISCIPLINES,
        })

    @app.post("/app/profile", dependencies=[Depends(check_csrf)])
    async def profile_save(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
        form = await request.form()
        p = db.get(Profile, user.id) or Profile(user_id=user.id)
        p.target_seniorities = _list_field(form, "target_seniorities", SENIORITIES) or ["entry"]
        try:
            p.max_years = max(0, min(30, int(form.get("max_years") or 2)))
        except ValueError:
            p.max_years = 2
        p.disciplines = _list_field(form, "disciplines", DISCIPLINES)
        p.role_families = _list_field(form, "role_families", ROLE_FAMILIES)
        p.excluded_role_families = _list_field(form, "excluded_role_families", ROLE_FAMILIES)
        p.locations = _csv(form.get("locations"))
        p.remote_ok = form.get("remote_ok") == "on"
        p.exclude_clearance = form.get("exclude_clearance") == "on"
        p.require_engineering = form.get("require_engineering") == "on"
        floor = (form.get("salary_floor") or "").replace(",", "").replace("$", "").strip()
        p.salary_floor = int(floor) if floor.isdigit() else None
        p.include_keywords = _csv(form.get("include_keywords"))
        p.exclude_keywords = _csv(form.get("exclude_keywords"))
        p.muted_companies = _csv(form.get("muted_companies"))
        freq = form.get("digest_frequency") or "weekly"
        p.digest_frequency = freq if freq in {"daily", "weekly", "off"} else "weekly"
        name = (form.get("name") or "").strip()
        user.name = name or user.name
        db.add(p)
        db.commit()
        flash(request, "Saved. Your matches are updated.", "success")
        return redirect("/app")

    @app.get("/app/digest", response_class=HTMLResponse)
    def digest_preview(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
        digest = build_digest(db, user, settings, preview=True)
        sends = db.scalars(select(DigestSend).where(DigestSend.user_id == user.id)
                           .order_by(DigestSend.sent_at.desc()).limit(10)).all()
        return render(request, "app/digest.html", {"user": user, "digest": digest, "sends": sends})

    @app.get("/app/digest/raw", response_class=HTMLResponse)
    def digest_raw(user: User = Depends(require_user), db: Session = Depends(get_db)):
        return HTMLResponse(build_digest(db, user, settings, preview=True).html)

    @app.get("/api/feed")
    def api_feed(user: User = Depends(require_user), db: Session = Depends(get_db), limit: int = 50):
        result = build_feed(db, user.id, user.profile)
        return {
            "matches": [{
                "id": m.job.id, "title": m.job.title, "company": m.job.company_name,
                "location": m.job.location, "url": m.job.url, "score": m.score,
                "reasons": m.reasons, "cautions": m.cautions,
                "seniority": m.facts.seniority if m.facts else None,
                "min_years": m.facts.min_years if m.facts else None,
            } for m in result.matches[: max(1, min(limit, 500))]],
            "hidden": dict(result.exclusion_counts),
        }

    # ------------------------------------------------------------------ admin

    @app.get("/admin", response_class=HTMLResponse)
    def admin(request: Request, user: User = Depends(require_admin), db: Session = Depends(get_db)):
        now = utcnow()
        week = now - timedelta(days=7)
        open_q = select(func.count(Job.id)).where(Job.closed_at.is_(None))
        facts_total = db.scalar(select(func.count(JobFacts.job_id)).join(Job).where(Job.closed_at.is_(None))) or 0
        open_jobs = db.scalar(open_q) or 0
        verified = dict(db.execute(select(JobFacts.verified_status, func.count()).group_by(JobFacts.verified_status)).all())
        agree, corrected = verified.get("agree", 0), verified.get("corrected", 0)
        llm = db.execute(select(
            LLMCall.purpose, LLMCall.model, func.count(), func.sum(LLMCall.input_tokens),
            func.sum(LLMCall.output_tokens), func.sum(LLMCall.cost_usd),
            func.sum(case((LLMCall.ok.is_(True), 1), else_=0)),
        ).group_by(LLMCall.purpose, LLMCall.model)).all()
        total_cost = sum((row[5] or 0) for row in llm)
        extracted_by_model = db.scalar(select(func.count(func.distinct(LLMCall.job_id))).where(
            LLMCall.purpose == "extract", LLMCall.ok.is_(True))) or 0
        users = db.scalar(select(func.count(User.id))) or 0
        new_users = db.scalar(select(func.count(User.id)).where(User.created_at >= week)) or 0
        active = db.scalar(select(func.count(func.distinct(UserJob.user_id))).where(UserJob.updated_at >= week)) or 0
        hide_reasons = db.execute(select(UserJob.hide_reason, func.count()).where(UserJob.status == "hidden")
                                  .group_by(UserJob.hide_reason)).all()
        sources = db.scalars(select(Source).order_by(Source.enabled.desc(), Source.company_name)).all()
        open_by_source = dict(db.execute(select(Job.source_id, func.count()).where(Job.closed_at.is_(None))
                                         .group_by(Job.source_id)).all())
        runs = db.scalars(select(Run).order_by(Run.id.desc()).limit(12)).all()
        seniority_mix = db.execute(select(JobFacts.seniority, func.count()).join(Job).where(
            Job.closed_at.is_(None), JobFacts.is_engineering.is_(True)).group_by(JobFacts.seniority)).all()
        return render(request, "admin/index.html", {
            "user": user, "settings": settings, "runner": runner.state,
            "kpi": {
                "open_jobs": open_jobs, "facts": facts_total,
                "closed_week": db.scalar(select(func.count(Job.id)).where(Job.closed_at >= week)) or 0,
                "new_week": db.scalar(select(func.count(Job.id)).where(Job.first_seen_at >= week)) or 0,
                "users": users, "new_users": new_users, "active_users": active,
                "digests_week": db.scalar(select(func.count(DigestSend.id)).where(DigestSend.sent_at >= week)) or 0,
                "agree": agree, "corrected": corrected, "verify_errors": verified.get("error", 0),
                "agreement_rate": (agree / (agree + corrected)) if (agree + corrected) else None,
                "total_cost": total_cost,
                "cost_per_job": (total_cost / extracted_by_model) if extracted_by_model else None,
                "extracted_by_model": extracted_by_model,
            },
            "llm": llm, "hide_reasons": hide_reasons, "hide_labels": HIDE_REASONS,
            "sources": sources, "open_by_source": open_by_source, "runs": runs,
            "seniority_mix": seniority_mix, "seniorities": SENIORITIES,
        })

    @app.post("/admin/run/{kind}", dependencies=[Depends(check_csrf)])
    def admin_run(request: Request, kind: str, user: User = Depends(require_admin)):
        if kind not in {"ingest", "extract", "digest", "all"}:
            raise HTTPException(400, "Unknown run type.")
        if runner.start(kind):
            flash(request, f"Started {kind}. Refresh in a minute to see results.", "success")
        else:
            flash(request, f"Already running {runner.state.running}. Try again when it finishes.", "error")
        return redirect("/admin")

    @app.post("/admin/sources", dependencies=[Depends(check_csrf)])
    def admin_add_source(request: Request, url: str = Form(""), company_name: str = Form(""),
                         industry: str = Form(""), user: User = Depends(require_admin), db: Session = Depends(get_db)):
        if not company_name.strip():
            flash(request, "Give the company a name.", "error")
            return redirect("/admin#sources")
        try:
            source, created = add_source_from_url(db, url, company_name.strip(), industry.strip() or None)
        except ValueError as exc:
            flash(request, str(exc), "error")
            return redirect("/admin#sources")
        db.commit()
        flash(request, f"{'Added' if created else 'Updated'} {source.company_name} ({source.ats}). "
                       "Run ingest to fetch its jobs.", "success")
        return redirect("/admin#sources")

    @app.post("/admin/sources/{source_id}/toggle", dependencies=[Depends(check_csrf)])
    def admin_toggle_source(request: Request, source_id: int, user: User = Depends(require_admin),
                            db: Session = Depends(get_db)):
        source = db.get(Source, source_id)
        if source is None:
            raise HTTPException(404, "No such source.")
        source.enabled = not source.enabled
        db.commit()
        return redirect("/admin#sources")

    @app.get("/admin/review", response_class=HTMLResponse)
    def admin_review(request: Request, status: str = "corrected", user: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
        if status not in {"corrected", "agree", "error"}:
            status = "corrected"
        rows = db.scalars(select(JobFacts).options(joinedload(JobFacts.job)).where(JobFacts.verified_status == status)
                          .order_by(JobFacts.verified_at.desc().nullslast()).limit(200)).all()
        return render(request, "admin/review.html", {"user": user, "rows": rows, "status": status,
                                                     "seniorities": SENIORITIES, "families": ROLE_FAMILIES})

    @app.post("/admin/jobs/{job_id}/reextract", dependencies=[Depends(check_csrf)])
    def admin_reextract(request: Request, job_id: int, verify: int = Form(0), user: User = Depends(require_admin),
                        db: Session = Depends(get_db)):
        job = db.scalar(select(Job).options(joinedload(Job.source), joinedload(Job.facts)).where(Job.id == job_id))
        if job is None:
            raise HTTPException(404, "No such job.")
        stats = ExtractStats()
        try:
            Extractor(settings).process(db, job, stats, force_verify=bool(verify))
            db.commit()
        except ValueError as exc:
            flash(request, str(exc), "error")
            return redirect(f"/app/jobs/{job_id}")
        msg = "Re-extracted." + (f" Verifier: {job.facts.verified_status}." if verify else "")
        if stats.errors:
            msg += " Errors: " + "; ".join(stats.errors[:2])
        flash(request, msg, "error" if stats.errors else "success")
        return redirect(f"/app/jobs/{job_id}")

    return app


app = None  # created lazily by `rolescout serve` / uvicorn factory


def factory() -> FastAPI:
    return create_app()
