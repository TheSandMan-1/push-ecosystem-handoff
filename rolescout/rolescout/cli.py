"""Command-line interface: `rolescout <command>`. Run `rolescout --help`."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from importlib import resources
from pathlib import Path

import yaml


def _setup(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from .db import init_db

    init_db()


def cmd_init(args) -> int:
    from .auth import AuthError, create_user
    from .db import session_scope

    if args.admin_email:
        password = args.admin_password or getpass.getpass("Admin password (8+ chars): ")
        with session_scope() as s:
            try:
                create_user(s, args.admin_email, password, is_admin=True)
            except AuthError as exc:
                print(f"Could not create admin: {exc}")
                return 1
        print(f"Created admin {args.admin_email}")
    print("Database ready.")
    return 0


def cmd_create_user(args) -> int:
    from .auth import AuthError, create_user
    from .db import session_scope

    password = args.password or getpass.getpass("Password (8+ chars): ")
    with session_scope() as s:
        try:
            user = create_user(s, args.email, password, args.name, is_admin=args.admin)
        except AuthError as exc:
            print(exc)
            return 1
        print(f"Created {'admin ' if user.is_admin else ''}user {user.email}")
    return 0


def cmd_seed(args) -> int:
    from .db import session_scope
    from .ingest import add_source_from_url

    text = Path(args.file).read_text() if args.file else resources.files("rolescout").joinpath("seeds/sources.yaml").read_text()
    entries = yaml.safe_load(text) or []
    added = updated = failed = 0
    with session_scope() as s:
        for e in entries:
            try:
                _src, created = add_source_from_url(s, e["url"], e["company"], e.get("industry"))
                added += created
                updated += not created
            except (ValueError, KeyError) as exc:
                failed += 1
                print(f"  skip {e}: {exc}")
    print(f"Sources: {added} added, {updated} updated, {failed} skipped. Next: rolescout check-sources")
    return 0


def cmd_add_source(args) -> int:
    from .db import session_scope
    from .ingest import add_source_from_url, probe_url

    if not args.no_probe:
        try:
            detected, count = probe_url(args.url)
            print(f"Detected {detected.ats} board '{detected.token}' with {count} jobs.")
        except Exception as exc:
            print(f"Could not fetch that board: {exc}")
            if not args.force:
                print("Use --force to add it anyway.")
                return 1
    with session_scope() as s:
        try:
            src, created = add_source_from_url(s, args.url, args.company, args.industry)
        except ValueError as exc:
            print(exc)
            return 1
        print(f"{'Added' if created else 'Updated'} {src.company_name} ({src.slug})")
    return 0


def cmd_probe(args) -> int:
    from .ingest import probe_url

    try:
        detected, count = probe_url(args.url)
    except Exception as exc:
        print(f"Failed: {exc}")
        return 1
    print(f"{detected.ats} · token={detected.token} · site={detected.workday_site or '-'} · {count} jobs")
    return 0


def cmd_check_sources(args) -> int:
    from sqlalchemy import select

    from .ats import get_connector
    from .db import session_scope
    from .ingest import make_client, spec_for
    from .models import Source

    ok = bad = 0
    with session_scope() as s, make_client() as client:
        for src in s.scalars(select(Source).where(Source.enabled.is_(True)).order_by(Source.company_name)):
            try:
                jobs = get_connector(src.ats).fetch(spec_for(src), client)
                print(f"  OK    {src.company_name:28} {src.ats:15} {len(jobs):5} jobs")
                ok += 1
            except Exception as exc:
                print(f"  FAIL  {src.company_name:28} {src.ats:15} {exc}")
                bad += 1
    print(f"{ok} ok, {bad} failed. Fix failing URLs in the admin page or with `rolescout add-source`.")
    return 0 if bad == 0 else 2


def cmd_ingest(args) -> int:
    from .db import session_scope
    from .ingest import run_ingest

    with session_scope() as s:
        st = run_ingest(s)
    print(json.dumps(st.as_dict(), indent=2, default=str))
    return 0 if st.sources_failed == 0 else 2


def cmd_extract(args) -> int:
    from .jobs import execute

    print(execute("extract"))
    return 0


def cmd_digest(args) -> int:
    from .db import session_scope
    from .digest import run_digests

    with session_scope() as s:
        st = run_digests(s, force=args.force)
    print(json.dumps(st, indent=2))
    return 0


def cmd_run(args) -> int:
    from .jobs import execute

    print(execute("all" if args.with_digest else "ingest"))
    if not args.with_digest:
        print(execute("extract"))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("rolescout.web.app:factory", factory=True, host=args.host, port=args.port, reload=args.reload,
                log_level="info")
    return 0


def cmd_demo(args) -> int:
    from .db import session_scope
    from .demo import load_demo
    from .jobs import execute

    with session_scope() as s:
        info = load_demo(s)
    print(execute("extract"))
    print(f"Demo ready: {info['jobs_created']} fictional jobs.")
    print(f"Log in at http://localhost:8000 with {info['user']} / {info['password']}")
    return 0


def cmd_eval(args) -> int:
    from .config import get_settings
    from .evaluate import format_report, load_cases, run_eval
    from .extract.providers import make_provider

    settings = get_settings()
    kind = args.extractor or "heuristic"
    model = args.model or (settings.extractor_model if kind == "local" else settings.verifier_model)
    provider = make_provider(kind, model, settings, effort=args.effort)
    report = run_eval(load_cases(args.file), provider, settings)
    print(format_report(report))
    return 0 if not report.errors else 2


def cmd_stats(args) -> int:
    from sqlalchemy import func, select

    from .db import session_scope
    from .models import Job, JobFacts, LLMCall, Source, User

    with session_scope() as s:
        print(f"Sources:       {s.scalar(select(func.count(Source.id)))}")
        print(f"Open jobs:     {s.scalar(select(func.count(Job.id)).where(Job.closed_at.is_(None)))}")
        print(f"Closed jobs:   {s.scalar(select(func.count(Job.id)).where(Job.closed_at.is_not(None)))}")
        print(f"Extracted:     {s.scalar(select(func.count(JobFacts.job_id)))}")
        print(f"Users:         {s.scalar(select(func.count(User.id)))}")
        for purpose, model, n, cost in s.execute(select(LLMCall.purpose, LLMCall.model, func.count(),
                                                        func.sum(LLMCall.cost_usd)).group_by(LLMCall.purpose, LLMCall.model)):
            print(f"Model {purpose:8} {model:28} {n:6} calls  ${cost or 0:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rolescout", description="RoleScout: real, open jobs at your level.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create the database (and optionally an admin)")
    p.add_argument("--admin-email")
    p.add_argument("--admin-password")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("create-user", help="Create a user")
    p.add_argument("email")
    p.add_argument("--password")
    p.add_argument("--name")
    p.add_argument("--admin", action="store_true")
    p.set_defaults(func=cmd_create_user)

    p = sub.add_parser("seed", help="Load company boards from a YAML file (default: bundled starter list)")
    p.add_argument("--file")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("add-source", help="Add a company board by its careers URL")
    p.add_argument("url")
    p.add_argument("--company", required=True)
    p.add_argument("--industry")
    p.add_argument("--no-probe", action="store_true", help="Skip the test fetch")
    p.add_argument("--force", action="store_true", help="Add even if the test fetch fails")
    p.set_defaults(func=cmd_add_source)

    p = sub.add_parser("probe", help="Detect and test-fetch a careers URL without saving")
    p.add_argument("url")
    p.set_defaults(func=cmd_probe)

    sub.add_parser("check-sources", help="Test-fetch every enabled board").set_defaults(func=cmd_check_sources)
    sub.add_parser("ingest", help="Fetch all boards and update open/closed status").set_defaults(func=cmd_ingest)
    sub.add_parser("extract", help="Extract facts for new or changed jobs").set_defaults(func=cmd_extract)

    p = sub.add_parser("digest", help="Send digests to users who are due")
    p.add_argument("--force", action="store_true", help="Send even if not due yet")
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("run", help="Ingest then extract (one full refresh)")
    p.add_argument("--with-digest", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="Run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    sub.add_parser("demo", help="Load fictional demo data and a demo login").set_defaults(func=cmd_demo)

    p = sub.add_parser("eval", help="Score an extractor against hand-labeled postings")
    p.add_argument("--extractor", choices=["heuristic", "local", "anthropic"])
    p.add_argument("--model")
    p.add_argument("--effort", default="low")
    p.add_argument("--file")
    p.set_defaults(func=cmd_eval)

    sub.add_parser("stats", help="Print counts and model spend").set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    _setup(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
