# RoleScout: guide for AI coding sessions

Job search for early-career engineers. Python 3.11+, FastAPI, SQLAlchemy 2, Jinja templates, no JS build step.

## Commands (run from this directory)

- `pip install -e ".[dev]"`, then `python -m pytest -q` (all tests must pass) and `rolescout eval` (rules accuracy must not drop)
- `rolescout demo && rolescout serve` for a clickable app with fictional data

## Map

- `rolescout/ats/`: one connector per job system; `detect_source` maps a careers URL to a connector
- `rolescout/ingest.py`: upsert jobs and track open/closed (never close on a failed fetch)
- `rolescout/extract/heuristics.py`: rules, the free tier and source of truth for explicit signals
- `rolescout/extract/pipeline.py`: tiering, guardrails, verifier triggers, cost logging
- `rolescout/extract/providers.py`: local (OpenAI-compatible) and Claude (official SDK) providers
- `rolescout/matching.py`: per-user scoring and the single exclusion reason per job
- `rolescout/web/`: routes, templates, `static/app.css` (all colors are tokens; keep light and dark in sync)
- `rolescout/seeds/eval_jobs.yaml`: labeled postings; add a case for every extraction bug before fixing it

## Model tiers and who does what

- Bulk extraction: a local model (Ollama/LM Studio) or a cheap Claude model. Never trust it on explicit title levels; guardrails re-apply rules.
- Verification: Claude Sonnet 5.5 (`claude-sonnet-5-5`) on disagreements, low confidence and a QA sample.
- Code review of changes: run the most capable available model (Opus or Fable class) with `REVIEW.md` before merging anything that touches extraction, matching, auth, or ingestion liveness.

## Rules

- Expensive work is per job, never per user. Don't add model calls to request handlers or matching. (One deliberate exception: the admin-only Re-extract button on a single job.)
- A failed, suspicious, or partial (page-capped) fetch must never close jobs. Do all HTTP before writing to the session.
- Jobs without extracted facts are never shown or emailed.
- Every POST needs CSRF (`check_csrf` dependency). Every page that shows user data needs `require_user`.
- Claude API code goes through the official `anthropic` SDK. Sonnet 5.5 / Opus 5.5 / Fable 5.1 calls use `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")`. Don't send `budget_tokens`, `temperature`, or forced `tool_choice` to those models.
- New extraction behavior ships with an `eval_jobs.yaml` case.
