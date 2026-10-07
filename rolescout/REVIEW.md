# Review checklist (advanced-model pass)

Run before merging changes to extraction, matching, ingestion, auth or billing. Report findings by severity with file:line.

## Correctness that users feel
- [ ] Can a senior role (title or required years) reach an entry-level user's feed? Trace `heuristics.extract` → `apply_guardrails` → `matching.evaluate`.
- [ ] Can a non-engineering "engineer" job (hotel maintenance, sales, technician, recruiter) pass `require_engineering`?
- [ ] Required vs preferred years: does a "preferred 5+ years" line ever become the requirement?
- [ ] Liveness: any path where a failed/partial fetch closes jobs? Any path where a closed job is never reopened?
- [ ] Digests: can a job be emailed twice? Can a hidden/applied job appear?

## Security
- [ ] Every POST route has `check_csrf`; every user-data route has `require_user`/`require_admin`.
- [ ] No open redirects (`next` param), no IDOR on `/app/jobs/{id}/status` (writes are keyed by the session user).
- [ ] Templates autoescape; no `|safe` on user or posting content.
- [ ] Secrets only from env; nothing logged that includes passwords, API keys, or session cookies.

## Model tiers
- [ ] No model call in a request handler or per-user path.
- [ ] Every model call is logged to `llm_calls` with tokens and cost, including failures.
- [ ] Claude calls: structured outputs, `stop_reason` checked for `refusal`/`max_tokens`, fallbacks on supported models, no unsupported params.
- [ ] Local provider errors degrade to rules, never crash a batch.

## Data
- [ ] Schema changes come with a migration plan (no Alembic yet, so call it out).
- [ ] Queries that scale with open jobs or users are flagged if they load everything into memory on a hot path.
