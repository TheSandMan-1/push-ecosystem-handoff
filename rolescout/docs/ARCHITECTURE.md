# Architecture

## The one rule that makes the economics work

Everything expensive happens **once per job** and is shared by every user: fetching, extraction, verification. Per-user work (matching) is deterministic filtering over stored facts and costs nothing. Model spend scales with the number of postings you watch, not the number of users.

## Pipeline

```
company boards ──ingest──▶ jobs ──extract──▶ job_facts ──match (per user, free)──▶ feed / digest
   (ATS APIs)       │                  │
                    │                  ├─ 1. rules (heuristics.py): every job, free
                    │                  ├─ 2. bulk model: only titles that could be engineering
                    │                  └─ 3. verifier: disagreements, low confidence, 5% sample
                    └─ liveness: missing from 2 successful fetches in a row = closed
```

### Ingest (`ingest.py`, `ats/`)

- One connector per applicant tracking system. Each turns a board into `RawJob`s.
- Workday and SmartRecruiters list endpoints have no descriptions. Details are fetched lazily, only for **new** jobs whose title passes `title_worth_reading`. That keeps a 2,000-job Workday board to a few hundred detail requests on the first run and almost none after.
- **Liveness.** A job closes only after it is missing from `close_after_missed_runs` (default 2) *successful* fetches. A failed fetch never closes anything. A board that suddenly returns zero jobs when it had 5+ open is treated as an outage. A job that reappears is reopened. A capped page walk (huge Workday or SmartRecruiters boards) is marked partial and skips the close pass.
- All HTTP for a board (list and details) finishes before its database rows are written, so ingest never holds the SQLite write lock while waiting on the network.
- `content_hash` (title + location + description) drives re-extraction when a posting is edited.

### Extract (`extract/`)

- `heuristics.py` is the ground truth for what rules see reliably: explicit title levels (Senior, Staff, II, Intern), required vs preferred years, hotel vs factory maintenance, clearance, ITAR, salary, work mode.
- `pipeline.py` calls the bulk model, then `apply_guardrails` re-applies certain rule facts so a model can't override them (an explicit "Senior" always wins; clearance is OR'd).
- The verifier sees the posting plus the proposed facts and returns `agree` or `corrected`. Trigger reasons are stored in `verifier_notes`, so `/admin/review` shows *why* each job was escalated.
- `providers.py`: `LocalProvider` (any OpenAI-compatible server; tries `json_schema`, falls back to `json_object`, remembers which works) and `AnthropicProvider` (official SDK, structured outputs, refusal fallback for models that support it, effort only where supported). Every call lands in `llm_calls` with tokens and cost.

### Match (`matching.py`)

- Jobs that haven't been analyzed yet are excluded (`pending`), never shown.
- Each job is either a match with a score and reasons, or excluded with exactly one reason. That is what powers "We hid 412 postings: 230 too senior..." and the Filtered-out tab, which lets users audit the filter.
- Mid-level titles still pass when the required years are within the user's experience ("Asks for only 2 yrs").
- Discipline mismatch is a penalty, not an exclusion, because postings often list several fields.

### Feedback loop

Hides carry a reason (`too_senior`, `not_engineering`, ...). `/admin` aggregates them. Each one is a labeled extraction failure; add it to `seeds/eval_jobs.yaml`, fix the rule or prompt, and prove the fix with `rolescout eval`.

## Data model

`sources` → `jobs` → `job_facts` (1:1). `users` → `profiles` (1:1), `user_jobs` (saved/applied/hidden), `digest_sends` (job ids already emailed, so nothing is sent twice). `runs` and `llm_calls` feed the admin dashboard.

## Web

FastAPI + Jinja templates + one small JS file. Signed-cookie sessions, CSRF token on every POST (form field or `X-CSRF-Token` header), scrypt password hashing, open-redirect check on login. Light and dark themes from one set of CSS tokens.

## Known limits (worth fixing before real scale)

- No migrations yet (tables are created with `create_all`). Add Alembic before the first schema change in production.
- Matching loads every open job per request. Fine to ~20k open jobs; past that, push the hard filters into SQL.
- No login rate limiting or password reset yet.
- iCIMS, Taleo, SuccessFactors and Oracle boards are not supported; many large employers use them.
- Workday's JSON endpoint is undocumented. Keep request rates low and expect it to change.
