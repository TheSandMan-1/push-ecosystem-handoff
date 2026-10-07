"""Runtime settings, read from environment variables (and an optional .env file).

Every setting has a default that works on a laptop with zero configuration:
SQLite database, heuristic-only extraction, no email sending.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _list(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [v.strip().lower() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///rolescout.db"
    secret_key: str = ""
    base_url: str = "http://localhost:8000"
    admin_emails: list[str] = field(default_factory=list)
    signups_open: bool = True

    # Model tiers. "heuristic" needs nothing installed; "local" talks to any
    # OpenAI-compatible server (Ollama, LM Studio, llama.cpp); "anthropic"
    # uses the Claude API (needs ANTHROPIC_API_KEY or an `ant auth login` profile).
    extractor: str = "heuristic"
    extractor_model: str = "qwen2.5:7b-instruct"
    verifier: str = "none"
    verifier_model: str = "claude-sonnet-5-5"
    verify_sample_rate: float = 0.05
    verify_confidence_below: float = 0.7
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_api_key: str = "local"
    llm_max_description_chars: int = 12000
    extract_batch_limit: int = 200

    # Ingestion
    http_timeout: float = 30.0
    close_after_missed_runs: int = 2
    workday_page_limit: int = 20
    workday_max_jobs: int = 2000
    workday_search_text: str = ""
    user_agent: str = "RoleScout/0.1 (+https://github.com/TheSandMan-1)"

    # Scheduler inside `rolescout serve` (minutes, 0 = off)
    schedule_minutes: int = 0

    # Email. If SMTP_HOST is empty, digests are written to OUTBOX_DIR instead.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "RoleScout <digest@localhost>"
    outbox_dir: str = "outbox"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(Path.cwd() / ".env")
    secret = os.environ.get("ROLESCOUT_SECRET_KEY", "")
    if not secret:
        # Stable across restarts for local use: persist a generated key.
        key_file = Path(os.environ.get("ROLESCOUT_SECRET_FILE", ".rolescout_secret"))
        if key_file.is_file():
            secret = key_file.read_text().strip()
        else:
            secret = secrets.token_urlsafe(48)
            try:
                key_file.write_text(secret)
            except OSError:
                pass
    return Settings(
        database_url=os.environ.get("DATABASE_URL", Settings.database_url),
        secret_key=secret,
        base_url=os.environ.get("ROLESCOUT_BASE_URL", Settings.base_url).rstrip("/"),
        admin_emails=_list("ROLESCOUT_ADMIN_EMAILS"),
        signups_open=_bool("ROLESCOUT_SIGNUPS_OPEN", True),
        extractor=os.environ.get("ROLESCOUT_EXTRACTOR", Settings.extractor).lower(),
        extractor_model=os.environ.get("ROLESCOUT_EXTRACTOR_MODEL", Settings.extractor_model),
        verifier=os.environ.get("ROLESCOUT_VERIFIER", Settings.verifier).lower(),
        verifier_model=os.environ.get("ROLESCOUT_VERIFIER_MODEL", Settings.verifier_model),
        verify_sample_rate=_float("ROLESCOUT_VERIFY_SAMPLE_RATE", Settings.verify_sample_rate),
        verify_confidence_below=_float("ROLESCOUT_VERIFY_CONFIDENCE_BELOW", Settings.verify_confidence_below),
        local_llm_base_url=os.environ.get("ROLESCOUT_LOCAL_LLM_BASE_URL", Settings.local_llm_base_url).rstrip("/"),
        local_llm_api_key=os.environ.get("ROLESCOUT_LOCAL_LLM_API_KEY", Settings.local_llm_api_key),
        llm_max_description_chars=_int("ROLESCOUT_LLM_MAX_DESCRIPTION_CHARS", Settings.llm_max_description_chars),
        extract_batch_limit=_int("ROLESCOUT_EXTRACT_BATCH_LIMIT", Settings.extract_batch_limit),
        http_timeout=_float("ROLESCOUT_HTTP_TIMEOUT", Settings.http_timeout),
        close_after_missed_runs=_int("ROLESCOUT_CLOSE_AFTER_MISSED_RUNS", Settings.close_after_missed_runs),
        workday_page_limit=_int("ROLESCOUT_WORKDAY_PAGE_LIMIT", Settings.workday_page_limit),
        workday_max_jobs=_int("ROLESCOUT_WORKDAY_MAX_JOBS", Settings.workday_max_jobs),
        workday_search_text=os.environ.get("ROLESCOUT_WORKDAY_SEARCH_TEXT", ""),
        user_agent=os.environ.get("ROLESCOUT_USER_AGENT", Settings.user_agent),
        schedule_minutes=_int("ROLESCOUT_SCHEDULE_MINUTES", 0),
        smtp_host=os.environ.get("SMTP_HOST", ""),
        smtp_port=_int("SMTP_PORT", 587),
        smtp_user=os.environ.get("SMTP_USER", ""),
        smtp_password=os.environ.get("SMTP_PASSWORD", ""),
        smtp_from=os.environ.get("SMTP_FROM", Settings.smtp_from),
        outbox_dir=os.environ.get("ROLESCOUT_OUTBOX_DIR", Settings.outbox_dir),
    )
