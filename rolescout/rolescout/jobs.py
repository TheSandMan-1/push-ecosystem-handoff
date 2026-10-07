"""Run pipeline steps in the background from the web app, one at a time."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from .config import get_settings
from .db import new_session
from .models import Run, utcnow

log = logging.getLogger(__name__)


@dataclass
class RunnerState:
    running: str | None = None
    started_at: datetime | None = None
    last_message: str = ""
    history: list[str] = field(default_factory=list)


class Runner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = RunnerState()
        self._scheduler: threading.Thread | None = None
        self._stop = threading.Event()

    def busy(self) -> bool:
        return self.state.running is not None

    def start(self, kind: str) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        self.state.running = kind
        self.state.started_at = utcnow()
        thread = threading.Thread(target=self._run, args=(kind,), daemon=True, name=f"rolescout-{kind}")
        thread.start()
        return True

    def run_sync(self, kind: str) -> str:
        with self._lock:
            self.state.running = kind
            self.state.started_at = utcnow()
            try:
                return execute(kind)
            finally:
                self.state.running = None

    def _run(self, kind: str) -> None:
        try:
            self.state.last_message = execute(kind)
        except Exception as exc:  # surfaced on the admin page
            log.exception("background %s failed", kind)
            self.state.last_message = f"{kind} failed: {exc}"
            with new_session() as s:
                s.add(Run(kind=kind, status="error", error=str(exc), finished_at=utcnow()))
                s.commit()
        finally:
            self.state.history.insert(0, f"{utcnow():%H:%M} {self.state.last_message}")
            del self.state.history[10:]
            self.state.running = None
            self._lock.release()

    def start_scheduler(self, minutes: int) -> None:
        if minutes <= 0 or self._scheduler is not None:
            return

        def loop() -> None:
            while not self._stop.wait(5):
                if not self.busy():
                    self.start("all")
                self._stop.wait(minutes * 60)

        self._scheduler = threading.Thread(target=loop, daemon=True, name="rolescout-scheduler")
        self._scheduler.start()

    def stop(self) -> None:
        self._stop.set()


def execute(kind: str) -> str:
    """Run one pipeline step to completion and return a one-line summary."""
    from .digest import run_digests
    from .extract.pipeline import Extractor
    from .ingest import run_ingest

    settings = get_settings()
    started = time.monotonic()
    parts: list[str] = []
    with new_session() as session:
        if kind in ("ingest", "all"):
            st = run_ingest(session, settings)
            parts.append(f"ingest: {st.sources_ok} boards ok, {st.sources_failed} failed, "
                         f"{st.new} new, {st.closed} closed")
        if kind in ("extract", "all"):
            run = Run(kind="extract")
            session.add(run)
            session.commit()
            total = None
            # Drain the backlog in batches so one run catches up fully.
            for _ in range(50):
                st = Extractor(settings).run(session)
                if total is None:
                    total = st
                else:
                    for name in ("processed", "rules_only", "gated_out", "llm_ok", "llm_errors",
                                 "verified", "corrected", "verify_errors"):
                        setattr(total, name, getattr(total, name) + getattr(st, name))
                    total.cost_usd += st.cost_usd
                    total.errors += st.errors
                if st.processed < settings.extract_batch_limit:
                    break
            assert total is not None
            run.finished_at = utcnow()
            run.stats = total.as_dict()
            run.status = "ok" if not total.llm_errors and not total.verify_errors else "partial"
            session.commit()
            parts.append(f"extract: {total.processed} jobs, {total.llm_ok} by model, "
                         f"{total.verified} verified, ${total.cost_usd:.4f}")
        if kind in ("digest", "all"):
            st = run_digests(session, settings)
            parts.append(f"digest: {st['sent']} sent")
    return f"{'; '.join(parts)} ({time.monotonic() - started:.0f}s)"


runner = Runner()
