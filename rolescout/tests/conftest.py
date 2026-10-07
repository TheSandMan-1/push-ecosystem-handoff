import os

import pytest

os.environ.setdefault("ROLESCOUT_SECRET_KEY", "test-secret-key-not-for-production")


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    """Every test gets its own database, outbox, and default settings."""
    from rolescout import config, db

    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("ROLESCOUT_") and key != "ROLESCOUT_SECRET_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("ROLESCOUT_OUTBOX_DIR", str(tmp_path / "outbox"))
    monkeypatch.delenv("SMTP_HOST", raising=False)
    config.get_settings.cache_clear()
    db.configure(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def session():
    from rolescout.db import new_session

    s = new_session()
    try:
        yield s
    finally:
        s.close()
