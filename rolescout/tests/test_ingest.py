from sqlalchemy import select

from rolescout.ats import detect_source
from rolescout.config import get_settings
from rolescout.ingest import run_ingest, upsert_source
from rolescout.models import Job, Source

from .fakeboards import Boards


def _add(session, url, name="Acme"):
    src, _ = upsert_source(session, detect_source(url), name)
    session.commit()
    return src


def _jobs(session, src):
    return {j.external_id: j for j in session.scalars(select(Job).where(Job.source_id == src.id))}


def test_ingest_new_then_close_after_two_misses_then_reopen(session):
    boards = Boards()
    src = _add(session, "https://boards.greenhouse.io/acme")
    with boards.client() as c:
        st = run_ingest(session, client=c)
        assert st.new == 2 and st.sources_ok == 1
        removed = boards.greenhouse["jobs"].pop()
        run_ingest(session, client=c)
        jobs = _jobs(session, src)
        assert jobs["102"].closed_at is None and jobs["102"].missed_runs == 1  # one miss: still open
        st = run_ingest(session, client=c)
        assert st.closed == 1
        session.expire_all()
        assert _jobs(session, src)["102"].closed_at is not None
        boards.greenhouse["jobs"].append(removed)
        st = run_ingest(session, client=c)
        assert st.reopened == 1
        session.expire_all()
        assert _jobs(session, src)["102"].closed_at is None


def test_failed_fetch_never_closes_jobs(session):
    boards = Boards()
    src = _add(session, "https://boards.greenhouse.io/acme")
    with boards.client() as c:
        run_ingest(session, client=c)
        boards.fail.add("boards-api.greenhouse.io")
        for _ in range(3):
            st = run_ingest(session, client=c)
            assert st.sources_failed == 1
    session.expire_all()
    assert all(j.closed_at is None for j in _jobs(session, src).values())
    assert session.get(Source, src.id).last_status == "error"


def test_empty_board_with_many_open_jobs_is_treated_as_outage(session):
    boards = Boards()
    boards.greenhouse["jobs"] = [dict(boards.greenhouse["jobs"][0], id=1000 + i) for i in range(6)]
    src = _add(session, "https://boards.greenhouse.io/acme")
    with boards.client() as c:
        run_ingest(session, client=c)
        boards.greenhouse["jobs"] = []
        for _ in range(3):
            run_ingest(session, client=c)
    session.expire_all()
    assert all(j.closed_at is None for j in _jobs(session, src).values())


def test_workday_lazy_details_only_for_engineering_titles(session):
    boards = Boards()
    _add(session, "https://acme.wd1.myworkdayjobs.com/External")
    with boards.client() as c:
        st = run_ingest(session, client=c)
        detail_calls = [x for x in boards.calls if x.startswith("GET")]
        assert st.new == 3
        # R&D Engineer I + Principal Systems Engineer fetched; Payroll Specialist skipped.
        assert len(detail_calls) == 2
        boards.calls.clear()
        run_ingest(session, client=c)
        assert not [x for x in boards.calls if x.startswith("GET")]  # nothing refetched
    jobs = {j.title: j for j in session.scalars(select(Job))}
    assert "Design catheters" in jobs["R&D Engineer I"].description
    assert jobs["Payroll Specialist"].description is None


def test_content_change_updates_hash(session):
    boards = Boards()
    src = _add(session, "https://jobs.lever.co/acme")
    with boards.client() as c:
        run_ingest(session, client=c)
        h1 = _jobs(session, src)["aaa-111"].content_hash
        boards.lever[0]["descriptionPlain"] = "Own production lines. Now with robots."
        st = run_ingest(session, client=c)
    assert st.updated == 1
    session.expire_all()
    assert _jobs(session, src)["aaa-111"].content_hash != h1


def test_close_threshold_is_configurable(session, monkeypatch):
    monkeypatch.setenv("ROLESCOUT_CLOSE_AFTER_MISSED_RUNS", "1")
    get_settings.cache_clear()
    boards = Boards()
    src = _add(session, "https://boards.greenhouse.io/acme")
    with boards.client() as c:
        run_ingest(session, client=c)
        boards.greenhouse["jobs"].pop()
        st = run_ingest(session, client=c)
    assert st.closed == 1
