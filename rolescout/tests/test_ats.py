import pytest

from rolescout.ats import SourceSpec, detect_source, get_connector
from rolescout.ats.base import FetchError, html_to_text, parse_relative_posted

from .fakeboards import Boards


@pytest.mark.parametrize("url,ats,token,site", [
    ("https://boards.greenhouse.io/acme", "greenhouse", "acme", None),
    ("https://job-boards.greenhouse.io/acme/jobs/123", "greenhouse", "acme", None),
    ("https://boards.greenhouse.io/embed/job_board?for=acme", "greenhouse", "acme", None),
    ("jobs.lever.co/acme", "lever", "acme", None),
    ("https://jobs.ashbyhq.com/acme", "ashby", "acme", None),
    ("https://acme.wd1.myworkdayjobs.com/en-US/External", "workday", "acme", "External"),
    ("https://acme.wd5.myworkdayjobs.com/External/job/San-Diego/X_R1", "workday", "acme", "External"),
    ("https://careers.smartrecruiters.com/AcmeCorp", "smartrecruiters", "AcmeCorp", None),
])
def test_detect_source(url, ats, token, site):
    d = detect_source(url)
    assert (d.ats, d.token, d.workday_site) == (ats, token, site)


def test_detect_rejects_unsupported():
    with pytest.raises(ValueError, match="iCIMS"):
        detect_source("https://careers-acme.icims.com/jobs")
    with pytest.raises(ValueError, match="site name"):
        detect_source("https://acme.wd1.myworkdayjobs.com/")


def test_html_to_text_unescapes_and_lists():
    text = html_to_text("&lt;p&gt;Hi&amp;nbsp;there&lt;/p&gt;&lt;ul&gt;&lt;li&gt;One&lt;/li&gt;&lt;li&gt;Two&lt;/li&gt;&lt;/ul&gt;")
    assert "Hi there" in text and "- One" in text and "- Two" in text


def test_parse_relative_posted():
    assert parse_relative_posted("Posted 30+ Days Ago") is not None
    assert parse_relative_posted("Posted Today") is not None
    assert parse_relative_posted("nonsense") is None


def test_greenhouse_lever_ashby_parse():
    boards = Boards()
    with boards.client() as c:
        gh = get_connector("greenhouse").fetch(SourceSpec("greenhouse", "acme", "Acme"), c)
        lv = get_connector("lever").fetch(SourceSpec("lever", "acme", "Acme"), c)
        ab = get_connector("ashby").fetch(SourceSpec("ashby", "acme", "Acme"), c)
    assert [j.external_id for j in gh] == ["101", "102"]
    assert "0-2 years of experience" in gh[0].description and gh[0].location == "San Diego, CA"
    assert lv[0].title == "Manufacturing Engineer I" and "1+ years" in lv[0].description
    assert lv[0].extra["salary"]["min"] == 80000
    assert [j.external_id for j in ab] == ["ash-1"]  # unlisted skipped
    assert ab[0].location == "Remote"


def test_workday_list_and_detail():
    boards = Boards()
    spec = SourceSpec("workday", "acme", "Acme", workday_host="acme.wd1.myworkdayjobs.com", workday_site="External")
    conn = get_connector("workday")
    with boards.client() as c:
        jobs = conn.fetch(spec, c)
        assert [j.external_id for j in jobs] == ["R1001", "R1002", "R1003"]
        assert all(j.description is None for j in jobs)
        detailed = conn.fetch_detail(spec, jobs[0], c)
    assert "Design catheters" in detailed.description
    assert detailed.url.startswith("https://acme.wd1.myworkdayjobs.com/External/")


def test_smartrecruiters():
    boards = Boards()
    spec = SourceSpec("smartrecruiters", "AcmeCorp", "Acme")
    conn = get_connector("smartrecruiters")
    with boards.client() as c:
        jobs = conn.fetch(spec, c)
        job = conn.fetch_detail(spec, jobs[0], c)
    assert job.location == "Oceanside, CA, us"
    assert "Support CAPA" in job.description and "0-2 years" in job.description


def test_fetch_error_on_http_failure():
    boards = Boards()
    boards.fail.add("boards-api.greenhouse.io")
    with boards.client() as c, pytest.raises(FetchError):
        get_connector("greenhouse").fetch(SourceSpec("greenhouse", "acme", "Acme"), c)
