"""Fake ATS servers (httpx.MockTransport) that mimic each public API's JSON shape."""

import json

import httpx

GREENHOUSE = {
    "jobs": [
        {"id": 101, "title": "Associate Mechanical Engineer", "absolute_url": "https://boards.greenhouse.io/acme/jobs/101",
         "location": {"name": "San Diego, CA"}, "updated_at": "2026-09-30T10:00:00-07:00", "first_published": "2026-09-28T09:00:00-07:00",
         "departments": [{"name": "R&D"}],
         "content": "&lt;p&gt;Design things.&lt;/p&gt;&lt;h3&gt;Requirements&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;BS in Mechanical Engineering&lt;/li&gt;&lt;li&gt;0-2 years of experience&lt;/li&gt;&lt;/ul&gt;"},
        {"id": 102, "title": "Senior Quality Engineer", "absolute_url": "https://boards.greenhouse.io/acme/jobs/102",
         "location": {"name": "Irvine, CA"}, "updated_at": "2026-09-29T10:00:00Z", "departments": [],
         "content": "&lt;p&gt;8+ years of experience in quality.&lt;/p&gt;"},
    ]
}

LEVER = [
    {"id": "aaa-111", "text": "Manufacturing Engineer I", "hostedUrl": "https://jobs.lever.co/acme/aaa-111",
     "createdAt": 1759000000000, "categories": {"location": "Carlsbad, CA", "team": "Ops", "commitment": "Full-time"},
     "descriptionPlain": "Own production lines.", "lists": [{"text": "Requirements", "content": "<li>1+ years of experience</li><li>BS in Industrial Engineering</li>"}],
     "additionalPlain": "", "workplaceType": "onsite", "salaryRange": {"min": 80000, "max": 95000, "currency": "USD", "interval": "per-year-salary"}},
]

ASHBY = {
    "jobs": [
        {"id": "ash-1", "title": "Software Engineer, New Grad", "location": "Remote", "isRemote": True, "department": "Eng",
         "descriptionPlain": "0-1 years of experience. BS in Computer Science.", "publishedAt": "2026-10-01T00:00:00Z",
         "jobUrl": "https://jobs.ashbyhq.com/acme/ash-1", "isListed": True, "workplaceType": "Remote",
         "compensation": {"compensationTierSummary": "$110K – $130K"}},
        {"id": "ash-2", "title": "Unlisted Role", "isListed": False, "jobUrl": "https://x", "descriptionPlain": ""},
    ]
}

WORKDAY_PAGE = [
    {"title": "R&D Engineer I", "externalPath": "/job/San-Diego/R-D-Engineer-I_R1001", "locationsText": "San Diego, CA",
     "postedOn": "Posted 3 Days Ago", "bulletFields": ["R1001"]},
    {"title": "Payroll Specialist", "externalPath": "/job/San-Diego/Payroll_R1002", "locationsText": "San Diego, CA",
     "postedOn": "Posted Today", "bulletFields": ["R1002"]},
    {"title": "Principal Systems Engineer", "externalPath": "/job/Remote/Principal_R1003", "locationsText": "Remote",
     "postedOn": "Posted 30+ Days Ago", "bulletFields": ["R1003"]},
]
WORKDAY_DETAIL = {
    "/job/San-Diego/R-D-Engineer-I_R1001": {"jobPostingInfo": {"title": "R&D Engineer I", "jobDescription": "<p>Design catheters.</p><p>0-2 years of experience. BS in Biomedical Engineering.</p>",
        "location": "San Diego, CA", "startDate": "2026-10-03", "externalUrl": "https://acme.wd1.myworkdayjobs.com/External/job/San-Diego/R-D-Engineer-I_R1001", "timeType": "Full time"}},
    "/job/Remote/Principal_R1003": {"jobPostingInfo": {"title": "Principal Systems Engineer", "jobDescription": "<p>15+ years of experience.</p>", "location": "Remote"}},
}

SMARTRECRUITERS_LIST = {"totalFound": 1, "content": [
    {"id": "sr-9", "name": "Quality Engineer I", "releasedDate": "2026-10-02T00:00:00.000Z",
     "location": {"city": "Oceanside", "region": "CA", "country": "us", "remote": False}, "department": {"label": "Quality"}},
]}
SMARTRECRUITERS_DETAIL = {"jobAd": {"sections": {
    "jobDescription": {"title": "Job Description", "text": "<p>Support CAPA.</p>"},
    "qualifications": {"title": "Qualifications", "text": "<ul><li>0-2 years of experience</li></ul>"}}}}


class Boards:
    """Mutable fake: tests edit the payloads between ingest runs."""

    def __init__(self):
        self.greenhouse = json.loads(json.dumps(GREENHOUSE))
        self.lever = json.loads(json.dumps(LEVER))
        self.ashby = json.loads(json.dumps(ASHBY))
        self.workday = json.loads(json.dumps(WORKDAY_PAGE))
        self.fail = set()
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(f"{request.method} {url}")
        host = request.url.host
        if host in self.fail:
            return httpx.Response(503, text="down")
        if host == "boards-api.greenhouse.io":
            return httpx.Response(200, json=self.greenhouse)
        if host == "api.lever.co":
            return httpx.Response(200, json=self.lever)
        if host == "api.ashbyhq.com":
            return httpx.Response(200, json=self.ashby)
        if host == "api.smartrecruiters.com":
            if request.url.path.endswith("/postings"):
                return httpx.Response(200, json=SMARTRECRUITERS_LIST)
            return httpx.Response(200, json=SMARTRECRUITERS_DETAIL)
        if host.endswith("myworkdayjobs.com"):
            if request.method == "POST":
                body = json.loads(request.content)
                offset, limit = body["offset"], body["limit"]
                page = self.workday[offset: offset + limit]
                return httpx.Response(200, json={"total": len(self.workday) if offset == 0 else 0, "jobPostings": page})
            path = request.url.path.split("/External", 1)[1]
            if path in WORKDAY_DETAIL:
                return httpx.Response(200, json=WORKDAY_DETAIL[path])
            return httpx.Response(404)
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))
