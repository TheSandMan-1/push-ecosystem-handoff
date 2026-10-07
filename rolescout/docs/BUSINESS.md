# RoleScout: the business

## One line

The job feed for new engineering grads where every role is real, open, and actually entry level, with the reason each posting was shown or hidden.

## The problem (validated by the founder's own search)

General job boards rank by engagement and keyword overlap. For an entry-level engineer that produces:

- Senior and staff roles in an "entry level" search
- "Engineer" jobs that aren't engineering (hotel building maintenance, sales, IT help desk)
- Defense roles that need a clearance a new grad doesn't have
- Postings that closed weeks ago but are still listed
- "Entry level" titles whose requirements ask for 5+ years

Each one costs a few minutes to open, read and reject. Across a search that's hours a week of pure filtering.

## Who it's for first

**New and recent engineering grads (0-2 years) in hardware fields** (mechanical, biomedical, electrical, manufacturing, quality) **in one region** (Southern California to start). A narrow first market is a feature: the company list is small enough to cover completely, the filtering rules are sharp, and you can reach the users directly (school alumni groups, engineering subreddits, career fairs, LinkedIn).

Expand by region (add companies) and by field (software later; it's the most crowded segment).

## Why this can win against bigger players

Be honest about the competition: LinkedIn, Indeed and Handshake have distribution; newer tools (Jobright, Simplify, Hiring Cafe and others) already use AI matching or pull directly from company job systems, some for free. *This landscape was written without a fresh check; spend an hour using each before you pitch anyone.*

The wedge isn't "AI matching." It's three things together that general tools don't optimize for:

1. **Precision over recall for one audience.** We would rather miss a borderline job than show a senior one.
2. **Transparency.** Every hidden posting has a reason, and users can audit the filter. That builds trust a black-box ranker can't.
3. **Freshness as a promise.** "Open, checked 2h ago" on every card, because we read the company's own system.

If a competitor already does all three for this audience at a quality you can't beat, stop or narrow further.

## Validation plan (2 weeks, before writing more product code)

1. Load 30-50 real company boards for the first market. Run the pipeline daily.
2. Post the value proposition with 3 real example "we filtered this out" screenshots in 3-5 places your users already are.
3. Send the weekly digest to everyone who signs up. Manually review each one before it goes out.

**Continue if, by day 14:**
- 50+ signups from organic posts
- 40%+ of week-1 signups open the week-2 digest
- 5+ people pay a $9 pre-order (or say yes to a specific price on a call)

**Stop or pivot if** signups come but nobody opens week 2 (the filtering isn't better enough), or nobody will pay (consider the B2B route below).

## Pricing hypothesis

- Free during validation.
- **Job Hunt Pass: $9/month or $19 for 3 months.** Job seekers churn when they're hired; price for a 2-4 month search instead of fighting churn.
- Later, B2B: university career centers and engineering bootcamps pay per student cohort. Same product, one buyer instead of hundreds. That's the durable revenue if consumer conversion is weak.

## Unit economics (estimates; the admin page measures the real numbers)

Model spend scales with **jobs watched**, not users, because extraction is shared.

Assumptions: 50 companies, about 15,000 open postings; about 20% have engineering-plausible titles and reach the model; about 10% of those turn over weekly; postings average about 3,000 input tokens.

| Item | Math | Per month |
|---|---|---|
| Initial backfill (one time) | 3,000 jobs x about $0.0045 (Haiku 4.5 bulk) + 15% verified x about $0.009 (Sonnet 5.5) | about $18 once |
| Ongoing bulk extraction | 300 new jobs/week x $0.0045 | about $6 |
| Ongoing verification | 15% x 300/week x $0.009 | about $2 |
| Bulk extraction on your own machine (local model) | | $0 |
| Hosting (small VPS) | | $5-20 |
| Email (transactional provider) | | $0-15 at this scale |

At a few hundred users, total cost is tens of dollars a month, so gross margin is near 100% at any reasonable price. **The constraint is distribution and trust, not cost.** Don't optimize model spend before you have users.

The Claude.ai subscription can't power a product that serves other people. Hosted tiers use the Claude API, billed per call. Running the bulk tier on your own machine with a local model is fine while you're the operator.

## Risks

- **Job board access.** Greenhouse, Lever and Ashby publish their board APIs for embedding. Workday's JSON endpoint is undocumented. Keep request rates low, identify yourself in the User-Agent, honor any objection, and get a real legal read before charging money. (Not legal advice.)
- **Coverage.** Many big employers use iCIMS, Taleo or SuccessFactors, which aren't supported yet. Check coverage of your first market before launch.
- **Extraction errors erode trust fast.** One senior role in a new grad's feed undoes the promise. That's why the verifier tier, the hide-with-reason loop and `rolescout eval` exist. Watch the "too senior" hide rate weekly.
- **Seasonality.** New grad hiring peaks in fall and spring.

## Metrics to watch weekly

- Signups, week-2 digest open rate, digest click rate
- Hides per user by reason ("too senior" and "not engineering" are product bugs)
- Applied clicks per active user (the outcome users care about)
- Verifier correction rate and cost per job (from `/admin`)
