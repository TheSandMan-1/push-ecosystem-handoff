# Launch kit: the 2-week validation test

Goal: 50 signups, 40% week-2 digest opens, 5 paid pre-orders. See BUSINESS.md for the kill criteria.

## Before you post (one evening)

1. Deploy (`docker compose up -d` on any small VPS, or Render/Fly/Railway) with real SMTP and `ROLESCOUT_BASE_URL` set to your domain.
2. `rolescout seed && rolescout check-sources`, fix failures, add 20-40 more SoCal companies.
3. `rolescout run`, then open `/app?tab=filtered` and screenshot 3 good "we filtered this out" examples: a senior role, a hotel "engineer", and an "entry level" posting that asks for 5+ years. These screenshots are the whole pitch.
4. Make your own account and use the product for 2 days. If you see a single senior role in your feed, fix that first.

## Where to post

- r/EngineeringStudents, r/MechanicalEngineering, r/biomedicalengineering, r/engineeringresumes (read each sub's self-promo rules first)
- Your university's engineering alumni groups (LinkedIn, Discord, Slack)
- Your own LinkedIn
- Career services contacts at 2-3 SoCal engineering schools (also your future B2B buyer)

## Post drafts

### Reddit / Discord

**A (calm, professional)**

> I got tired of "entry level" engineering searches full of senior roles, so I built a filter
>
> I'm an early-career engineer in Southern California. Every job board I tried showed me senior roles, hotel "maintenance engineer" jobs, and postings that closed weeks ago.
>
> So I built RoleScout. It reads postings directly from company career sites and keeps only the roles that are actually entry level and actually engineering. It also tells you why it hid everything else (screenshot: it caught an "Entry Level Design Engineer" posting that requires 5+ years).
>
> It's free while I test it. Right now it covers [N] SoCal companies, mostly medical devices and aerospace. If you're job hunting in mechanical, biomedical, electrical or manufacturing, I'd appreciate you trying it and telling me what it got wrong: [link]

**B (shorter, direct)**

> Built a job feed that only shows real entry-level engineering roles in SoCal. No senior roles, no hotel "engineer" jobs, no closed postings, and it shows you why it hid each one. Free while I test it. Tell me what it gets wrong: [link]

### LinkedIn

**A**

> As an early-career engineer, I kept getting "entry level" recommendations for senior roles and maintenance jobs at hotels.
>
> So I built a tool that fixes it for new grads. RoleScout reads postings straight from company career sites, pulls out what each one actually requires (years, level, clearance, real engineering or not), and keeps only the roles that fit. Every hidden posting comes with a reason.
>
> It's free and early. If you know someone graduating in engineering who's job hunting in Southern California, I'd be grateful if you sent it their way: [link]

**B**

> Built a job feed for new engineering grads: only real, open, entry-level roles, with the reason every other posting was filtered out. SoCal for now, free while I test. [link]

### Email to a career services office

**A**

> Subject: Free tool for your engineering job seekers
>
> Hi [Name],
>
> I'm a [school] engineering alum building RoleScout, a job feed for new engineering grads that keeps only roles that are actually entry level and open. It reads postings directly from company career sites and filters out senior roles, non-engineering "engineer" jobs, and closed listings.
>
> It's free during early access. Would you be open to sharing it with this year's graduating engineers, or a 15-minute call so I can hear what your students struggle with most? I'd value your input either way.
>
> [Your name]

**B**

> Subject: Entry-level engineering job feed for your students
>
> Hi [Name], I built a free job feed that shows new engineering grads only real, open, entry-level roles in SoCal. Happy to set it up for your students. Worth a 15-minute call? [Your name]

## The pre-order ask (day 10-14)

> RoleScout will move to $9/month (or $19 for 3 months) when early access ends. If it's saved you time, you can lock in the 3-month pass at $12 now: [payment link]. If it hasn't been worth paying for, I'd really like to know why.

Use a Stripe Payment Link; no billing code is needed for the test.
