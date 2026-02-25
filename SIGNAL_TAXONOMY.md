# Signal Taxonomy (v1)

Lens classifies prospect failure signals into 3 tiers.
Tier controls: 
    (1) how deterministic the signal is
    (2) how “pitchable” it is
    (3) scoring weight.

Definitions:
- **Deterministic** = you can pull a measurable number or binary truth from a reliable source.
- **Scrapable** = we can fetch it via API / HTML / SERP tooling without needing private analytics.

---

## Tier 1 — Visibility Failures (hard signals · always included · high weight)

1) **PageSpeed mobile performance < 40**
   - Source: Google PageSpeed Insights API
   - Scrapable: ✅ (API)
   - Evidence: Lighthouse score + core metrics

2) **Missing / broken indexability**
   - Examples: robots.txt blocks site, meta noindex on key pages, sitemap missing
   - Source: direct fetch of /robots.txt, sitemap.xml, page HTML
   - Scrapable: ✅ (HTTP + HTML)

3) **Google Business Profile review neglect**
   - Heuristic: many low-star reviews + no owner responses (or responses older than X days)
   - Source: Tavily + SERP snippets
   - Scrapable: ✅ (via Tavily)

4) **Brand SERP weakness**
   - Heuristic: branded query doesn’t show official site / shows spam / wrong properties
   - Source: Tavily search results
   - Scrapable: ✅

5) **SSL / security trust issues**
   - Examples: no HTTPS redirect, cert errors, mixed content warnings (basic)
   - Source: HTTP checks + headers
   - Scrapable: ✅

---

## Tier 2 — Content & Social Neglect (medium signals · medium weight)

1) **Blog/news inactive > 6 months**
   - Source: site crawl + date extraction
   - Scrapable: ✅

2) **Last social post > 60 days**
   - Source: Tavily + platform page scraping
   - Scrapable: ✅

3) **Thin / outdated core pages**
   - Heuristic: services pages < N words, no FAQs, no proof blocks
   - Source: page HTML → text extract
   - Scrapable: ✅

4) **No “proof” assets**
   - Missing: testimonials / case studies / client logos
   - Source: page content analysis
   - Scrapable: ✅

5) **No video presence while competitors have it**
   - Source: Tavily + YouTube presence checks
   - Scrapable: ✅

---

## Tier 3 — Paid & Conversion Gaps (soft signals · low weight, supporting evidence)

1) **No clear CTA above the fold**
   - Source: homepage HTML analysis (buttons/links patterns)
   - Scrapable: ✅ (approx)

2) **Broken contact funnel**
   - Examples: contact form 4xx/5xx, missing mailto/phone, dead booking link
   - Source: link checking
   - Scrapable: ✅

3) **No retargeting pixel detected**
   - Examples: Meta pixel / Google Ads tag not present
   - Source: HTML script detection
   - Scrapable: ✅

4) **Ad presence asymmetry (competitors run ads, prospect doesn’t)**
   - Source: Tavily (soft inference)
   - Scrapable: ✅ (but noisy)

5) **Pricing / offer clarity gap**
   - Heuristic: no packages, no starting price, unclear next step
   - Source: page content analysis
   - Scrapable: ✅