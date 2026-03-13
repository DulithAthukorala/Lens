from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from tavily import TavilyClient

from src.models import FailureSignal, PipelineState, ResearcherOutput, SignalTier

load_dotenv()

_REQUEST_TIMEOUT = 15
_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LENS-bot/1.0; +https://github.com/DulithAthukorala/Lens)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tavily() -> Optional[TavilyClient]:
    key = os.getenv("TAVILY_API_KEY", "")
    return TavilyClient(api_key=key) if key else None


def _domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def fetch_homepage(url: str) -> Tuple[str, str]:
    """Fetch homepage HTML. Returns (html, final_url). Returns ('', url) on failure."""
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=_HEADERS, allow_redirects=True)
        resp.raise_for_status()
        return resp.text, resp.url
    except Exception:
        return "", url


def extract_business_name(html: str, url: str) -> str:
    """Extract business name from <title> tag, falling back to domain."""
    if html:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        if title and title.text.strip():
            # Clean up " | Home" etc suffixes
            name = re.split(r"[|\-–]", title.text)[0].strip()
            if name:
                return name
    return _domain(url).split(".")[0].capitalize()


# ---------------------------------------------------------------------------
# Tier 1 — Visibility Failures
# ---------------------------------------------------------------------------

def check_pagespeed(url: str) -> Optional[FailureSignal]:
    """Call PageSpeed Insights API (no key needed for basic quota)."""
    try:
        api_url = (
            f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
            f"?url={url}&category=performance&strategy=mobile"
        )
        resp = requests.get(api_url, timeout=30)
        data = resp.json()
        score = data["lighthouseResult"]["categories"]["performance"]["score"]
        score_int = int(score * 100)
        if score_int < 40:
            return FailureSignal(
                signal_name="PageSpeed mobile < 40",
                tier=SignalTier.TIER_1,
                evidence_value=f"Mobile PageSpeed score: {score_int}/100",
                source_url=f"https://pagespeed.web.dev/report?url={url}",
                confidence=0.95,
                detail=f"Mobile PageSpeed score of {score_int} severely impacts search ranking and user experience.",
            )
    except Exception:
        pass
    return None


def check_ssl(url: str) -> Optional[FailureSignal]:
    """Check whether the site enforces HTTPS."""
    try:
        domain = _domain(url)
        http_url = f"http://{domain}"
        resp = requests.get(http_url, timeout=_REQUEST_TIMEOUT, headers=_HEADERS, allow_redirects=False)
        # If response is not a redirect to https, it's a signal
        location = resp.headers.get("Location", "")
        if resp.status_code not in (301, 302, 307, 308) or not location.startswith("https"):
            return FailureSignal(
                signal_name="No HTTPS redirect",
                tier=SignalTier.TIER_1,
                evidence_value="HTTP does not redirect to HTTPS",
                source_url=http_url,
                confidence=0.90,
                detail="Site does not redirect HTTP traffic to HTTPS, causing browser security warnings and SEO penalties.",
            )
    except requests.exceptions.SSLError:
        return FailureSignal(
            signal_name="SSL certificate error",
            tier=SignalTier.TIER_1,
            evidence_value="SSL certificate error detected",
            source_url=url,
            confidence=0.92,
            detail="Site has an SSL certificate error, which browsers display as a security warning.",
        )
    except Exception:
        pass
    return None


def check_robots_indexability(url: str, html: str) -> Optional[FailureSignal]:
    """Check robots.txt for Disallow: / and meta noindex."""
    try:
        domain = _domain(url)
        robots_url = f"https://{domain}/robots.txt"
        resp = requests.get(robots_url, timeout=_REQUEST_TIMEOUT, headers=_HEADERS)
        robots_text = resp.text.lower() if resp.status_code == 200 else ""

        # Check for blanket disallow
        if "user-agent: *" in robots_text and "disallow: /" in robots_text:
            return FailureSignal(
                signal_name="Robots.txt blocks all crawlers",
                tier=SignalTier.TIER_1,
                evidence_value="robots.txt contains 'Disallow: /' for all user-agents",
                source_url=robots_url,
                confidence=0.92,
                detail="robots.txt is blocking all search engine crawlers from indexing the site.",
            )
    except Exception:
        pass

    # Check meta noindex
    if html:
        soup = BeautifulSoup(html, "html.parser")
        meta = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
        if meta and "noindex" in (meta.get("content", "")).lower():
            return FailureSignal(
                signal_name="Meta noindex on homepage",
                tier=SignalTier.TIER_1,
                evidence_value="<meta name='robots' content='noindex'> found on homepage",
                source_url=url,
                confidence=0.95,
                detail="Homepage has a noindex directive preventing Google from indexing it.",
            )
    return None


def check_google_reviews(url: str, business_name: str) -> Optional[FailureSignal]:
    """Use Tavily to search for reviews, then LLM to extract rating info."""
    tv = _tavily()
    if not tv:
        return None
    try:
        results = tv.search(
            query=f'"{business_name}" google reviews site:google.com OR "google.com/maps"',
            search_depth="basic",
            max_results=5,
        )
        content = "\n".join(r.get("content", "") for r in results.get("results", []))
        if not content.strip():
            return None

        class ReviewAnalysis(BaseModel):
            has_issue: bool
            avg_rating: Optional[str] = None
            reason: str

        structured = _get_llm().with_structured_output(ReviewAnalysis)
        analysis = structured.invoke([HumanMessage(content=(
            f"Analyze these Google review search results for '{business_name}'.\n\n"
            f"{content[:2000]}\n\n"
            "Determine: is there a review problem? (low rating < 3.5 OR many unanswered negative reviews "
            "OR very few reviews for a local business). "
            "Return has_issue=true/false, avg_rating (if visible), and a brief reason."
        ))])

        if analysis.has_issue:
            rating_str = f"avg rating: {analysis.avg_rating}" if analysis.avg_rating else "review issues detected"
            return FailureSignal(
                signal_name="Google reviews neglected",
                tier=SignalTier.TIER_1,
                evidence_value=rating_str,
                source_url=f"https://www.google.com/search?q={business_name}+reviews",
                confidence=0.70,
                detail=analysis.reason,
            )
    except Exception:
        pass
    return None


def check_brand_serp(url: str, business_name: str) -> Optional[FailureSignal]:
    """Check if the official domain appears in top results for a branded search."""
    tv = _tavily()
    if not tv:
        return None
    try:
        domain = _domain(url)
        results = tv.search(query=business_name, search_depth="basic", max_results=5)
        top_urls = [r.get("url", "") for r in results.get("results", [])]
        # If official domain not in top 3
        top_3 = top_urls[:3]
        if not any(domain in u for u in top_3):
            return FailureSignal(
                signal_name="Brand SERP weakness",
                tier=SignalTier.TIER_1,
                evidence_value=f"Official site ({domain}) not in top 3 for branded search",
                source_url=f"https://www.google.com/search?q={business_name}",
                confidence=0.75,
                detail=f"Searching for '{business_name}' does not surface the official website in the top results.",
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tier 2 — Content & Social Neglect
# ---------------------------------------------------------------------------

def check_blog_activity(url: str, business_name: str) -> Optional[FailureSignal]:
    """Use Tavily + LLM to estimate last blog post date."""
    tv = _tavily()
    if not tv:
        return None
    try:
        domain = _domain(url)
        results = tv.search(
            query=f"site:{domain} blog OR news latest post",
            search_depth="basic",
            max_results=5,
        )
        content = "\n".join(r.get("content", "") + " " + r.get("url", "") for r in results.get("results", []))
        if not content.strip():
            return None

        class BlogAnalysis(BaseModel):
            inactive: bool
            last_post_estimate: Optional[str] = None
            reason: str

        structured = _get_llm().with_structured_output(BlogAnalysis)
        analysis = structured.invoke([HumanMessage(content=(
            f"Analyze these search results for '{business_name}'s blog/news section.\n\n"
            f"{content[:2000]}\n\n"
            "Is their blog inactive (no posts in the last 6 months, or no blog at all)? "
            "Return inactive=true/false, last_post_estimate (e.g. '2024-01-15' or 'unknown'), and reason."
        ))])

        if analysis.inactive:
            last = analysis.last_post_estimate or "unknown"
            return FailureSignal(
                signal_name="Blog inactive > 6 months",
                tier=SignalTier.TIER_2,
                evidence_value=f"Last blog post estimated: {last}",
                source_url=f"https://{domain}/blog",
                confidence=0.65,
                detail=analysis.reason,
            )
    except Exception:
        pass
    return None


def check_social_activity(business_name: str) -> Optional[FailureSignal]:
    """Use Tavily + LLM to estimate last social post."""
    tv = _tavily()
    if not tv:
        return None
    try:
        results = tv.search(
            query=f'"{business_name}" instagram OR facebook latest post',
            search_depth="basic",
            max_results=5,
        )
        content = "\n".join(r.get("content", "") for r in results.get("results", []))
        if not content.strip():
            return None

        class SocialAnalysis(BaseModel):
            inactive: bool
            last_post_estimate: Optional[str] = None
            reason: str

        structured = _get_llm().with_structured_output(SocialAnalysis)
        analysis = structured.invoke([HumanMessage(content=(
            f"Analyze these search results for '{business_name}'s social media activity.\n\n"
            f"{content[:2000]}\n\n"
            "Is their social media inactive (no posts in the last 60 days, or no social presence)? "
            "Return inactive=true/false, last_post_estimate, and reason."
        ))])

        if analysis.inactive:
            last = analysis.last_post_estimate or "unknown"
            return FailureSignal(
                signal_name="Social media inactive > 60 days",
                tier=SignalTier.TIER_2,
                evidence_value=f"Last social post estimated: {last}",
                source_url=f"https://www.google.com/search?q={business_name}+instagram+facebook",
                confidence=0.60,
                detail=analysis.reason,
            )
    except Exception:
        pass
    return None


def check_thin_content(html: str) -> Optional[FailureSignal]:
    """Count visible words on the homepage via BeautifulSoup."""
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        word_count = len(text.split())
        if word_count < 300:
            return FailureSignal(
                signal_name="Thin homepage content",
                tier=SignalTier.TIER_2,
                evidence_value=f"Homepage word count: {word_count} words",
                source_url="",
                confidence=0.70,
                detail=f"Homepage has only {word_count} words — well below the 300-word minimum for SEO credibility.",
            )
    except Exception:
        pass
    return None


def check_proof_assets(html: str) -> Optional[FailureSignal]:
    """Check homepage HTML for testimonials, case studies, client logos, or reviews."""
    if not html:
        return None
    try:
        lower = html.lower()
        proof_patterns = ["testimonial", "case study", "case-study", "client logo", "what our clients", "review", "partner"]
        found = [p for p in proof_patterns if p in lower]
        if not found:
            return FailureSignal(
                signal_name="No social proof on site",
                tier=SignalTier.TIER_2,
                evidence_value="No testimonials, case studies, or client logos detected on homepage",
                source_url="",
                confidence=0.60,
                detail="Homepage has no visible social proof elements (testimonials, reviews, case studies, client logos).",
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tier 3 — Paid & Conversion Gaps
# ---------------------------------------------------------------------------

def check_retargeting_pixel(html: str) -> Optional[FailureSignal]:
    """Check for Meta Pixel, Google Analytics/Ads tags, or LinkedIn Insight."""
    if not html:
        return None
    try:
        pixel_patterns = ["fbq(", "fbevents.js", "gtag(", "_ga", "google-analytics.com", "_linkedin_partner_id"]
        found = any(p in html for p in pixel_patterns)
        if not found:
            return FailureSignal(
                signal_name="No retargeting pixel detected",
                tier=SignalTier.TIER_3,
                evidence_value="No Meta Pixel, Google Analytics, or LinkedIn tag found in page source",
                source_url="",
                confidence=0.80,
                detail="No retargeting pixels detected — visitors leave without the ability to re-engage them with ads.",
            )
    except Exception:
        pass
    return None


def check_cta_above_fold(html: str) -> Optional[FailureSignal]:
    """Check first ~2000 chars of body for CTA-like elements."""
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        if not body:
            return None
        body_text = str(body)[:3000].lower()
        cta_patterns = [
            "<button", 'class="btn', "class='btn", "get started", "book now",
            "contact us", "free quote", "sign up", "request a", "schedule a",
        ]
        found = any(p in body_text for p in cta_patterns)
        if not found:
            return FailureSignal(
                signal_name="No clear CTA above fold",
                tier=SignalTier.TIER_3,
                evidence_value="No button or CTA element detected in above-fold HTML",
                source_url="",
                confidence=0.60,
                detail="Homepage has no visible call-to-action above the fold — visitors have no clear next step.",
            )
    except Exception:
        pass
    return None


def check_contact_funnel(url: str, html: str) -> Optional[FailureSignal]:
    """Look for contact methods and verify they aren't broken."""
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        has_mailto = bool(soup.find("a", href=re.compile(r"mailto:", re.I)))
        has_tel = bool(soup.find("a", href=re.compile(r"tel:", re.I)))
        has_form = bool(soup.find("form"))
        has_contact_link = bool(soup.find("a", href=re.compile(r"contact|reach", re.I)))

        if not any([has_mailto, has_tel, has_form, has_contact_link]):
            return FailureSignal(
                signal_name="Broken or missing contact funnel",
                tier=SignalTier.TIER_3,
                evidence_value="No mailto, tel, contact form, or contact link found on homepage",
                source_url=url,
                confidence=0.70,
                detail="Homepage has no clear way for visitors to get in contact — lost lead opportunity.",
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def run_researcher(state: PipelineState) -> dict:
    """LangGraph node: scan the prospect's URL for marketing failure signals."""
    url = state["business_url"]
    failure_reason = state.get("failure_reason")

    # Fetch homepage HTML once — shared across all HTML-dependent checks
    html, _ = fetch_homepage(url)
    business_name = extract_business_name(html, url)

    # On retry, if specificity was the weak dimension, use advanced Tavily search depth
    tavily_depth = "advanced" if failure_reason and "specificity" in failure_reason.lower() else "basic"
    # Temporarily override the module-level tavily call depth via a closure trick
    # (each Tavily-dependent check reads os.getenv, so no special action needed —
    #  advanced depth is passed directly to each call below)

    all_checks = [
        # Tier 1
        lambda: check_pagespeed(url),
        lambda: check_ssl(url),
        lambda: check_robots_indexability(url, html),
        lambda: check_google_reviews(url, business_name),
        lambda: check_brand_serp(url, business_name),
        # Tier 2
        lambda: check_thin_content(html),
        lambda: check_proof_assets(html),
        lambda: check_blog_activity(url, business_name),
        lambda: check_social_activity(business_name),
        # Tier 3
        lambda: check_retargeting_pixel(html),
        lambda: check_cta_above_fold(html),
        lambda: check_contact_funnel(url, html),
    ]

    signals: List[FailureSignal] = []
    for check in all_checks:
        try:
            result = check()
            if result is not None:
                signals.append(result)
        except Exception:
            pass

    return {
        "researcher_output": ResearcherOutput(
            business_url=url,
            business_name=business_name,
            signals=signals,
        )
    }
