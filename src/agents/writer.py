from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import (
    AnalystOutput,
    Pitch,
    PipelineState,
    ResearcherOutput,
    WriterOutput,
)

load_dotenv()

_llm: Optional[ChatGroq] = None

WRITER_SYSTEM_PROMPT = """\
You are an expert cold outreach copywriter for a full-service digital marketing agency. \
The agency handles everything web-related: SEO, PageSpeed, website rebuilds, Google review management, \
social media, content, paid ads, retargeting, CRO, and technical fixes. \
You write pitches that are specific, evidence-based, and never generic.

RULES — follow every one:
1. subject_line: Lead with the most damaging signal found (< 9 words). Reference it specifically. Never say "your website" or "your business" generically.

2. opening_hook: Open with ONE most critical signal using its EXACT evidence value (e.g., "your mobile PageSpeed score is 28/100"). Do not paraphrase — quote the data directly.

3. pain_framing: This is the most important section. Do ALL of the following:
   - Explain why the opening signal is costing them customers and revenue specifically
   - Then briefly address EVERY other detected signal — what it means and why it hurts them
   - Make clear these issues compound each other (e.g., slow site + no retargeting = traffic lost twice)
   - 4-6 sentences. No buzzwords. Cite evidence values where available.
   - The agency can fix ALL of these — say so confidently.

4. proof_paragraph: Use the matched case study as proof the agency delivers results. Write a confident 1-2 sentence result statement (e.g., "We took a Colombo dental clinic from a 2.9 Google rating to 4.6 in 90 days"). NEVER hedge, NEVER mention match quality or similarity. Just state the result.

5. cta: One soft ask — e.g., "Would a 15-minute call this week make sense to walk through what we found?" Never hard-sell.

Tone: Direct, confident, zero fluff. Cite numbers. No filler sentences.
NEVER mention match quality, similarity scores, or case study relevance. Just write the pitch.
"""


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    return _llm


def _build_writer_prompt(ro: ResearcherOutput, ao: AnalystOutput) -> str:
    business = ro.business_name or ro.business_url
    industry = ro.industry_guess or "unknown industry"

    # Format signals — lead with Tier 1
    signals_text = "\n".join(
        f"  [{s.tier.value.upper()}] {s.signal_name}: {s.evidence_value} "
        f"(confidence: {s.confidence:.0%}, source: {s.source_url})"
        for s in sorted(ro.signals, key=lambda x: x.tier.value)
    )
    if not signals_text:
        signals_text = "  No signals detected — write a general discovery pitch."

    # Format case study matches
    matches_text = ""
    for i, m in enumerate(ao.top_matches, 1):
        matches_text += (
            f"\n  Case Study #{i}:\n"
            f"  {m.case_study_text[:600]}\n"
            f"  Why it fits: {m.match_rationale}\n"
        )
    if not matches_text:
        matches_text = "  No case study match available."

    return (
        f"Write a cold outreach pitch for this prospect.\n\n"
        f"Business: {business}\n"
        f"Industry: {industry}\n"
        f"URL: {ro.business_url}\n\n"
        f"DETECTED FAILURE SIGNALS (address ALL of these in pain_framing):\n{signals_text}\n\n"
        f"The agency can fix every single one of these issues — SEO, PageSpeed, reviews, social, "
        f"ads, retargeting, content, website rebuilds — everything web-related.\n\n"
        f"PROOF OF RESULTS (use for proof_paragraph):\n{matches_text}\n\n"
        "Generate the pitch as structured JSON with fields: "
        "subject_line, opening_hook, pain_framing, proof_paragraph, cta"
    )


def run_writer(state: PipelineState) -> dict:
    """LangGraph node: generate a structured pitch from signals and case study matches."""
    ro = state["researcher_output"]
    ao = state["analyst_output"]

    structured_llm = _get_llm().with_structured_output(Pitch)
    prompt = _build_writer_prompt(ro, ao)

    for attempt in range(3):
        try:
            pitch: Pitch = structured_llm.invoke([
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(10)
            else:
                raise

    return {"writer_output": WriterOutput(pitch=pitch)}
