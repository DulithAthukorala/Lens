from __future__ import annotations

import os
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
You are an expert cold outreach copywriter for a digital marketing agency. \
You write pitches that are specific, evidence-based, and never generic.

RULES — follow every one:
1. subject_line: Must reference a specific finding about this business (< 9 words). Never use "your website", "your business", or any generic phrase.
2. opening_hook: Must cite ONE concrete signal with its EXACT evidence value (e.g., "your mobile PageSpeed score is 28/100", "your last blog post was over 8 months ago"). Do not paraphrase — quote the data.
3. pain_framing: Explain WHY this specific failure costs them business. Must include the evidence value from the signal. 2-3 sentences max. No buzzwords.
4. proof_paragraph: Cite the matched case study by industry/type and include a SPECIFIC result metric (e.g., "We took a Colombo dental clinic from a 2.9 Google rating to 4.6 in 90 days"). 2-3 sentences.
5. cta: A soft, single ask — e.g., "Would a 15-minute call this week make sense?" Never hard-sell.

Tone: Direct, confident, zero fluff. No "I hope this email finds you well." No filler. Cite numbers.
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
            f"\n  Case Study #{i} (similarity: {m.similarity_score:.2f}):\n"
            f"  {m.case_study_text[:600]}\n"
            f"  Rationale: {m.match_rationale}\n"
        )
    if not matches_text:
        matches_text = "  No case study match available."

    return (
        f"Write a cold outreach pitch for this prospect.\n\n"
        f"Business: {business}\n"
        f"Industry: {industry}\n"
        f"URL: {ro.business_url}\n\n"
        f"DETECTED FAILURE SIGNALS:\n{signals_text}\n\n"
        f"MATCHED CASE STUDIES:\n{matches_text}\n\n"
        "Generate the pitch as structured JSON with fields: "
        "subject_line, opening_hook, pain_framing, proof_paragraph, cta"
    )


def run_writer(state: PipelineState) -> dict:
    """LangGraph node: generate a structured pitch from signals and case study matches."""
    ro = state["researcher_output"]
    ao = state["analyst_output"]

    structured_llm = _get_llm().with_structured_output(Pitch)
    prompt = _build_writer_prompt(ro, ao)

    pitch: Pitch = structured_llm.invoke([
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"writer_output": WriterOutput(pitch=pitch)}
