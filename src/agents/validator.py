from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.models import (
    FailureSignal,
    PipelineState,
    ValidatorOutput,
)

load_dotenv()

_llm: Optional[ChatGroq] = None
_MAX_RETRIES = 2
_PASS_THRESHOLD = 0.7
_COMPARABILITY_CLIFF = 0.75  # top match similarity above this -> comparability=1.0


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm


class _SpecificityJudgment(BaseModel):
    score: float   # 0.0–1.0
    reason: str


def _judge_specificity(pain_framing: str, signals: List[FailureSignal]) -> float:
    """LLM-as-judge: does the pain_framing cite concrete evidence from the signals?

    Scoring guide:
    1.0 = directly quotes a specific numeric/measurable evidence value
    0.7 = references the signal type with approximate evidence
    0.4 = mentions a general problem area without evidence
    0.0 = completely generic template language
    """
    signal_evidence = "\n".join(
        f"- {s.signal_name}: {s.evidence_value}" for s in signals
    )
    prompt = (
        f"Evaluate the specificity of this pitch's pain_framing section.\n\n"
        f"Pain framing text:\n\"{pain_framing}\"\n\n"
        f"Available signals with evidence values:\n{signal_evidence}\n\n"
        "Scoring guide:\n"
        "1.0 = Pain framing directly quotes a specific numeric/measurable evidence value "
        "(e.g. 'your PageSpeed score is 28' or 'your last post was 8 months ago')\n"
        "0.7 = References the signal type but not the exact evidence value\n"
        "0.4 = Mentions a general problem without citing any evidence\n"
        "0.0 = Completely generic — could apply to any business\n\n"
        "Return a score (0.0–1.0) and a brief reason."
    )
    try:
        structured = _get_llm().with_structured_output(_SpecificityJudgment)
        result: _SpecificityJudgment = structured.invoke([HumanMessage(content=prompt)])
        return max(0.0, min(1.0, result.score))
    except Exception:
        # Fallback: rule-based heuristic — check if any evidence_value appears in the text
        pain_lower = pain_framing.lower()
        for s in signals:
            # Check for numeric patterns or key words from evidence_value
            evidence_words = s.evidence_value.lower().split()
            matches = sum(1 for w in evidence_words if len(w) > 3 and w in pain_lower)
            if matches >= 2:
                return 0.8
        return 0.4


def _build_failure_reason(
    specificity: float,
    alignment: float,
    comparability: float,
    composite: float,
) -> str:
    if specificity < 0.6:
        return (
            f"Specificity too low ({specificity:.2f}): pain_framing does not cite "
            "concrete evidence values from signals. Include exact numbers."
        )
    if alignment < 0.5:
        return (
            f"Alignment too low ({alignment:.2f}): no closely matching case study found. "
            "Try different signals or check KB ingestion."
        )
    if comparability < 0.5:
        return (
            f"Comparability too low ({comparability:.2f}): top case study match is too dissimilar. "
            "Consider re-querying with more specific signal language."
        )
    return (
        f"Composite score {composite:.2f} below threshold {_PASS_THRESHOLD}. "
        f"Specificity={specificity:.2f}, Alignment={alignment:.2f}, Comparability={comparability:.2f}. "
        "Improve specificity of pain framing and try to find stronger case study matches."
    )


def run_validator(state: PipelineState) -> dict:
    """LangGraph node: composite quality gate — score the pitch and decide pass/fail."""
    pitch = state["writer_output"].pitch
    signals = state["researcher_output"].signals
    ao = state["analyst_output"]

    # Guard: if no case study matches, force a low alignment score
    if not ao.top_matches:
        return {
            "validator_output": ValidatorOutput(
                specificity_score=0.0,
                alignment_score=0.0,
                comparability_score=0.0,
                composite_score=0.0,
                passed=False,
                failure_reason="No case study matches returned by analyst. Check ChromaDB ingestion.",
            ),
            "retry_count": state["retry_count"] + 1,
            "failure_reason": "No case study matches returned by analyst.",
        }

    top_match = ao.top_matches[0]

    # --- Specificity (40%) — LLM judge on pain_framing ---
    specificity = _judge_specificity(pitch.pain_framing, signals)

    # --- Alignment (40%) — direct from retrieval similarity score ---
    alignment = top_match.similarity_score

    # --- Comparability (20%) — cliff at 0.75 similarity ---
    if top_match.similarity_score >= _COMPARABILITY_CLIFF:
        comparability = 1.0
    else:
        comparability = top_match.similarity_score / _COMPARABILITY_CLIFF

    # --- Composite ---
    composite = (specificity * 0.4) + (alignment * 0.4) + (comparability * 0.2)
    composite = round(composite, 4)
    passed = composite >= _PASS_THRESHOLD

    # --- Max retry guard: force pass to prevent infinite loop ---
    warning_flag = False
    if not passed and state["retry_count"] >= _MAX_RETRIES:
        passed = True
        warning_flag = True

    failure_reason: Optional[str] = None
    if not passed:
        failure_reason = _build_failure_reason(specificity, alignment, comparability, composite)

    return {
        "validator_output": ValidatorOutput(
            specificity_score=round(specificity, 4),
            alignment_score=round(alignment, 4),
            comparability_score=round(comparability, 4),
            composite_score=composite,
            passed=passed,
            failure_reason=failure_reason,
            warning_flag=warning_flag,
        ),
        "retry_count": state["retry_count"] + (0 if passed else 1),
        "failure_reason": failure_reason,
    }
