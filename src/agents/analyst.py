from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.models import (
    AnalystOutput,
    CaseStudyMatch,
    FailureSignal,
    PipelineState,
)
from src.rag.pipeline import build_analyst_query, retrieve_case_studies

load_dotenv()

_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm


def _generate_rationale(case_study_text: str, signals: List[FailureSignal]) -> str:
    """Ask the LLM to write a 1-2 sentence rationale for why this case study matches."""
    signal_summary = "; ".join(f"{s.signal_name} ({s.evidence_value})" for s in signals[:4])
    prompt = (
        f"A prospect has these marketing failure signals: {signal_summary}.\n\n"
        f"Here is a case study from the agency's knowledge base:\n{case_study_text[:1200]}\n\n"
        "In 1-2 sentences, explain specifically why this case study is relevant to the prospect's situation. "
        "Reference both the prospect's problem and the case study's outcome."
    )
    try:
        resp = _get_llm().invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception:
        return "This case study addresses similar marketing challenges and demonstrates a comparable solution."


def run_analyst(state: PipelineState) -> dict:
    """LangGraph node: retrieve top case studies from ChromaDB and enrich with rationale."""
    ro = state["researcher_output"]
    query = build_analyst_query(ro.signals, ro.industry_guess)
    raw_matches = retrieve_case_studies(query, n_results=5)

    if not raw_matches:
        # Return empty matches — validator will catch low alignment score
        return {
            "analyst_output": AnalystOutput(top_matches=[], query_used=query)
        }

    top_3 = raw_matches[:3]
    enriched: List[CaseStudyMatch] = []
    for match in top_3:
        rationale = _generate_rationale(match["text"], ro.signals)
        enriched.append(CaseStudyMatch(
            case_study_id=match["id"],
            case_study_text=match["text"],
            similarity_score=match["similarity"],
            match_rationale=rationale,
        ))

    return {
        "analyst_output": AnalystOutput(top_matches=enriched, query_used=query)
    }
