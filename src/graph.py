from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.analyst import run_analyst
from src.agents.researcher import run_researcher
from src.agents.validator import run_validator
from src.agents.writer import run_writer
from src.models import PipelineState


def _route_validator(state: PipelineState) -> str:
    """Conditional edge: loop back to researcher on failure, exit on pass."""
    vo = state.get("validator_output")
    if vo is None or not vo.passed:
        return "researcher"
    return "end"


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("researcher", run_researcher)
    g.add_node("analyst", run_analyst)
    g.add_node("writer", run_writer)
    g.add_node("validator", run_validator)

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", "validator")

    g.add_conditional_edges(
        "validator",
        _route_validator,
        {"end": END, "researcher": "researcher"},
    )

    return g.compile()


def run_pipeline(url: str) -> PipelineState:
    """Entry point: run the full LENS pipeline against a prospect URL."""
    if not url.startswith("http"):
        url = f"https://{url}"

    graph = build_graph()
    initial_state: PipelineState = {
        "business_url": url,
        "retry_count": 0,
        "failure_reason": None,
        "researcher_output": None,
        "analyst_output": None,
        "writer_output": None,
        "validator_output": None,
    }
    return graph.invoke(initial_state)
