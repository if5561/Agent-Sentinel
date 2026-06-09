from __future__ import annotations

from typing import TypedDict

from agent_sentinel.graph.state import DiagnosisState


class TopicFlowState(TypedDict, total=False):
    """State carried by the interactive topic LangGraph demo workflow."""

    query: str
    task_id: str
    chat_id: str
    root_message_id: str
    current_node: str
    node_result: str
    last_action: str
    retry_counts: dict[str, int]
    node_results: list[dict[str, str]]
    diagnosis_state: DiagnosisState
    tool_plan: dict[str, object]
    tool_results: list[dict[str, object]]
    evidence_review: dict[str, object]
    final_text: str


def append_node_result(
    state: TopicFlowState,
    node_name: str,
    node_result: str,
) -> list[dict[str, str]]:
    return [
        *state.get("node_results", []),
        {"node_name": node_name, "result": node_result},
    ]


def increment_retry(state: TopicFlowState, node_name: str) -> dict[str, int]:
    retry_counts = dict(state.get("retry_counts", {}))
    retry_counts[node_name] = retry_counts.get(node_name, 0) + 1
    return retry_counts
