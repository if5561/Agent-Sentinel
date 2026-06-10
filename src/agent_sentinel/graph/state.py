from __future__ import annotations

from typing import Any, TypedDict


class DiagnosisState(TypedDict, total=False):
    # 定义诊断流程共享的“资料袋”，每个节点都会往这里读取或补充信息。
    raw_alert: dict[str, Any]
    alert_summary: str
    retrieved_docs: list[str]
    live_data: dict[str, Any]
    recommended_plan: dict[str, Any]
    evidence: list[str]
    validation_result: bool
    need_human: bool
    messages: list[dict[str, str]]
    chat_id: str
    thread_root_message_id: str | None
    mention_open_id: str | None
    mention_name: str | None
    workflow_thread_id: str
    workflow_run_id: str
    trace_id: str
    decision_id: str
    human_card_sent: bool
    human_feedback: str | None
    human_decision: str
    validation_attempts: int
    final_text: str
    cache_candidates: list[dict[str, Any]]
    cache_candidate_index: int
    cache_decision_id: str
    cache_decision: str
    cache_hit: bool
    cache_selected_case: dict[str, Any]
    feedback_decision_id: str
    feedback_card_sent: bool
    feedback_decision: str
    saved_case_id: str


def append_message(state: DiagnosisState, role: str, content: str) -> list[dict[str, str]]:
    # 在不修改原状态的前提下追加一条对话消息，方便 LangGraph 合并状态。
    return [*state.get("messages", []), {"role": role, "content": content}]


def append_evidence(state: DiagnosisState, *items: str) -> list[str]:
    # 在证据链末尾追加新的判断依据，空内容会被自动忽略。
    return [*state.get("evidence", []), *[item for item in items if item]]
