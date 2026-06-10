from __future__ import annotations

from typing import Any, TypedDict


# 定义 DiagnosisState 组件，集中管理这个模块的状态和行为。
class DiagnosisState(TypedDict, total=False):
    # 定义诊断流程共享的“资料袋”，每个节点都会往这里读取或补充信息。
    raw_alert: dict[str, Any]
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_summary: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    retrieved_docs: list[str]
    # 执行当前业务步骤，推动流程继续向下游推进。
    live_data: dict[str, Any]
    # 执行当前业务步骤，推动流程继续向下游推进。
    recommended_plan: dict[str, Any]
    # 执行当前业务步骤，推动流程继续向下游推进。
    evidence: list[str]
    # 执行当前业务步骤，推动流程继续向下游推进。
    validation_result: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    need_human: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    messages: list[dict[str, str]]
    # 执行当前业务步骤，推动流程继续向下游推进。
    chat_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    thread_root_message_id: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    mention_open_id: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    mention_name: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_thread_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_run_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    trace_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    human_card_sent: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    human_feedback: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    human_decision: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    validation_attempts: int
    # 执行当前业务步骤，推动流程继续向下游推进。
    final_text: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    cache_candidates: list[dict[str, Any]]
    # 执行当前业务步骤，推动流程继续向下游推进。
    cache_candidate_index: int
    # 执行当前业务步骤，推动流程继续向下游推进。
    cache_decision_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    cache_decision: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    cache_hit: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    cache_selected_case: dict[str, Any]
    # 执行当前业务步骤，推动流程继续向下游推进。
    feedback_decision_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    feedback_card_sent: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    feedback_decision: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    saved_case_id: str


# 定义 append_message 相关的处理逻辑，供流程或外部调用复用。
def append_message(state: DiagnosisState, role: str, content: str) -> list[dict[str, str]]:
    # 在不修改原状态的前提下追加一条对话消息，方便 LangGraph 合并状态。
    return [*state.get("messages", []), {"role": role, "content": content}]


# 定义 append_evidence 相关的处理逻辑，供流程或外部调用复用。
def append_evidence(state: DiagnosisState, *items: str) -> list[str]:
    # 在证据链末尾追加新的判断依据，空内容会被自动忽略。
    return [*state.get("evidence", []), *[item for item in items if item]]
