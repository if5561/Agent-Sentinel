from __future__ import annotations

import logging
import uuid

from langgraph.types import interrupt

from agent_sentinel.feishu.card_handler import DecisionContext, HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor
from agent_sentinel.rag.history_cases import HistoryCaseStore

logger = logging.getLogger(__name__)


async def feedback_learning_node(
    state: DiagnosisState,
    sender: FeishuSender,
    decision_store: HumanDecisionStore,
    case_store: HistoryCaseStore | None = None,
    enabled: bool = True,
) -> DiagnosisState:
    # 方法说明：诊断结束后询问用户结果是否有效；有效案例会写入历史库，供下次相似告警复用。
    logger.info("Node feedback_learning started enabled=%s", enabled)
    if not enabled or not case_store or not state.get("chat_id"):
        return {
            "feedback_decision": "skipped",
            "messages": append_message(state, "assistant", "反馈学习已跳过。"),
        }

    workflow_thread_id = state.get("workflow_thread_id", "")
    workflow_run_id = state.get("workflow_run_id", "")
    decision_id = state.get("feedback_decision_id") or (
        f"{workflow_thread_id}:case-feedback" if workflow_thread_id else str(uuid.uuid4())
    )

    context = await decision_store.get_decision_context(decision_id)
    if context and context.status in {"valid", "invalid"}:
        decision = context.status
    else:
        await decision_store.register_pending_decision(
            DecisionContext(
                decision_id=decision_id,
                workflow_thread_id=workflow_thread_id,
                workflow_run_id=workflow_run_id,
                chat_id=state.get("chat_id"),
                thread_root_message_id=state.get("thread_root_message_id"),
            )
        )
        await sender.send_feedback_card(
            state.get("chat_id"),
            decision_id,
            workflow_thread_id,
            workflow_run_id,
            state.get("recommended_plan", {}),
            state.get("evidence", []),
            thread_root_message_id=state.get("thread_root_message_id"),
        )
        resume_payload = interrupt(
            {
                "decision_id": decision_id,
                "workflow_thread_id": workflow_thread_id,
                "workflow_run_id": workflow_run_id,
            }
        )
        decision = str((resume_payload or {}).get("decision") or "invalid")

    if decision == "valid":
        case_id = await case_store.save_case_to_history(state)
        monitor.record_feedback(True, state.get("chat_id"))
        await sender.send_message(
            state.get("chat_id"),
            f"✅ 已存入历史案例知识库：{case_id}",
            thread_root_message_id=state.get("thread_root_message_id"),
        )
        logger.info("Feedback learning saved case_id=%s", case_id)
        return {
            "feedback_decision_id": decision_id,
            "feedback_card_sent": True,
            "feedback_decision": "valid",
            "saved_case_id": case_id,
            "messages": append_message(state, "assistant", f"有效反馈已写入历史案例库: {case_id}"),
            "evidence": append_evidence(state, f"Feedback accepted; case saved id={case_id}."),
        }

    logger.info("Feedback learning skipped by user decision=%s", decision)
    monitor.record_feedback(False, state.get("chat_id"))
    return {
        "feedback_decision_id": decision_id,
        "feedback_card_sent": True,
        "feedback_decision": "invalid",
        "messages": append_message(state, "assistant", "用户反馈无效，本次结果不存储。"),
        "evidence": append_evidence(state, "Feedback rejected; case not saved."),
    }
