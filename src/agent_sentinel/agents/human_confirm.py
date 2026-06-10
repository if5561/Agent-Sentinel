from __future__ import annotations

import logging
import uuid

from langgraph.types import interrupt

from agent_sentinel.feishu.card_handler import DecisionContext, HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor

logger = logging.getLogger(__name__)


async def human_confirm_node(
    state: DiagnosisState,
    sender: FeishuSender,
    decision_store: HumanDecisionStore,
    timeout_seconds: int = 300,
    enabled: bool = True,
) -> DiagnosisState:
    # 方法说明：把模型生成的方案交给人确认；工作流会暂停，直到用户在飞书卡片上做选择。
    logger.info("Node human_confirm started enabled=%s", enabled)
    if not enabled or not state.get("need_human", True) or not state.get("chat_id"):
        logger.info("Node human_confirm auto-approved")
        monitor.record_feedback(True, state.get("chat_id"))
        return {
            "human_decision": "approved",
            "messages": append_message(state, "assistant", "无需人工确认，自动继续。"),
        }

    decision_id = state.get("decision_id") or str(uuid.uuid4())
    if not state.get("human_card_sent", False):
        await decision_store.register_pending_decision(
            DecisionContext(
                decision_id=decision_id,
                workflow_thread_id=state.get("workflow_thread_id", ""),
                workflow_run_id=state.get("workflow_run_id", ""),
                chat_id=state.get("chat_id"),
                thread_root_message_id=state.get("thread_root_message_id"),
            )
        )
        await sender.send_card(
            state.get("chat_id"),
            decision_id,
            state.get("workflow_thread_id", ""),
            state.get("workflow_run_id", ""),
            state.get("recommended_plan", {}),
            state.get("evidence", []),
            thread_root_message_id=state.get("thread_root_message_id"),
        )

    resume_payload = interrupt(
        {
            "decision_id": decision_id,
            "workflow_thread_id": state.get("workflow_thread_id", ""),
            "workflow_run_id": state.get("workflow_run_id", ""),
            "timeout_seconds": timeout_seconds,
        }
    )
    decision = str((resume_payload or {}).get("decision") or "timeout")
    feedback = str((resume_payload or {}).get("feedback") or "")
    logger.info("Node human_confirm completed decision=%s", decision)
    if decision == "approved":
        monitor.record_feedback(True, state.get("chat_id"))
    elif decision == "rejected":
        monitor.record_feedback(False, state.get("chat_id"))
    evidence = append_evidence(state, f"Human confirmation decision={decision}.")
    messages = append_message(state, "assistant", f"人工确认结果: {decision}")
    if decision == "rejected" and feedback:
        messages = [*messages, {"role": "user", "content": feedback}]
        evidence = [*evidence, f"Human rejection feedback: {feedback}"]
    return {
        "decision_id": decision_id,
        "human_card_sent": True,
        "human_decision": decision,
        "human_feedback": feedback or None,
        "messages": messages,
        "evidence": evidence,
    }
