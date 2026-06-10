from __future__ import annotations

import logging

from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_message

logger = logging.getLogger(__name__)


async def final_result_node(state: DiagnosisState, sender: FeishuSender) -> DiagnosisState:
    # 把诊断摘要、处理方案、校验结果和证据链整理成最终文本并发送给用户。
    logger.info("Node final_result started")
    plan = state.get("recommended_plan", {})
    evidence = state.get("evidence", [])
    evidence_text = "\n".join(f"- {item}" for item in evidence[:10]) or "- no evidence"
    final_text = (
        "[AIOps Diagnosis]\n"
        f"Summary: {state.get('alert_summary', '')}\n\n"
        f"Plan: {plan}\n\n"
        f"Validation: {state.get('validation_result', False)}\n"
        f"Human Decision: {state.get('human_decision', 'unknown')}\n\n"
        f"Evidence:\n{evidence_text}"
    )
    await sender.send_message(
        state.get("chat_id"),
        final_text,
        thread_root_message_id=state.get("thread_root_message_id"),
        mention_open_id=state.get("mention_open_id"),
        mention_name=state.get("mention_name"),
    )
    logger.info("Node final_result completed text_chars=%s", len(final_text))
    return {
        "final_text": final_text,
        "messages": append_message(state, "assistant", "最终诊断结果已生成。"),
    }
