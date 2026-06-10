from __future__ import annotations

import logging

from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_message

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 final_result_node 相关的处理逻辑，供流程或外部调用复用。
async def final_result_node(state: DiagnosisState, sender: FeishuSender) -> DiagnosisState:
    # 把诊断摘要、处理方案、校验结果和证据链整理成最终文本并发送给用户。
    logger.info("Node final_result started")
    # 将 plan 的值保存下来，供后续流程判断或组装响应时使用。
    plan = state.get("recommended_plan", {})
    # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
    evidence = state.get("evidence", [])
    # 将 evidence_text 的值保存下来，供后续流程判断或组装响应时使用。
    evidence_text = "\n".join(f"- {item}" for item in evidence[:10]) or "- no evidence"
    # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
    final_text = (
        # 执行当前业务步骤，推动流程继续向下游推进。
        "[AIOps Diagnosis]\n"
        # 调用 state.get 完成当前步骤需要的业务处理。
        f"Summary: {state.get('alert_summary', '')}\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        f"Plan: {plan}\n\n"
        # 调用 state.get 完成当前步骤需要的业务处理。
        f"Validation: {state.get('validation_result', False)}\n"
        # 调用 state.get 完成当前步骤需要的业务处理。
        f"Human Decision: {state.get('human_decision', 'unknown')}\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        f"Evidence:\n{evidence_text}"
    )
    # 等待异步操作完成，再继续推进当前业务流程。
    await sender.send_message(
        # 调用 state.get 完成当前步骤需要的业务处理。
        state.get("chat_id"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        final_text,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id=state.get("thread_root_message_id"),
        # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
        mention_open_id=state.get("mention_open_id"),
        # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
        mention_name=state.get("mention_name"),
    )
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Node final_result completed text_chars=%s", len(final_text))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "final_text": final_text,
        # 调用 append_message 完成当前步骤需要的业务处理。
        "messages": append_message(state, "assistant", "最终诊断结果已生成。"),
    }
