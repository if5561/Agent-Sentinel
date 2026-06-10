from __future__ import annotations

import logging
import uuid

from langgraph.types import interrupt

from agent_sentinel.feishu.card_handler import DecisionContext, HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 human_confirm_node 相关的处理逻辑，供流程或外部调用复用。
async def human_confirm_node(
    # 执行当前业务步骤，推动流程继续向下游推进。
    state: DiagnosisState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    sender: FeishuSender,
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_store: HumanDecisionStore,
    # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    timeout_seconds: int = 300,
    # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
    enabled: bool = True,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> DiagnosisState:
    # 把模型生成的方案交给人确认；工作流会暂停，直到用户在飞书卡片上做选择。
    logger.info("Node human_confirm started enabled=%s", enabled)
    # 根据 not enabled or not state.get("need_human", True) or ... 判断当前流程该进入哪个处理分支。
    if not enabled or not state.get("need_human", True) or not state.get("chat_id"):
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Node human_confirm auto-approved")
        # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
        monitor.record_feedback(True, state.get("chat_id"))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "human_decision": "approved",
            # 调用 append_message 完成当前步骤需要的业务处理。
            "messages": append_message(state, "assistant", "无需人工确认，自动继续。"),
        }

    # 将 decision_id 的值保存下来，供后续流程判断或组装响应时使用。
    decision_id = state.get("decision_id") or str(uuid.uuid4())
    # 根据 not state.get("human_card_sent", False) 判断当前流程该进入哪个处理分支。
    if not state.get("human_card_sent", False):
        # 等待异步操作完成，再继续推进当前业务流程。
        await decision_store.register_pending_decision(
            # 调用 DecisionContext 完成当前步骤需要的业务处理。
            DecisionContext(
                # 将 decision_id 的值保存下来，供后续流程判断或组装响应时使用。
                decision_id=decision_id,
                # 将 workflow_thread_id 的值保存下来，供后续流程判断或组装响应时使用。
                workflow_thread_id=state.get("workflow_thread_id", ""),
                # 将 workflow_run_id 的值保存下来，供后续流程判断或组装响应时使用。
                workflow_run_id=state.get("workflow_run_id", ""),
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=state.get("chat_id"),
                # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                thread_root_message_id=state.get("thread_root_message_id"),
            )
        )
        # 等待异步操作完成，再继续推进当前业务流程。
        await sender.send_card(
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("chat_id"),
            # 执行当前业务步骤，推动流程继续向下游推进。
            decision_id,
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("workflow_thread_id", ""),
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("workflow_run_id", ""),
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("recommended_plan", {}),
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("evidence", []),
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=state.get("thread_root_message_id"),
        )

    # 将 resume_payload 的值保存下来，供后续流程判断或组装响应时使用。
    resume_payload = interrupt(
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "decision_id": decision_id,
            # 调用 state.get 完成当前步骤需要的业务处理。
            "workflow_thread_id": state.get("workflow_thread_id", ""),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "workflow_run_id": state.get("workflow_run_id", ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "timeout_seconds": timeout_seconds,
        }
    )
    # 将 decision 的值保存下来，供后续流程判断或组装响应时使用。
    decision = str((resume_payload or {}).get("decision") or "timeout")
    # 将 feedback 的值保存下来，供后续流程判断或组装响应时使用。
    feedback = str((resume_payload or {}).get("feedback") or "")
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Node human_confirm completed decision=%s", decision)
    # 根据 decision == "approved" 判断当前流程该进入哪个处理分支。
    if decision == "approved":
        # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
        monitor.record_feedback(True, state.get("chat_id"))
    # 根据 decision == "rejected" 判断当前流程该进入哪个处理分支。
    elif decision == "rejected":
        # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
        monitor.record_feedback(False, state.get("chat_id"))
    # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
    evidence = append_evidence(state, f"Human confirmation decision={decision}.")
    # 将 messages 的值保存下来，供后续流程判断或组装响应时使用。
    messages = append_message(state, "assistant", f"人工确认结果: {decision}")
    # 根据 decision == "rejected" and feedback 判断当前流程该进入哪个处理分支。
    if decision == "rejected" and feedback:
        # 将 messages 的值保存下来，供后续流程判断或组装响应时使用。
        messages = [*messages, {"role": "user", "content": feedback}]
        # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
        evidence = [*evidence, f"Human rejection feedback: {feedback}"]
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "decision_id": decision_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "human_card_sent": True,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "human_decision": decision,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "human_feedback": feedback or None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "messages": messages,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "evidence": evidence,
    }
