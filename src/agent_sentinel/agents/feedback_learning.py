from __future__ import annotations

import logging
import uuid

from langgraph.types import interrupt

from agent_sentinel.feishu.card_handler import DecisionContext, HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor
from agent_sentinel.rag.history_cases import HistoryCaseStore

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 feedback_learning_node 相关的处理逻辑，供流程或外部调用复用。
async def feedback_learning_node(
    # 执行当前业务步骤，推动流程继续向下游推进。
    state: DiagnosisState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    sender: FeishuSender,
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_store: HumanDecisionStore,
    # 将 case_store 的值保存下来，供后续流程判断或组装响应时使用。
    case_store: HistoryCaseStore | None = None,
    # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
    enabled: bool = True,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> DiagnosisState:
    # 诊断结束后询问用户结果是否有效；有效案例会写入历史库，供下次相似告警复用。
    logger.info("Node feedback_learning started enabled=%s", enabled)
    # 根据 not enabled or not case_store or not state.get("chat... 判断当前流程该进入哪个处理分支。
    if not enabled or not case_store or not state.get("chat_id"):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "feedback_decision": "skipped",
            # 调用 append_message 完成当前步骤需要的业务处理。
            "messages": append_message(state, "assistant", "反馈学习已跳过。"),
        }

    # 将 workflow_thread_id 的值保存下来，供后续流程判断或组装响应时使用。
    workflow_thread_id = state.get("workflow_thread_id", "")
    # 将 workflow_run_id 的值保存下来，供后续流程判断或组装响应时使用。
    workflow_run_id = state.get("workflow_run_id", "")
    # 将 decision_id 的值保存下来，供后续流程判断或组装响应时使用。
    decision_id = state.get("feedback_decision_id") or (
        # 调用 str 完成当前步骤需要的业务处理。
        f"{workflow_thread_id}:case-feedback" if workflow_thread_id else str(uuid.uuid4())
    )

    # 将 context 的值保存下来，供后续流程判断或组装响应时使用。
    context = await decision_store.get_decision_context(decision_id)
    # 根据 context and context.status in {"valid", "invalid"} 判断当前流程该进入哪个处理分支。
    if context and context.status in {"valid", "invalid"}:
        # 将 decision 的值保存下来，供后续流程判断或组装响应时使用。
        decision = context.status
    # 处理前面条件都不满足时的默认分支。
    else:
        # 等待异步操作完成，再继续推进当前业务流程。
        await decision_store.register_pending_decision(
            # 调用 DecisionContext 完成当前步骤需要的业务处理。
            DecisionContext(
                # 将 decision_id 的值保存下来，供后续流程判断或组装响应时使用。
                decision_id=decision_id,
                # 将 workflow_thread_id 的值保存下来，供后续流程判断或组装响应时使用。
                workflow_thread_id=workflow_thread_id,
                # 将 workflow_run_id 的值保存下来，供后续流程判断或组装响应时使用。
                workflow_run_id=workflow_run_id,
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=state.get("chat_id"),
                # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                thread_root_message_id=state.get("thread_root_message_id"),
            )
        )
        # 等待异步操作完成，再继续推进当前业务流程。
        await sender.send_feedback_card(
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("chat_id"),
            # 执行当前业务步骤，推动流程继续向下游推进。
            decision_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            workflow_thread_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            workflow_run_id,
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
                # 执行当前业务步骤，推动流程继续向下游推进。
                "workflow_thread_id": workflow_thread_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "workflow_run_id": workflow_run_id,
            }
        )
        # 将 decision 的值保存下来，供后续流程判断或组装响应时使用。
        decision = str((resume_payload or {}).get("decision") or "invalid")

    # 根据 decision == "valid" 判断当前流程该进入哪个处理分支。
    if decision == "valid":
        # 将 case_id 的值保存下来，供后续流程判断或组装响应时使用。
        case_id = await case_store.save_case_to_history(state)
        # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
        monitor.record_feedback(True, state.get("chat_id"))
        # 等待异步操作完成，再继续推进当前业务流程。
        await sender.send_message(
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("chat_id"),
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"✅ 已存入历史案例知识库：{case_id}",
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=state.get("thread_root_message_id"),
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feedback learning saved case_id=%s", case_id)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "feedback_decision_id": decision_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "feedback_card_sent": True,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "feedback_decision": "valid",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "saved_case_id": case_id,
            # 调用 append_message 完成当前步骤需要的业务处理。
            "messages": append_message(state, "assistant", f"有效反馈已写入历史案例库: {case_id}"),
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "evidence": append_evidence(state, f"Feedback accepted; case saved id={case_id}."),
        }

    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Feedback learning skipped by user decision=%s", decision)
    # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
    monitor.record_feedback(False, state.get("chat_id"))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "feedback_decision_id": decision_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "feedback_card_sent": True,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "feedback_decision": "invalid",
        # 调用 append_message 完成当前步骤需要的业务处理。
        "messages": append_message(state, "assistant", "用户反馈无效，本次结果不存储。"),
        # 调用 append_evidence 完成当前步骤需要的业务处理。
        "evidence": append_evidence(state, "Feedback rejected; case not saved."),
    }
