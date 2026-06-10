from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from langgraph.types import interrupt

from agent_sentinel.agents.timeouts import llm_timeout_seconds
from agent_sentinel.feishu.card_handler import DecisionContext, HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.llm.executor import LLMExecutor
from agent_sentinel.monitoring import monitor, trace_id_from_state
from agent_sentinel.rag.history_cases import HistoryCaseStore, build_alert_text, case_to_dict, final_plan_from_case
from agent_sentinel.utils.config_loader import format_prompt
from agent_sentinel.utils.retry_utils import async_retry

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@async_retry(max_attempts=2)
# 定义 understand_node 相关的处理逻辑，供流程或外部调用复用。
async def understand_node(
    # 执行当前业务步骤，推动流程继续向下游推进。
    state: DiagnosisState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    llm: LLMExecutor,
    # 将 case_store 的值保存下来，供后续流程判断或组装响应时使用。
    case_store: HistoryCaseStore | None = None,
    # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
    sender: FeishuSender | None = None,
    # 将 decision_store 的值保存下来，供后续流程判断或组装响应时使用。
    decision_store: HumanDecisionStore | None = None,
    # 将 cache_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    cache_top_k: int = 2,
    # 将 cache_threshold 的值保存下来，供后续流程判断或组装响应时使用。
    cache_threshold: float = 0.85,
    # 将 cache_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    cache_enabled: bool = True,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> DiagnosisState:
    # 诊断第一步，先让模型把原始告警翻译成简洁摘要，再尝试匹配可复用的历史案例。
    logger.info("Node understand started")
    # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_variables = {"raw_alert": state.get("raw_alert", {})}
    # understand 节点先把原始告警压缩成摘要，后续检索和方案生成都围绕这个摘要展开。
    prompt = format_prompt("understand", prompt_variables)
    # 进入异步上下文管理区域，确保异步资源按约定释放。
    async with asyncio.timeout(llm_timeout_seconds(llm)):
        # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
        summary = await _call_llm(
            # 执行当前业务步骤，推动流程继续向下游推进。
            llm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            prompt,
            # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_name="understand",
            # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_variables=prompt_variables,
            # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
            metadata={"node_name": "understand", "trace_id": trace_id_from_state(state)},
        )
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Node understand completed summary_chars=%s", len(summary))
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info(
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "Understand node alert summary trace_id=%s chat_id=%s workflow_thread_id=%s summary_chars=%s summary=%s",
        # 调用 trace_id_from_state 完成当前步骤需要的业务处理。
        trace_id_from_state(state),
        # 调用 state.get 完成当前步骤需要的业务处理。
        state.get("chat_id") or "unknown",
        # 调用 state.get 完成当前步骤需要的业务处理。
        state.get("workflow_thread_id") or "unknown",
        # 调用 len 完成当前步骤需要的业务处理。
        len(summary),
        # 调用 _truncate_log_text 完成当前步骤需要的业务处理。
        _truncate_log_text(summary),
    )
    # 将 base_update 的值保存下来，供后续流程判断或组装响应时使用。
    base_update: DiagnosisState = {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "alert_summary": summary,
        # 调用 append_message 完成当前步骤需要的业务处理。
        "messages": append_message(state, "assistant", f"告警理解完成: {summary}"),
        # 调用 append_evidence 完成当前步骤需要的业务处理。
        "evidence": append_evidence(state, "LLM generated initial alert summary."),
    }

    # 根据 not cache_enabled or not case_store 判断当前流程该进入哪个处理分支。
    if not cache_enabled or not case_store:
        # 历史案例缓存是可选优化；缺少 Milvus 或未启用时不能阻塞主诊断链路。
        monitor.record_cache_miss(state.get("chat_id"))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return base_update

    # 将 alert_text 的值保存下来，供后续流程判断或组装响应时使用。
    alert_text = build_alert_text(state.get("raw_alert", {}))
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 先按相似度阈值召回历史成功案例，命中后可以减少重复诊断成本。
        candidates = await case_store.search_similar_cases(
            # 执行当前业务步骤，推动流程继续向下游推进。
            alert_text,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=cache_top_k,
            # 将 threshold 的值保存下来，供后续流程判断或组装响应时使用。
            threshold=cache_threshold,
        )
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.exception("History case cache lookup failed; continuing workflow")
        # 调用 monitor.record_error 完成当前步骤需要的业务处理。
        monitor.record_error("milvus_error")
        # 调用 monitor.record_cache_miss 完成当前步骤需要的业务处理。
        monitor.record_cache_miss(state.get("chat_id"))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            **base_update,
            # 调用 append_evidence 完成当前步骤需要的业务处理。
            "evidence": append_evidence(
                # 执行当前业务步骤，推动流程继续向下游推进。
                {**state, **base_update},
                # 执行当前业务步骤，推动流程继续向下游推进。
                "History case cache lookup failed; skipped cache reuse.",
            ),
        }

    # 根据 not candidates 判断当前流程该进入哪个处理分支。
    if not candidates:
        # 调用 monitor.record_cache_miss 完成当前步骤需要的业务处理。
        monitor.record_cache_miss(state.get("chat_id"))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            **base_update,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "cache_candidates": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "cache_hit": False,
            # 调用 append_evidence 完成当前步骤需要的业务处理。
            "evidence": append_evidence({**state, **base_update}, "No reusable history case matched cache threshold."),
        }

    # 将 candidate_dicts 的值保存下来，供后续流程判断或组装响应时使用。
    candidate_dicts = [case_to_dict(candidate) for candidate in candidates]
    # 根据 not state.get("chat_id") or not sender or not decisi... 判断当前流程该进入哪个处理分支。
    if not state.get("chat_id") or not sender or not decision_store:
        # 没有交互通道时不自动采用历史案例，避免误把相似告警当成完全相同故障。
        logger.info("History cases found but Feishu cache decision unavailable; continuing workflow")
        # 调用 monitor.record_cache_miss 完成当前步骤需要的业务处理。
        monitor.record_cache_miss(state.get("chat_id"))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            **base_update,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "cache_candidates": candidate_dicts,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "cache_hit": False,
            # 调用 append_evidence 完成当前步骤需要的业务处理。
            "evidence": append_evidence(
                # 执行当前业务步骤，推动流程继续向下游推进。
                {**state, **base_update},
                # 调用 len 完成当前步骤需要的业务处理。
                f"Found {len(candidate_dicts)} similar history cases; cache reuse requires Feishu confirmation.",
            ),
        }

    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, candidate in enumerate(candidates[:cache_top_k]):
        # 历史案例复用需要人工确认；用户拒绝当前候选后继续看下一个候选。
        decision = await _ask_case_cache_decision(
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state={**state, **base_update},
            # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
            sender=sender,
            # 将 decision_store 的值保存下来，供后续流程判断或组装响应时使用。
            decision_store=decision_store,
            # 将 case 的值保存下来，供后续流程判断或组装响应时使用。
            case=candidate_dicts[index],
            # 将 candidate_index 的值保存下来，供后续流程判断或组装响应时使用。
            candidate_index=index,
        )
        # 根据 decision == "adopt" 判断当前流程该进入哪个处理分支。
        if decision == "adopt":
            # 将 plan 的值保存下来，供后续流程判断或组装响应时使用。
            plan = final_plan_from_case(candidate)
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("History case cache adopted index=%s score=%s", index, candidate.score)
            # 调用 monitor.record_cache_hit 完成当前步骤需要的业务处理。
            monitor.record_cache_hit(state.get("chat_id"))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 执行当前业务步骤，推动流程继续向下游推进。
                **base_update,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_candidates": candidate_dicts,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_candidate_index": index,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_decision": "adopt",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_hit": True,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_selected_case": candidate_dicts[index],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "recommended_plan": plan,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "validation_result": True,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "need_human": False,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "human_decision": "cache_approved",
                # 调用 append_message 完成当前步骤需要的业务处理。
                "messages": append_message(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    {**state, **base_update},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "assistant",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    f"采用历史相似案例 Top {index + 1}，跳过后续诊断节点。",
                ),
                # 调用 append_evidence 完成当前步骤需要的业务处理。
                "evidence": append_evidence(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    {**state, **base_update},
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    f"History case cache adopted id={candidate.id} score={candidate.score:.3f}.",
                ),
            }
        # 根据 decision == "reject" 判断当前流程该进入哪个处理分支。
        if decision == "reject":
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("History case cache rejected index=%s", index)
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue

    # 调用 monitor.record_cache_miss 完成当前步骤需要的业务处理。
    monitor.record_cache_miss(state.get("chat_id"))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        **base_update,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "cache_candidates": candidate_dicts,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "cache_decision": "reject",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "cache_hit": False,
        # 调用 append_evidence 完成当前步骤需要的业务处理。
        "evidence": append_evidence(
            # 执行当前业务步骤，推动流程继续向下游推进。
            {**state, **base_update},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "All similar history cases were rejected; continuing full workflow.",
        ),
    }


# 定义 _ask_case_cache_decision 相关的处理逻辑，供流程或外部调用复用。
async def _ask_case_cache_decision(
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    state: DiagnosisState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    sender: FeishuSender,
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_store: HumanDecisionStore,
    # 执行当前业务步骤，推动流程继续向下游推进。
    case: dict[str, Any],
    # 执行当前业务步骤，推动流程继续向下游推进。
    candidate_index: int,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> str:
    # 向飞书发送历史案例复用确认卡片，并等待用户选择“采用”或“拒绝”。
    workflow_thread_id = state.get("workflow_thread_id", "")
    # 将 workflow_run_id 的值保存下来，供后续流程判断或组装响应时使用。
    workflow_run_id = state.get("workflow_run_id", "")
    # decision_id 与 workflow_thread_id 绑定，回调时才能恢复到对应的 interrupt 位置。
    decision_id = f"{workflow_thread_id}:case-cache:{candidate_index}" if workflow_thread_id else str(uuid.uuid4())
    # 将 context 的值保存下来，供后续流程判断或组装响应时使用。
    context = await decision_store.get_decision_context(decision_id)
    # 根据 context and context.status in {"adopt", "reject"} 判断当前流程该进入哪个处理分支。
    if context and context.status in {"adopt", "reject"}:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return context.status

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
    await sender.send_case_cache_card(
        # 调用 state.get 完成当前步骤需要的业务处理。
        state.get("chat_id"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        decision_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_thread_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_run_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        case,
        # 将 candidate_index 的值保存下来，供后续流程判断或组装响应时使用。
        candidate_index=candidate_index,
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
            # 执行当前业务步骤，推动流程继续向下游推进。
            "candidate_index": candidate_index,
        }
    )
    # interrupt 返回值来自后续卡片回调；异常或未知动作统一当作拒绝处理，保证流程继续。
    decision = str((resume_payload or {}).get("decision") or "reject")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return decision if decision in {"adopt", "reject"} else "reject"


# 定义 _truncate_log_text 相关的处理逻辑，供流程或外部调用复用。
def _truncate_log_text(text: str, limit: int = 1000) -> str:
    # 截断过长日志内容，避免一条告警摘要把日志刷得难以阅读。
    if len(text) <= limit:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return text
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return f"{text[:limit]}..."


# 定义 _call_llm 相关的处理逻辑，供流程或外部调用复用。
async def _call_llm(llm: LLMExecutor, prompt: str, **kwargs: object) -> str:
    # 调用大模型生成文本，并兼容测试替身不支持额外参数的情况。
    try:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await llm.call(prompt, **kwargs)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except TypeError as exc:
        # 根据 "unexpected keyword" not in str(exc) 判断当前流程该进入哪个处理分支。
        if "unexpected keyword" not in str(exc):
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await llm.call(prompt)  # type: ignore[call-arg]
