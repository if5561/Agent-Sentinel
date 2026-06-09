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

logger = logging.getLogger(__name__)


@async_retry(max_attempts=2)
async def understand_node(
    state: DiagnosisState,
    llm: LLMExecutor,
    case_store: HistoryCaseStore | None = None,
    sender: FeishuSender | None = None,
    decision_store: HumanDecisionStore | None = None,
    cache_top_k: int = 2,
    cache_threshold: float = 0.85,
    cache_enabled: bool = True,
) -> DiagnosisState:
    logger.info("Node understand started")
    prompt_variables = {"raw_alert": state.get("raw_alert", {})}
    prompt = format_prompt("understand", prompt_variables)
    async with asyncio.timeout(llm_timeout_seconds(llm)):
        summary = await _call_llm(
            llm,
            prompt,
            prompt_name="understand",
            prompt_variables=prompt_variables,
            metadata={"node_name": "understand", "trace_id": trace_id_from_state(state)},
        )
    logger.info("Node understand completed summary_chars=%s", len(summary))
    logger.info(
        "Understand node alert summary trace_id=%s chat_id=%s workflow_thread_id=%s summary_chars=%s summary=%s",
        trace_id_from_state(state),
        state.get("chat_id") or "unknown",
        state.get("workflow_thread_id") or "unknown",
        len(summary),
        _truncate_log_text(summary),
    )
    base_update: DiagnosisState = {
        "alert_summary": summary,
        "messages": append_message(state, "assistant", f"告警理解完成: {summary}"),
        "evidence": append_evidence(state, "LLM generated initial alert summary."),
    }

    if not cache_enabled or not case_store:
        monitor.record_cache_miss(state.get("chat_id"))
        return base_update

    alert_text = build_alert_text(state.get("raw_alert", {}))
    try:
        candidates = await case_store.search_similar_cases(
            alert_text,
            top_k=cache_top_k,
            threshold=cache_threshold,
        )
    except Exception:
        logger.exception("History case cache lookup failed; continuing workflow")
        monitor.record_error("milvus_error")
        monitor.record_cache_miss(state.get("chat_id"))
        return {
            **base_update,
            "evidence": append_evidence(
                {**state, **base_update},
                "History case cache lookup failed; skipped cache reuse.",
            ),
        }

    if not candidates:
        monitor.record_cache_miss(state.get("chat_id"))
        return {
            **base_update,
            "cache_candidates": [],
            "cache_hit": False,
            "evidence": append_evidence({**state, **base_update}, "No reusable history case matched cache threshold."),
        }

    candidate_dicts = [case_to_dict(candidate) for candidate in candidates]
    if not state.get("chat_id") or not sender or not decision_store:
        logger.info("History cases found but Feishu cache decision unavailable; continuing workflow")
        monitor.record_cache_miss(state.get("chat_id"))
        return {
            **base_update,
            "cache_candidates": candidate_dicts,
            "cache_hit": False,
            "evidence": append_evidence(
                {**state, **base_update},
                f"Found {len(candidate_dicts)} similar history cases; cache reuse requires Feishu confirmation.",
            ),
        }

    for index, candidate in enumerate(candidates[:cache_top_k]):
        decision = await _ask_case_cache_decision(
            state={**state, **base_update},
            sender=sender,
            decision_store=decision_store,
            case=candidate_dicts[index],
            candidate_index=index,
        )
        if decision == "adopt":
            plan = final_plan_from_case(candidate)
            logger.info("History case cache adopted index=%s score=%s", index, candidate.score)
            monitor.record_cache_hit(state.get("chat_id"))
            return {
                **base_update,
                "cache_candidates": candidate_dicts,
                "cache_candidate_index": index,
                "cache_decision": "adopt",
                "cache_hit": True,
                "cache_selected_case": candidate_dicts[index],
                "recommended_plan": plan,
                "validation_result": True,
                "need_human": False,
                "human_decision": "cache_approved",
                "messages": append_message(
                    {**state, **base_update},
                    "assistant",
                    f"采用历史相似案例 Top {index + 1}，跳过后续诊断节点。",
                ),
                "evidence": append_evidence(
                    {**state, **base_update},
                    f"History case cache adopted id={candidate.id} score={candidate.score:.3f}.",
                ),
            }
        if decision == "reject":
            logger.info("History case cache rejected index=%s", index)
            continue

    monitor.record_cache_miss(state.get("chat_id"))
    return {
        **base_update,
        "cache_candidates": candidate_dicts,
        "cache_decision": "reject",
        "cache_hit": False,
        "evidence": append_evidence(
            {**state, **base_update},
            "All similar history cases were rejected; continuing full workflow.",
        ),
    }


async def _ask_case_cache_decision(
    *,
    state: DiagnosisState,
    sender: FeishuSender,
    decision_store: HumanDecisionStore,
    case: dict[str, Any],
    candidate_index: int,
) -> str:
    workflow_thread_id = state.get("workflow_thread_id", "")
    workflow_run_id = state.get("workflow_run_id", "")
    decision_id = f"{workflow_thread_id}:case-cache:{candidate_index}" if workflow_thread_id else str(uuid.uuid4())
    context = await decision_store.get_decision_context(decision_id)
    if context and context.status in {"adopt", "reject"}:
        return context.status

    await decision_store.register_pending_decision(
        DecisionContext(
            decision_id=decision_id,
            workflow_thread_id=workflow_thread_id,
            workflow_run_id=workflow_run_id,
            chat_id=state.get("chat_id"),
            thread_root_message_id=state.get("thread_root_message_id"),
        )
    )
    await sender.send_case_cache_card(
        state.get("chat_id"),
        decision_id,
        workflow_thread_id,
        workflow_run_id,
        case,
        candidate_index=candidate_index,
        thread_root_message_id=state.get("thread_root_message_id"),
    )
    resume_payload = interrupt(
        {
            "decision_id": decision_id,
            "workflow_thread_id": workflow_thread_id,
            "workflow_run_id": workflow_run_id,
            "candidate_index": candidate_index,
        }
    )
    decision = str((resume_payload or {}).get("decision") or "reject")
    return decision if decision in {"adopt", "reject"} else "reject"


def _truncate_log_text(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


async def _call_llm(llm: LLMExecutor, prompt: str, **kwargs: object) -> str:
    try:
        return await llm.call(prompt, **kwargs)
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        return await llm.call(prompt)  # type: ignore[call-arg]
