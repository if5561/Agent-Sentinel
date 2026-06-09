from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from agent_sentinel.agents.fetch_tools import fetch_live_data_node
from agent_sentinel.agents.feedback_learning import feedback_learning_node
from agent_sentinel.agents.final_result import final_result_node
from agent_sentinel.agents.generate_plan import generate_plan_node
from agent_sentinel.agents.human_confirm import human_confirm_node
from agent_sentinel.agents.retrieve_agent import retrieve_node, should_fetch
from agent_sentinel.agents.understand_agent import understand_node
from agent_sentinel.agents.validate_plan import validate_plan_node, validation_result
from agent_sentinel.config import Settings
from agent_sentinel.feishu.card_handler import HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.state import DiagnosisState
from agent_sentinel.llm.executor import LLMExecutor
from agent_sentinel.monitoring import monitor, trace_id_from_state
from agent_sentinel.observability.langfuse import build_langfuse_prompt_service
from agent_sentinel.rag.base import BaseRetriever
from agent_sentinel.rag.factory import build_retriever
from agent_sentinel.rag.history_cases import HistoryCaseStore, build_history_case_store
from agent_sentinel.utils.config_loader import load_yaml

logger = logging.getLogger(__name__)

NodeFn = Callable[[DiagnosisState], Awaitable[DiagnosisState]]


class DiagnosisWorkflow:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMExecutor,
        sender: FeishuSender,
        decision_store: HumanDecisionStore,
        retriever: BaseRetriever | None = None,
        case_store: HistoryCaseStore | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.sender = sender
        self.decision_store = decision_store
        self.retriever = retriever or build_retriever(settings)
        self.case_store = case_store if case_store is not None else build_history_case_store(settings)
        self.checkpointer = checkpointer or InMemorySaver()
        self._compiled: Any | None = None

    def compile(self) -> Any:
        if self._compiled is not None:
            return self._compiled
        workflow_config = load_yaml(self.settings.aiops_workflow_config_path)
        graph = StateGraph(DiagnosisState)
        registry = self._node_registry()

        nodes = workflow_config.get("nodes", [])
        if not nodes:
            raise ValueError("workflow.yaml must define nodes.")
        for item in nodes:
            name = str(item["name"])
            graph.add_node(name, registry[name])

        graph.set_entry_point(str(nodes[0]["name"]))

        for edge in workflow_config.get("edges", []):
            graph.add_edge(str(edge["from"]), str(edge["to"]))

        route_registry = self._route_registry()
        for edge in workflow_config.get("conditional_edges", []):
            source = str(edge["from"])
            condition_name = str(edge["condition"])
            mapping = {
                str(key): END if value == "END" else str(value)
                for key, value in dict(edge["mapping"]).items()
            }
            graph.add_conditional_edges(source, route_registry[condition_name], mapping)

        self._compiled = graph.compile(checkpointer=self.checkpointer)
        logger.info("LangGraph workflow compiled nodes=%s", [item["name"] for item in nodes])
        return self._compiled

    async def run_streaming(self, initial_state: DiagnosisState) -> DiagnosisState:
        app = self.compile()
        workflow_thread_id = initial_state.get("workflow_thread_id", "")
        started = time.perf_counter()
        trace_id = trace_id_from_state(initial_state)
        metadata = self._langfuse_metadata(initial_state)
        config = {
            "configurable": {
                "thread_id": workflow_thread_id,
                "trace_id": trace_id,
                "workflow_thread_id": workflow_thread_id,
                "workflow_run_id": initial_state.get("workflow_run_id"),
                "chat_id": initial_state.get("chat_id"),
            },
            "recursion_limit": 20,
            "metadata": metadata,
        }
        logger.info(
            "Diagnosis LangGraph run start trace_id=%s workflow_thread_id=%s workflow_run_id=%s chat_id=%s raw_alert_keys=%s",
            trace_id,
            workflow_thread_id,
            initial_state.get("workflow_run_id"),
            initial_state.get("chat_id"),
            sorted((initial_state.get("raw_alert") or {}).keys()),
        )
        final_state: DiagnosisState = dict(initial_state)
        token = self.llm.set_trace_context(**metadata)
        try:
            with monitor.track_workflow("diagnosis", initial_state.get("chat_id")):
                async for update in app.astream(
                    initial_state,
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_update in update.items():
                        logger.info(
                            "Diagnosis LangGraph node update trace_id=%s workflow_thread_id=%s node=%s update_keys=%s",
                            trace_id,
                            workflow_thread_id,
                            node_name,
                            sorted(node_update.keys()) if isinstance(node_update, dict) else type(node_update).__name__,
                        )
                        if isinstance(node_update, dict):
                            final_state.update(node_update)
                        await self._send_progress(node_name, final_state)
                state_snapshot = await app.aget_state(config)
                final_values = getattr(state_snapshot, "values", None)
                if isinstance(final_values, dict):
                    final_state.update(final_values)
        finally:
            self.llm.reset_trace_context(token)
        logger.info(
            "Diagnosis LangGraph run completed trace_id=%s workflow_thread_id=%s final_keys=%s elapsed_ms=%s",
            trace_id,
            workflow_thread_id,
            sorted(final_state.keys()),
            int((time.perf_counter() - started) * 1000),
        )
        return final_state

    async def resume(self, workflow_thread_id: str, resume_payload: dict[str, Any]) -> DiagnosisState:
        app = self.compile()
        started = time.perf_counter()
        base_config = {
            "configurable": {"thread_id": workflow_thread_id},
            "recursion_limit": 20,
        }
        state_snapshot = await app.aget_state(base_config)
        final_state: DiagnosisState = {}
        values = getattr(state_snapshot, "values", None)
        if isinstance(values, dict):
            final_state.update(values)
        trace_id = trace_id_from_state(final_state)
        metadata = self._langfuse_metadata(final_state)
        config = {
            "configurable": {
                "thread_id": workflow_thread_id,
                "trace_id": trace_id,
                "workflow_thread_id": workflow_thread_id,
                "workflow_run_id": final_state.get("workflow_run_id"),
                "chat_id": final_state.get("chat_id"),
            },
            "recursion_limit": 20,
            "metadata": metadata,
        }
        logger.info(
            "Diagnosis LangGraph resume start trace_id=%s workflow_thread_id=%s resume_keys=%s existing_keys=%s",
            trace_id,
            workflow_thread_id,
            sorted(resume_payload.keys()),
            sorted(final_state.keys()),
        )
        token = self.llm.set_trace_context(**metadata)
        try:
            with monitor.track_workflow("diagnosis_resume", final_state.get("chat_id")):
                async for update in app.astream(
                    Command(resume=resume_payload),
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_update in update.items():
                        logger.info(
                            "Diagnosis LangGraph resume node update trace_id=%s workflow_thread_id=%s node=%s update_keys=%s",
                            trace_id,
                            workflow_thread_id,
                            node_name,
                            sorted(node_update.keys()) if isinstance(node_update, dict) else type(node_update).__name__,
                        )
                        if isinstance(node_update, dict):
                            final_state.update(node_update)
                        await self._send_progress(node_name, final_state)
                state_snapshot = await app.aget_state(config)
                final_values = getattr(state_snapshot, "values", None)
                if isinstance(final_values, dict):
                    final_state.update(final_values)
        finally:
            self.llm.reset_trace_context(token)
        logger.info(
            "Diagnosis LangGraph resume completed trace_id=%s workflow_thread_id=%s final_keys=%s elapsed_ms=%s",
            trace_id,
            workflow_thread_id,
            sorted(final_state.keys()),
            int((time.perf_counter() - started) * 1000),
        )
        return final_state

    async def update_state(self, workflow_thread_id: str, state_update: dict[str, Any]) -> None:
        app = self.compile()
        await app.aupdate_state(
            {"configurable": {"thread_id": workflow_thread_id}, "recursion_limit": 20},
            state_update,
        )

    def _node_registry(self) -> dict[str, NodeFn]:
        registry = {
            "understand": partial(
                understand_node,
                llm=self.llm,
                case_store=self.case_store,
                sender=self.sender,
                decision_store=self.decision_store,
                cache_top_k=self.settings.rag_case_cache_top_k,
                cache_threshold=self.settings.rag_case_cache_threshold,
                cache_enabled=self.settings.rag_case_cache_enabled,
            ),
            "retrieve": partial(retrieve_node, retriever=self.retriever),
            "fetch_live_data": fetch_live_data_node,
            "generate_plan": partial(generate_plan_node, llm=self.llm),
            "validate": partial(validate_plan_node, llm=self.llm),
            "human_confirm": partial(
                human_confirm_node,
                sender=self.sender,
                decision_store=self.decision_store,
                timeout_seconds=self.settings.aiops_human_confirm_timeout_seconds,
                enabled=self.settings.aiops_human_confirm_enabled,
            ),
            "final_result": partial(final_result_node, sender=self.sender),
            "feedback_learning": partial(
                feedback_learning_node,
                sender=self.sender,
                decision_store=self.decision_store,
                case_store=self.case_store,
                enabled=self.settings.rag_feedback_enabled,
            ),
        }
        return {name: self._track_node(name, node) for name, node in registry.items()}

    def _track_node(self, node_name: str, node: NodeFn) -> NodeFn:
        async def wrapped(state: DiagnosisState) -> DiagnosisState:
            trace_id = trace_id_from_state(state)
            logger.info("Diagnosis node execution start trace_id=%s node=%s", trace_id, node_name)
            with monitor.track_node(node_name, state.get("chat_id")):
                result = await node(state)
            logger.info("Diagnosis node execution completed trace_id=%s node=%s", trace_id, node_name)
            return result

        return wrapped

    def _route_registry(self) -> dict[str, Callable[[DiagnosisState], str]]:
        return {
            "should_fetch": should_fetch,
            "validation_result": validation_result,
            "cache_decision": lambda state: "hit" if state.get("cache_hit", False) else "miss",
            "human_decision": lambda state: state.get("human_decision", "timeout"),
            "always": lambda state: "next",
            "end": lambda state: "end",
        }

    async def _send_progress(self, node_name: str, state: DiagnosisState) -> None:
        status = {
            "understand": "✅ 已完成告警理解（缓存查找），正在检索历史案例...",
            "retrieve": "✅ 历史案例检索完成，正在判断是否需要实时数据...",
            "fetch_live_data": "✅ 实时指标、日志、拓扑已获取，正在生成诊断方案...",
            "generate_plan": "✅ 诊断方案已生成，正在进行安全校验...",
            "validate": "✅ 方案校验完成，等待人工确认...",
            "human_confirm": "✅ 人工确认流程结束，正在发送最终结果...",
            "final_result": "✅ 最终诊断结果已发送。",
            "feedback_learning": "✅ 反馈学习流程结束。",
        }.get(node_name)
        if not status:
            return
        logger.info(
            "Sending workflow progress node=%s chat_id=%s thread_root_message_id=%s status=%s",
            node_name,
            state.get("chat_id"),
            state.get("thread_root_message_id"),
            status,
        )
        await self.sender.send_message(
            state.get("chat_id"),
            status,
            thread_root_message_id=state.get("thread_root_message_id"),
        )

    def _langfuse_metadata(self, state: DiagnosisState) -> dict[str, object]:
        trace_id = trace_id_from_state(state)
        return {
            "trace_id": trace_id,
            "business_trace_id": trace_id,
            "trace_name": "diagnosis",
            "workflow_thread_id": state.get("workflow_thread_id"),
            "workflow_run_id": state.get("workflow_run_id"),
            "chat_id": state.get("chat_id"),
            "thread_root_message_id": state.get("thread_root_message_id"),
            "app_env": self.settings.app_env,
            "workflow_type": "diagnosis",
            "tags": ["agent-sentinel", "diagnosis", "feishu", self.settings.app_env],
        }


def build_llm_executor(settings: Settings) -> LLMExecutor:
    return LLMExecutor(
        models=settings.aiops_llm_models,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.openai_temperature,
        trust_env=settings.openai_http_trust_env,
        timeout_seconds=settings.aiops_llm_timeout_seconds,
        max_retries=settings.aiops_llm_max_retries,
        mock_enabled=settings.aiops_mock_llm_enabled,
        langfuse=build_langfuse_prompt_service(settings.langfuse),
    )
