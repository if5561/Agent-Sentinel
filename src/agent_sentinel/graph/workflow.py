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
        # 方法说明：保存诊断流程需要的依赖，例如模型、飞书发送器、检索器和人工确认记录。
        self.settings = settings
        self.llm = llm
        self.sender = sender
        self.decision_store = decision_store
        self.retriever = retriever or build_retriever(settings)
        self.case_store = case_store if case_store is not None else build_history_case_store(settings)
        self.checkpointer = checkpointer or InMemorySaver()
        self._compiled: Any | None = None

    def compile(self) -> Any:
        # 方法说明：把 workflow.yaml 里的节点和连线编译成可执行流程，相当于生成一张诊断流程图。
        if self._compiled is not None:
            return self._compiled
        # workflow.yaml 负责声明节点和边，代码里的 registry 只提供可执行函数映射。
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
            # 配置文件用字符串 END 保持可读性，编译时转换成 LangGraph 的终止节点常量。
            mapping = {
                str(key): END if value == "END" else str(value)
                for key, value in dict(edge["mapping"]).items()
            }
            graph.add_conditional_edges(source, route_registry[condition_name], mapping)

        self._compiled = graph.compile(checkpointer=self.checkpointer)
        logger.info("LangGraph workflow compiled nodes=%s", [item["name"] for item in nodes])
        return self._compiled

    async def run_streaming(self, initial_state: DiagnosisState) -> DiagnosisState:
        # 方法说明：从第一步开始运行诊断流程，并边执行边合并每个节点产出的中间结果。
        app = self.compile()
        workflow_thread_id = initial_state.get("workflow_thread_id", "")
        started = time.perf_counter()
        trace_id = trace_id_from_state(initial_state)
        metadata = self._langfuse_metadata(initial_state)
        # thread_id 是 LangGraph checkpointer 的会话键；metadata 用于日志和观测平台串联同一次诊断。
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
        # LLM 调用通过 contextvars 读取 trace 上下文，确保节点内部多次调用也归属同一链路。
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
                            # streaming 返回的是节点增量，合并后形成 API 层可直接返回的最终状态视图。
                            final_state.update(node_update)
                        await self._send_progress(node_name, final_state)
                state_snapshot = await app.aget_state(config)
                final_values = getattr(state_snapshot, "values", None)
                if isinstance(final_values, dict):
                    # 用 checkpointer 中的最终状态兜底，覆盖 streaming 中可能未显式返回的字段。
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
        # 方法说明：当人工点击飞书卡片后，从之前暂停的位置继续执行同一条诊断流程。
        app = self.compile()
        started = time.perf_counter()
        # 恢复前先读取已挂起线程的状态，才能重建 trace、chat_id 和业务上下文。
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
        # resume 沿用同一个 thread_id，让 LangGraph 从 interrupt 停住的位置继续执行。
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
        # 方法说明：把外部补充的信息写回指定诊断线程，例如人工填写的反馈内容。
        app = self.compile()
        await app.aupdate_state(
            {"configurable": {"thread_id": workflow_thread_id}, "recursion_limit": 20},
            state_update,
        )

    def _node_registry(self) -> dict[str, NodeFn]:
        # partial 在这里注入外部依赖，让各节点函数保持“输入 state，输出 state 增量”的简单形态。
        # 方法说明：登记每个诊断步骤对应的实际函数，配置文件中的节点名会在这里找到执行逻辑。
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
        # 所有节点统一包一层监控和日志，避免每个 agent 重复写相同的埋点逻辑。
        return {name: self._track_node(name, node) for name, node in registry.items()}

    def _track_node(self, node_name: str, node: NodeFn) -> NodeFn:
        # 方法说明：给每个诊断节点外面包一层日志和监控，便于看到哪一步开始、结束或报错。
        async def wrapped(state: DiagnosisState) -> DiagnosisState:
            # 方法说明：真正执行单个节点，并把节点耗时记录到监控指标里。
            trace_id = trace_id_from_state(state)
            logger.info("Diagnosis node execution start trace_id=%s node=%s", trace_id, node_name)
            with monitor.track_node(node_name, state.get("chat_id")):
                result = await node(state)
            logger.info("Diagnosis node execution completed trace_id=%s node=%s", trace_id, node_name)
            return result

        return wrapped

    def _route_registry(self) -> dict[str, Callable[[DiagnosisState], str]]:
        # 路由函数返回值必须和 workflow.yaml 里的 conditional_edges.mapping key 保持一致。
        # 方法说明：登记分支判断规则，例如是否需要取实时数据、方案是否需要重试。
        return {
            "should_fetch": should_fetch,
            "validation_result": validation_result,
            "cache_decision": lambda state: "hit" if state.get("cache_hit", False) else "miss",
            "human_decision": lambda state: state.get("human_decision", "timeout"),
            "always": lambda state: "next",
            "end": lambda state: "end",
        }

    async def _send_progress(self, node_name: str, state: DiagnosisState) -> None:
        # 进度消息是用户体验层增强；真正的流程状态仍以 LangGraph state 为准。
        # 方法说明：每完成一个诊断节点，就往飞书会话里发送一条可读的进度提示。
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
        # 方法说明：整理写入 Langfuse 的追踪信息，让一次诊断里的多次模型调用能串起来。
        trace_id = trace_id_from_state(state)
        # 观测元数据同时携带业务 trace 和工作流线程 ID，便于从日志跳到模型调用链路。
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
    # 方法说明：根据配置创建模型执行器，内部会处理多模型 fallback、重试和观测上报。
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
