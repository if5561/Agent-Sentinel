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
        # 保存运行配置，供流程构建和节点执行时读取。
        self.settings = settings
        # 保存大模型执行器，用于诊断分析和结果生成。
        self.llm = llm
        # 保存飞书发送器，用于推送人工确认消息。
        self.sender = sender
        # 保存人工决策存储，用于记录和查询确认结果。
        self.decision_store = decision_store
        # 优先使用外部传入的检索器，否则根据配置构建默认检索器。
        self.retriever = retriever or build_retriever(settings)
        # 优先使用外部传入的案例存储，否则根据配置构建历史案例存储。
        self.case_store = case_store if case_store is not None else build_history_case_store(settings)
        # 优先使用外部传入的检查点器，否则使用内存检查点器。
        self.checkpointer = checkpointer or InMemorySaver()
        # 缓存已编译的流程图，避免重复构建。
        self._compiled: Any | None = None

    def compile(self) -> Any:
        # 把 workflow.yaml 里的节点和连线编译成可执行流程，相当于生成一张诊断流程图。
        if self._compiled is not None:
            return self._compiled
        # workflow.yaml 负责声明节点和边，代码里的 registry 只提供可执行函数映射。
        workflow_config = load_yaml(self.settings.aiops_workflow_config_path)
        # 初始化状态图，使用 DiagnosisState 作为图的状态类型
        graph = StateGraph(DiagnosisState)
        # 获取节点名称到可执行函数的映射
        registry = self._node_registry()

        nodes = workflow_config.get("nodes", [])
        if not nodes:
            raise ValueError("workflow.yaml must define nodes.")
        # 遍历配置中的节点定义，逐一注册到状态图
        for item in nodes:
            name = str(item["name"])
            graph.add_node(name, registry[name])

        # 设置图的入口节点为配置中的第一个节点
        graph.set_entry_point(str(nodes[0]["name"]))

        # 注册普通边，表示节点之间的固定跳转
        for edge in workflow_config.get("edges", []):
            graph.add_edge(str(edge["from"]), str(edge["to"]))

        # 获取条件路由函数映射
        route_registry = self._route_registry()
        # 注册条件边，根据运行时状态决定跳转目标
        for edge in workflow_config.get("conditional_edges", []):
            source = str(edge["from"])
            condition_name = str(edge["condition"])
            # 配置文件用字符串 END 保持可读性，编译时转换成 LangGraph 的终止节点常量。
            mapping = {
                str(key): END if value == "END" else str(value)
                for key, value in dict(edge["mapping"]).items()
            }
            graph.add_conditional_edges(source, route_registry[condition_name], mapping)

        # 编译图并注入检查点持久化器，支持断点续跑和状态恢复
        self._compiled = graph.compile(checkpointer=self.checkpointer)
        logger.info("LangGraph workflow compiled nodes=%s", [item["name"] for item in nodes])
        return self._compiled

    async def run_streaming(self, initial_state: DiagnosisState) -> DiagnosisState:
        # 获取已编译的流程图，后续直接用它驱动诊断任务。
        app = self.compile()
        # 取出工作流线程 ID，作为 LangGraph checkpointer 保存和恢复状态的 key。
        workflow_thread_id = initial_state.get("workflow_thread_id", "")
        # 记录开始时间，用于最后统计本次完整诊断耗时。
        started = time.perf_counter()
        # 根据初始状态生成 trace_id，方便日志、监控和模型观测链路串联同一次请求。
        trace_id = trace_id_from_state(initial_state)
        # 整理 Langfuse 元数据，后续 LLM 调用会带上这些业务上下文。
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
        # 记录本次诊断启动日志，重点输出 trace、线程、运行 ID 和原始告警字段。
        logger.info(
            "Diagnosis LangGraph run start trace_id=%s workflow_thread_id=%s workflow_run_id=%s chat_id=%s raw_alert_keys=%s",
            trace_id,
            workflow_thread_id,
            initial_state.get("workflow_run_id"),
            initial_state.get("chat_id"),
            sorted((initial_state.get("raw_alert") or {}).keys()),
        )
        # final_state 从初始状态拷贝开始，后面每个节点返回增量时持续合并进去。
        final_state: DiagnosisState = dict(initial_state)
        # LLM 调用通过 contextvars 读取 trace 上下文，确保节点内部多次调用也归属同一链路。
        token = self.llm.set_trace_context(**metadata)
        try:
            # 用 Prometheus 监控包住整条诊断流程，统计端到端耗时和活跃工作流数量。
            with monitor.track_workflow("diagnosis", initial_state.get("chat_id")):
                # 以 updates 模式流式运行图，LangGraph 每完成一个节点就会返回该节点的状态增量。
                async for update in app.astream(
                    initial_state,
                    config=config,
                    stream_mode="updates",
                ):
                    # 一个 update 可能包含一个或多个节点结果，逐个合并并发送进度。
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
                        # 节点完成后给飞书发送一条进度消息，让用户知道流程已经走到哪一步。
                        await self._send_progress(node_name, final_state)
                # 流式执行结束后再从 checkpointer 读取一次最终状态，补齐未在 streaming 中显式返回的字段。
                state_snapshot = await app.aget_state(config)
                final_values = getattr(state_snapshot, "values", None)
                if isinstance(final_values, dict):
                    # 用 checkpointer 中的最终状态兜底，覆盖 streaming 中可能未显式返回的字段。
                    final_state.update(final_values)
        finally:
            # 无论流程成功还是失败，都恢复 LLM trace 上下文，避免污染下一次请求。
            self.llm.reset_trace_context(token)
        # 输出完成日志，包含最终状态字段和总耗时，方便排查流程是否完整走完。
        logger.info(
            "Diagnosis LangGraph run completed trace_id=%s workflow_thread_id=%s final_keys=%s elapsed_ms=%s",
            trace_id,
            workflow_thread_id,
            sorted(final_state.keys()),
            int((time.perf_counter() - started) * 1000),
        )
        return final_state

    async def resume(self, workflow_thread_id: str, resume_payload: dict[str, Any]) -> DiagnosisState:
        # 获取已编译的流程图，恢复执行时必须使用同一套节点和边定义。
        app = self.compile()
        # 记录恢复开始时间，用于统计本次 resume 耗时。
        started = time.perf_counter()
        # 恢复前先读取已挂起线程的状态，才能重建 trace、chat_id 和业务上下文。
        base_config = {
            "configurable": {"thread_id": workflow_thread_id},
            "recursion_limit": 20,
        }
        # 根据 thread_id 从 checkpointer 读取暂停前的状态快照。
        state_snapshot = await app.aget_state(base_config)
        # final_state 用于承接旧状态和恢复后每个节点返回的新状态。
        final_state: DiagnosisState = {}
        values = getattr(state_snapshot, "values", None)
        if isinstance(values, dict):
            # 如果快照里有状态值，先恢复到 final_state，后续再在此基础上合并增量。
            final_state.update(values)
        # 从已恢复状态里重新生成 trace_id，保证 resume 前后的日志和观测仍属于同一条链路。
        trace_id = trace_id_from_state(final_state)
        # 基于旧状态整理观测元数据，让恢复后的模型调用继续带上原始业务上下文。
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
        # 输出恢复启动日志，方便确认本次人工回调传入了哪些恢复参数。
        logger.info(
            "Diagnosis LangGraph resume start trace_id=%s workflow_thread_id=%s resume_keys=%s existing_keys=%s",
            trace_id,
            workflow_thread_id,
            sorted(resume_payload.keys()),
            sorted(final_state.keys()),
        )
        # 将恢复阶段的 trace 信息写入 LLM 执行器，保证后续节点模型调用仍能被观测系统串联。
        token = self.llm.set_trace_context(**metadata)
        try:
            # 单独用 diagnosis_resume 统计恢复流程耗时，便于和首次运行区分。
            with monitor.track_workflow("diagnosis_resume", final_state.get("chat_id")):
                # Command(resume=...) 会把人工确认结果交回 LangGraph，让流程从 interrupt 处继续。
                async for update in app.astream(
                    Command(resume=resume_payload),
                    config=config,
                    stream_mode="updates",
                ):
                    # 恢复后的每个节点仍然按增量返回，逐个合并到最终状态里。
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
                        # 恢复执行期间也继续发送进度消息，保持用户侧体验一致。
                        await self._send_progress(node_name, final_state)
                # 恢复完成后再读一次 checkpointer，确保最终状态和持久化状态一致。
                state_snapshot = await app.aget_state(config)
                final_values = getattr(state_snapshot, "values", None)
                if isinstance(final_values, dict):
                    final_state.update(final_values)
        finally:
            # 清理本次 resume 写入的 LLM trace 上下文。
            self.llm.reset_trace_context(token)
        # 输出 resume 完成日志，记录最终状态字段和耗时。
        logger.info(
            "Diagnosis LangGraph resume completed trace_id=%s workflow_thread_id=%s final_keys=%s elapsed_ms=%s",
            trace_id,
            workflow_thread_id,
            sorted(final_state.keys()),
            int((time.perf_counter() - started) * 1000),
        )
        return final_state

    async def update_state(self, workflow_thread_id: str, state_update: dict[str, Any]) -> None:
        # 获取已编译的流程图，状态更新需要通过 LangGraph 应用实例写入 checkpointer。
        app = self.compile()
        # 按 thread_id 定位要更新的诊断线程，把外部补充字段合并到该线程状态里。
        await app.aupdate_state(
            {"configurable": {"thread_id": workflow_thread_id}, "recursion_limit": 20},
            state_update,
        )

    def _node_registry(self) -> dict[str, NodeFn]:
        # partial 在这里注入外部依赖，让各节点函数保持“输入 state，输出 state 增量”的简单形态。
        # registry 的 key 必须和 workflow.yaml 中的节点名一致，value 是该节点真正要执行的函数。
        registry = {
            # understand 节点负责理解告警，并可在历史案例缓存命中时提前给出候选方案。
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
            # retrieve 节点负责 RAG 检索，把知识库和历史经验补充进状态。
            "retrieve": partial(retrieve_node, retriever=self.retriever),
            # fetch_live_data 节点负责查询实时指标、日志和拓扑。
            "fetch_live_data": fetch_live_data_node,
            # generate_plan 节点把上下文交给 LLM，生成推荐诊断方案。
            "generate_plan": partial(generate_plan_node, llm=self.llm),
            # validate 节点对推荐方案做安全性和可执行性校验。
            "validate": partial(validate_plan_node, llm=self.llm),
            # human_confirm 节点负责把方案发给人确认，并在需要时暂停流程。
            "human_confirm": partial(
                human_confirm_node,
                sender=self.sender,
                decision_store=self.decision_store,
                timeout_seconds=self.settings.aiops_human_confirm_timeout_seconds,
                enabled=self.settings.aiops_human_confirm_enabled,
            ),
            # final_result 节点负责发送最终诊断结果。
            "final_result": partial(final_result_node, sender=self.sender),
            # feedback_learning 节点根据用户反馈决定是否把本次诊断沉淀成历史案例。
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
        # 返回一个包裹后的节点函数，LangGraph 实际执行的是 wrapped。
        async def wrapped(state: DiagnosisState) -> DiagnosisState:
            # 每次节点执行前先从状态里提取 trace_id，保证日志能关联到同一次诊断。
            trace_id = trace_id_from_state(state)
            logger.info("Diagnosis node execution start trace_id=%s node=%s", trace_id, node_name)
            # monitor.track_node 会自动记录节点耗时和成功/失败次数。
            with monitor.track_node(node_name, state.get("chat_id")):
                result = await node(state)
            # 节点执行完成后记录日志，便于和 start 日志配对查看耗时和异常位置。
            logger.info("Diagnosis node execution completed trace_id=%s node=%s", trace_id, node_name)
            return result

        return wrapped

    def _route_registry(self) -> dict[str, Callable[[DiagnosisState], str]]:
        # 路由函数返回值必须和 workflow.yaml 里的 conditional_edges.mapping key 保持一致。
        # 这些函数只负责返回路由 key，真正跳到哪个节点由 workflow.yaml 的 mapping 决定。
        return {
            # 根据告警内容判断是否需要继续查询实时指标、日志和拓扑。
            "should_fetch": should_fetch,
            # 根据方案校验结果决定进入人工确认还是回到方案生成。
            "validation_result": validation_result,
            # 缓存命中时走历史案例复用分支，未命中时继续普通 RAG 检索。
            "cache_decision": lambda state: "hit" if state.get("cache_hit", False) else "miss",
            # 人工确认节点把用户选择写进 state，这里直接读取选择结果作为路由 key。
            "human_decision": lambda state: state.get("human_decision", "timeout"),
            # always 和 end 是配置文件中使用的简单路由占位。
            "always": lambda state: "next",
            "end": lambda state: "end",
        }

    async def _send_progress(self, node_name: str, state: DiagnosisState) -> None:
        # 进度消息是用户体验层增强；真正的流程状态仍以 LangGraph state 为准。
        # 不同节点对应不同的人类可读提示，便于用户理解当前诊断进度。
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
            # 没有配置提示文案的节点不发送进度消息，避免飞书里出现无意义提醒。
            return
        # 记录发送进度前的关键信息，方便排查飞书消息是否发到正确会话和话题。
        logger.info(
            "Sending workflow progress node=%s chat_id=%s thread_root_message_id=%s status=%s",
            node_name,
            state.get("chat_id"),
            state.get("thread_root_message_id"),
            status,
        )
        # 进度消息回复到原话题下，避免一次诊断在群里刷出多条独立消息。
        await self.sender.send_message(
            state.get("chat_id"),
            status,
            thread_root_message_id=state.get("thread_root_message_id"),
        )

    def _langfuse_metadata(self, state: DiagnosisState) -> dict[str, object]:
        # trace_id 是业务链路的核心标识，Langfuse、日志和节点状态都会使用它。
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
    # 根据配置创建模型执行器，内部会处理多模型 fallback、重试和观测上报。
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
