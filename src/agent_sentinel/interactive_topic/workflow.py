from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent_sentinel.agents.fetch_tools import fetch_live_data_node
from agent_sentinel.agents.generate_plan import generate_plan_node
from agent_sentinel.agents.understand_agent import understand_node
from agent_sentinel.agents.validate_plan import validate_plan_node
from agent_sentinel.graph.state import DiagnosisState
from agent_sentinel.interactive_topic.state import TopicFlowState, append_node_result, increment_retry
from agent_sentinel.interactive_topic.task_store import TopicTaskStore
from agent_sentinel.interactive_topic.topic_sender import InteractiveTopicSender, WORKFLOW_STEPS
from agent_sentinel.llm.executor import LLMExecutor
from agent_sentinel.monitoring import monitor, trace_id_from_parts
from agent_sentinel.observability.langfuse import current_trace_context
from agent_sentinel.rag.base import BaseRetriever
from agent_sentinel.rag.history_cases import HistoryCaseStore

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)
# 将 node_event_json_logger 的值保存下来，供后续流程判断或组装响应时使用。
node_event_json_logger = logging.getLogger("agent_sentinel.node_event_json")

# 将 NodeRunResult 的值保存下来，供后续流程判断或组装响应时使用。
NodeRunResult = str | tuple[str, DiagnosisState]
# 将 NodeRunner 的值保存下来，供后续流程判断或组装响应时使用。
NodeRunner = Callable[[TopicFlowState], Awaitable[NodeRunResult]]
# 将 MAX_NODE_RETRIES 的值保存下来，供后续流程判断或组装响应时使用。
MAX_NODE_RETRIES = 2
# 将 RAG_DISPLAY_TOP_K 的值保存下来，供后续流程判断或组装响应时使用。
RAG_DISPLAY_TOP_K = 2
# 将 MAX_ROUTED_TOOL_CALLS 的值保存下来，供后续流程判断或组装响应时使用。
MAX_ROUTED_TOOL_CALLS = 3
# 将 _tools_provider 的值保存下来，供后续流程判断或组装响应时使用。
_tools_provider: Any | None = None


# 定义 _get_tools_provider 相关的处理逻辑，供流程或外部调用复用。
def _get_tools_provider() -> Any:
    # 按需创建实时工具提供者，只有单卡片流程走到工具节点时才连接外部系统。
    global _tools_provider
    # 根据 _tools_provider is None 判断当前流程该进入哪个处理分支。
    if _tools_provider is None:
        # 交互式 ReAct 工具按需初始化，避免没有进入工具节点时就连接外部系统。
        from agent_sentinel.config import get_settings
        from agent_sentinel.tools.factory import build_tools_provider

        # 将 _tools_provider 的值保存下来，供后续流程判断或组装响应时使用。
        _tools_provider = build_tools_provider(get_settings())
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive ReAct tools provider initialized provider=%s", type(_tools_provider).__name__)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return _tools_provider


# 定义 InteractiveTopicWorkflow 组件，集中管理这个模块的状态和行为。
class InteractiveTopicWorkflow:
    """Interactive Feishu workflow using one continuously updated card."""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        sender: InteractiveTopicSender,
        # 将 wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        wait_seconds: int = 5,
        # 将 retriever 的值保存下来，供后续流程判断或组装响应时使用。
        retriever: BaseRetriever | None = None,
        # 将 case_store 的值保存下来，供后续流程判断或组装响应时使用。
        case_store: HistoryCaseStore | None = None,
        # 将 llm 的值保存下来，供后续流程判断或组装响应时使用。
        llm: LLMExecutor | None = None,
        # 将 checkpointer 的值保存下来，供后续流程判断或组装响应时使用。
        checkpointer: Any | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 组装单卡片交互式诊断流程需要的发送器、检索器、案例库、模型和任务状态。
        self.sender = sender
        # 将 self.wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        self.wait_seconds = wait_seconds
        # 将 self.retriever 的值保存下来，供后续流程判断或组装响应时使用。
        self.retriever = retriever
        # 将 self.case_store 的值保存下来，供后续流程判断或组装响应时使用。
        self.case_store = case_store
        # 将 self.llm 的值保存下来，供后续流程判断或组装响应时使用。
        self.llm = llm
        # 将 self.task_store 的值保存下来，供后续流程判断或组装响应时使用。
        self.task_store = TopicTaskStore(wait_seconds)
        # 将 self.checkpointer 的值保存下来，供后续流程判断或组装响应时使用。
        self.checkpointer = checkpointer or InMemorySaver()
        # 将 self._compiled 的值保存下来，供后续流程判断或组装响应时使用。
        self._compiled: Any | None = None
        # 将 self._task_locks 的值保存下来，供后续流程判断或组装响应时使用。
        self._task_locks: dict[str, asyncio.Lock] = {}
        # 将 self._task_locks_guard 的值保存下来，供后续流程判断或组装响应时使用。
        self._task_locks_guard = asyncio.Lock()
        # 将 self._loop 的值保存下来，供后续流程判断或组装响应时使用。
        self._loop: asyncio.AbstractEventLoop | None = None
        # 将 self._loop_thread 的值保存下来，供后续流程判断或组装响应时使用。
        self._loop_thread: threading.Thread | None = None
        # 将 self._loop_started 的值保存下来，供后续流程判断或组装响应时使用。
        self._loop_started = threading.Event()
        # 将 self._loop_guard 的值保存下来，供后续流程判断或组装响应时使用。
        self._loop_guard = threading.Lock()

    # 定义 compile 相关的处理逻辑，供流程或外部调用复用。
    def compile(self) -> Any:
        # 把单卡片诊断步骤编译成 LangGraph 流程，每一步执行后都等待用户确认或重试。
        if self._compiled is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return self._compiled

        # 将 builder 的值保存下来，供后续流程判断或组装响应时使用。
        builder = StateGraph(TopicFlowState)
        # 每个节点都通过 _interactive_node 包装，统一处理卡片更新、确认、重试和指标。
        builder.add_node("understand", partial(self._interactive_node, "understand", self._understand))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("cache_check", partial(self._interactive_node, "cache_check", self._cache_check))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("rag_retrieve", partial(self._interactive_node, "rag_retrieve", self._rag_retrieve))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("tool_router", partial(self._interactive_node, "tool_router", self._tool_router))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("tool_executor", partial(self._interactive_node, "tool_executor", self._tool_executor))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("evidence_review", partial(self._interactive_node, "evidence_review", self._evidence_review))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("generate_plan", partial(self._interactive_node, "generate_plan", self._generate_plan))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("validate", partial(self._interactive_node, "validate", self._validate))
        # 调用 builder.add_node 完成当前步骤需要的业务处理。
        builder.add_node("summary", partial(self._interactive_node, "summary", self._summary))

        # 调用 builder.set_entry_point 完成当前步骤需要的业务处理。
        builder.set_entry_point("understand")
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "understand",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "understand", "next": "cache_check"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "cache_check",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "cache_check", "next": "rag_retrieve"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "rag_retrieve",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "rag_retrieve", "next": "tool_router"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tool_router",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "tool_router", "next": "tool_executor"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tool_executor",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "tool_executor", "next": "evidence_review"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "evidence_review",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "evidence_review", "next": "generate_plan"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "generate_plan",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "generate_plan", "next": "validate"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "validate",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "validate", "next": "summary"},
        )
        # 调用 builder.add_conditional_edges 完成当前步骤需要的业务处理。
        builder.add_conditional_edges(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "summary",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._route_after_confirm,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"retry": "summary", "next": END},
        )
        # 将 self._compiled 的值保存下来，供后续流程判断或组装响应时使用。
        self._compiled = builder.compile(checkpointer=self.checkpointer)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive topic LangGraph compiled")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._compiled

    # 定义 start 相关的处理逻辑，供流程或外部调用复用。
    async def start(self, chat_id: str, root_message_id: str, query: str) -> str:
        # HTTP/长连接回调线程不直接跑工作流，而是提交到内部事件循环串行管理。
        # 异步启动一次单卡片诊断任务，并返回本次任务 ID。
        future = self._submit(self._start_impl(chat_id, root_message_id, query))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.wrap_future(future)

    # 定义 start_sync 相关的处理逻辑，供流程或外部调用复用。
    def start_sync(self, chat_id: str, root_message_id: str, query: str) -> tuple[str, bool]:
        # 同步启动一次单卡片诊断任务，供普通同步入口复用同一套流程。
        future = self._submit(self._start_impl(chat_id, root_message_id, query))
        # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
        task_id = future.result()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"Interactive topic workflow started: {task_id}", True

    # 定义 handle_card_callback 相关的处理逻辑，供流程或外部调用复用。
    async def handle_card_callback(self, payload: dict[str, Any], *, source: str = "card") -> dict[str, str]:
        # 异步处理飞书卡片按钮点击，把用户的“下一步/重试/反馈”动作送回流程。
        future = self._submit(self._handle_card_callback_impl(payload, source=source))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.wrap_future(future)

    # 定义 handle_card_callback_sync 相关的处理逻辑，供流程或外部调用复用。
    def handle_card_callback_sync(self, payload: dict[str, Any], *, source: str = "card") -> dict[str, str]:
        # 同步处理飞书卡片按钮点击，供长连接回调等同步场景调用。
        future = self._submit(self._handle_card_callback_impl(payload, source=source))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return future.result()

    # 定义 _start_impl 相关的处理逻辑，供流程或外部调用复用。
    async def _start_impl(self, chat_id: str, root_message_id: str, query: str) -> str:
        # 真正创建诊断任务、发送初始卡片，并把初始状态交给 LangGraph 执行。
        started = time.perf_counter()
        # 将 existing_task 的值保存下来，供后续流程判断或组装响应时使用。
        existing_task = self.task_store.get_task_by_source(chat_id, root_message_id)
        # 根据 existing_task is not None 判断当前流程该进入哪个处理分支。
        if existing_task is not None:
            # 同一条飞书消息可能被重复投递，按 chat_id + root_message_id 去重，避免创建多张卡片。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive topic duplicate start ignored trace_id=%s existing_task_id=%s chat_id=%s root_message_id=%s query_chars=%s",
                # 调用 trace_id_from_parts 完成当前步骤需要的业务处理。
                trace_id_from_parts(chat_id, root_message_id),
                # 执行当前业务步骤，推动流程继续向下游推进。
                existing_task.task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 调用 len 完成当前步骤需要的业务处理。
                len(query or ""),
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return existing_task.task_id
        # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
        task_id = f"topic-{uuid.uuid4()}"
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive topic start trace_id=%s task_id=%s chat_id=%s root_message_id=%s query_chars=%s wait_seconds=%s",
            # 调用 trace_id_from_parts 完成当前步骤需要的业务处理。
            trace_id_from_parts(chat_id, root_message_id),
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            root_message_id,
            # 调用 len 完成当前步骤需要的业务处理。
            len(query or ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.wait_seconds,
        )
        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task = self.task_store.create_task(task_id, chat_id, root_message_id, query)
        # 根据 task.task_id != task_id 判断当前流程该进入哪个处理分支。
        if task.task_id != task_id:
            # create_task 内部也会处理并发去重，这里兜底返回已经存在的任务。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive topic task already exists after create task_id=%s requested_task_id=%s chat_id=%s root_message_id=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                task.task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return task.task_id
        # 根据 monitor.enabled 判断当前流程该进入哪个处理分支。
        if monitor.enabled:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            monitor.current_workflow_active.labels(workflow_type="interactive_topic", group_id=chat_id or "unknown").inc()
        # 将 card_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        card_message_id = await self.sender.send_workflow_card(chat_id, root_message_id, task_id, query)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive topic initial card sent task_id=%s card_message_id=%s", task_id, card_message_id)
        # 调用 task_store.set_card_message_id 完成当前步骤需要的业务处理。
        self.task_store.set_card_message_id(task_id, card_message_id)
        # 将 initial_state 的值保存下来，供后续流程判断或组装响应时使用。
        initial_state: TopicFlowState = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "query": query,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "task_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "chat_id": chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "root_message_id": root_message_id,
            # 调用 trace_id_from_parts 完成当前步骤需要的业务处理。
            "trace_id": trace_id_from_parts(chat_id, root_message_id),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "retry_counts": {},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "node_results": [],
            # 调用 self._initial_diagnosis_state 完成当前步骤需要的业务处理。
            "diagnosis_state": self._initial_diagnosis_state(
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=chat_id,
                # 将 root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                root_message_id=root_message_id,
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=query,
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id=task_id,
            ),
        }
        # 等待异步操作完成，再继续推进当前业务流程。
        await self._drive(task_id, initial_state)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive topic start submitted task_id=%s elapsed_ms=%s", task_id, int((time.perf_counter() - started) * 1000))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return task_id

    # 定义 _handle_card_callback_impl 相关的处理逻辑，供流程或外部调用复用。
    async def _handle_card_callback_impl(self, payload: dict[str, Any], *, source: str = "card") -> dict[str, str]:
        # 解析卡片回调里的任务、节点和动作，再决定是继续流程、重试节点还是记录反馈。
        value = self._extract_card_value(payload)
        # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
        task_id = str(value.get("task_id") or "")
        # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
        node_name = str(value.get("node_name") or value.get("node") or "")
        # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
        action = str(value.get("action") or value.get("operate") or "")
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive topic card callback parsed task_id=%s node=%s action=%s source=%s value_keys=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            action,
            # 执行当前业务步骤，推动流程继续向下游推进。
            source,
            # 调用 sorted 完成当前步骤需要的业务处理。
            sorted(value.keys()),
        )
        # 根据 action == "noop" 判断当前流程该进入哪个处理分支。
        if action == "noop":
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic callback ignored noop task_id=%s source=%s", task_id, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}
        # 根据 action in {"feedback_valid", "feedback_invalid"} 判断当前流程该进入哪个处理分支。
        if action in {"feedback_valid", "feedback_invalid"}:
            # 最终反馈不恢复 LangGraph 节点，只记录用户对结果有效性的评价。
            return await self._handle_feedback_callback(task_id, action, source=source)
        # 根据 not task_id or not node_name or action not in {"next... 判断当前流程该进入哪个处理分支。
        if not task_id or not node_name or action not in {"next", "retry"}:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic callback ignored invalid value task_id=%s node=%s action=%s source=%s", task_id, node_name, action, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}

        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task = self.task_store.confirm_action(task_id, node_name, action, source=source)
        # 根据 task is None 判断当前流程该进入哪个处理分支。
        if task is None:
            # 过期或重复按钮点击会被忽略，避免旧卡片状态覆盖新节点执行。
            logger.info("Interactive topic callback ignored stale task_id=%s node=%s action=%s source=%s", task_id, node_name, action, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}

        # 根据 action == "retry" 判断当前流程该进入哪个处理分支。
        if action == "retry":
            # 等待异步操作完成，再继续推进当前业务流程。
            await self.sender.update_workflow_card(
                # 执行当前业务步骤，推动流程继续向下游推进。
                task.card_message_id,
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id=task_id,
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=task.query,
                # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                node_statuses=self._node_statuses_until(node_name, "failed", self._node_results(task_id)),
                # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                current_node=node_name,
                # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                current_result="用户已拒绝当前节点结果，准备重试该节点。",
                # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                buttons_node=None,
            )
        # 处理前面条件都不满足时的默认分支。
        else:
            # 将 status_text 的值保存下来，供后续流程判断或组装响应时使用。
            status_text = "超时自动继续" if source == "timeout" else "用户已同意，继续执行"
            # 等待异步操作完成，再继续推进当前业务流程。
            await self.sender.update_workflow_card(
                # 执行当前业务步骤，推动流程继续向下游推进。
                task.card_message_id,
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id=task_id,
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=task.query,
                # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                node_statuses=self._node_statuses_until(node_name, "done", self._node_results(task_id)),
                # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                current_node=node_name,
                # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                current_result=status_text,
                # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                buttons_node=None,
            )

        # 等待异步操作完成，再继续推进当前业务流程。
        await self._drive(task_id, Command(resume={"action": action, "source": source}))
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive topic callback applied task_id=%s node=%s action=%s source=%s", task_id, node_name, action, source)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"status": "ok"}

    # 定义 _interactive_node 相关的处理逻辑，供流程或外部调用复用。
    async def _interactive_node(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        runner: NodeRunner,
        # 执行当前业务步骤，推动流程继续向下游推进。
        state: TopicFlowState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> TopicFlowState:
        # 包装单卡片流程里的每个节点，统一处理执行、展示、等待确认、重试和错误状态。
        task_id = state["task_id"]
        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task = self.task_store.get_task(task_id)
        # 将 card_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        card_message_id = task.card_message_id if task else None
        # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
        query = state.get("query", "")
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 将 replay_result 的值保存下来，供后续流程判断或组装响应时使用。
        replay_result = self._confirmed_replay_result(task, node_name)
        # 根据 replay_result is not None 判断当前流程该进入哪个处理分支。
        if replay_result is not None:
            # 用户确认会触发 LangGraph resume；节点重入时复用已展示结果，不重复调用 LLM/工具。
            action = str(task.last_action or "next") if task else "next"
            # 根据 action not in {"next", "retry"} 判断当前流程该进入哪个处理分支。
            if action not in {"next", "retry"}:
                # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
                action = "next"
            # 将 retry_counts 的值保存下来，供后续流程判断或组装响应时使用。
            retry_counts = increment_retry(state, node_name) if action == "retry" else dict(state.get("retry_counts", {}))
            # 调用 self._log_node_event 完成当前步骤需要的业务处理。
            self._log_node_event(
                # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
                event="confirmation_replay",
                # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
                node_name=node_name,
                # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
                state=state,
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status="resumed",
                # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
                task=task,
                # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
                action=action,
                # 将 source_state 的值保存下来，供后续流程判断或组装响应时使用。
                source_state=task.status if task else "",
                # 将 retry_count 的值保存下来，供后续流程判断或组装响应时使用。
                retry_count=retry_counts.get(node_name, 0),
            )
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive node confirmation replay task_id=%s node=%s action=%s source_state=%s retry_count=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                node_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                action,
                # 执行当前业务步骤，推动流程继续向下游推进。
                task.status if task else None,
                # 调用 retry_counts.get 完成当前步骤需要的业务处理。
                retry_counts.get(node_name, 0),
            )
            # 根据 action == "retry" and retry_counts.get(node_name, 0)... 判断当前流程该进入哪个处理分支。
            if action == "retry" and retry_counts.get(node_name, 0) >= MAX_NODE_RETRIES:
                # 连续拒绝达到上限后自动跳过当前节点，防止人工交互导致流程卡死。
                action = "skip"
                # 将 replay_result 的值保存下来，供后续流程判断或组装响应时使用。
                replay_result = f"{replay_result}\n\n已重试 {MAX_NODE_RETRIES} 次，自动跳过当前节点并进入下一节点。"
                # 等待异步操作完成，再继续推进当前业务流程。
                await self.sender.update_workflow_card(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    card_message_id,
                    # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                    task_id=task_id,
                    # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                    query=query,
                    # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                    node_statuses=self._node_statuses_until(node_name, "skipped", state.get("node_results", [])),
                    # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                    current_node=node_name,
                    # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                    current_result=replay_result,
                    # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                    buttons_node=None,
                )
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Interactive node retry limit reached task_id=%s node=%s max_retries=%s", task_id, node_name, MAX_NODE_RETRIES)
                # 调用 self._log_node_event 完成当前步骤需要的业务处理。
                self._log_node_event(
                    # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
                    event="retry_limit",
                    # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
                    node_name=node_name,
                    # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
                    state=state,
                    # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                    status="skipped",
                    # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
                    task=task,
                    # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
                    action=action,
                    # 将 max_retries 的值保存下来，供后续流程判断或组装响应时使用。
                    max_retries=MAX_NODE_RETRIES,
                    # 将 retry_count 的值保存下来，供后续流程判断或组装响应时使用。
                    retry_count=retry_counts.get(node_name, 0),
                )
            # 将 replay_update 的值保存下来，供后续流程判断或组装响应时使用。
            replay_update: TopicFlowState = {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "current_node": node_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "node_result": replay_result,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "last_action": action,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "retry_counts": retry_counts,
                # 调用 append_node_result 完成当前步骤需要的业务处理。
                "node_results": append_node_result(state, node_name, replay_result),
            }
            # 根据 action == "next" and task and task.pending_diagnosis... 判断当前流程该进入哪个处理分支。
            if action == "next" and task and task.pending_diagnosis_state:
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                replay_update["diagnosis_state"] = task.pending_diagnosis_state  # type: ignore[typeddict-item]
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replay_update

        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive node start trace_id=%s task_id=%s node=%s retry_count=%s previous_results=%s card_message_id=%s",
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("trace_id") or trace_id_from_parts(state.get("chat_id"), state.get("root_message_id")),
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_name,
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("retry_counts", {}).get(node_name, 0),
            # 调用 len 完成当前步骤需要的业务处理。
            len(state.get("node_results", [])),
            # 执行当前业务步骤，推动流程继续向下游推进。
            card_message_id,
        )
        # 调用 self._log_node_event 完成当前步骤需要的业务处理。
        self._log_node_event(
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event="start",
            # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
            node_name=node_name,
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state=state,
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status="running",
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task=task,
        )

        # 等待异步操作完成，再继续推进当前业务流程。
        await self.sender.update_workflow_card(
            # 执行当前业务步骤，推动流程继续向下游推进。
            card_message_id,
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=task_id,
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query=query,
            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
            node_statuses=self._node_statuses_until(node_name, "running", state.get("node_results", [])),
            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
            current_node=node_name,
            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
            current_result=f"正在执行 {_node_title(node_name)} 节点...",
            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
            buttons_node=None,
        )

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 进入上下文管理器保护的区域，自动处理资源生命周期。
            with monitor.track_node(node_name, state.get("chat_id")):
                # runner 返回展示文本和可选诊断状态，统一由 _normalize_node_result 归一化。
                run_result = await runner(state)
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                node_result, diagnosis_state = self._normalize_node_result(run_result, state)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Interactive topic node failed task_id=%s node=%s", task_id, node_name)
            # 调用 self._log_node_event 完成当前步骤需要的业务处理。
            self._log_node_event(
                # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
                event="failed",
                # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
                node_name=node_name,
                # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
                state=state,
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status="skipped",
                # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
                task=task,
                # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                # 将 error 的值保存下来，供后续流程判断或组装响应时使用。
                error=exc,
            )
            # 将 node_result 的值保存下来，供后续流程判断或组装响应时使用。
            node_result = f"{_node_title(node_name)} 节点执行失败，已跳过当前节点并继续后续流程：{exc}"
            # 等待异步操作完成，再继续推进当前业务流程。
            await self.sender.update_workflow_card(
                # 执行当前业务步骤，推动流程继续向下游推进。
                card_message_id,
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id=task_id,
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=query,
                # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                node_statuses=self._node_statuses_until(node_name, "skipped", state.get("node_results", [])),
                # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                current_node=node_name,
                # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                current_result=node_result,
                # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                buttons_node=None,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "current_node": node_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "node_result": node_result,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "last_action": "skip",
                # 调用 dict 完成当前步骤需要的业务处理。
                "retry_counts": dict(state.get("retry_counts", {})),
                # 调用 append_node_result 完成当前步骤需要的业务处理。
                "node_results": append_node_result(state, node_name, node_result),
            }

        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive node completed task_id=%s node=%s result_chars=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_name,
            # 调用 len 完成当前步骤需要的业务处理。
            len(node_result or ""),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 调用 self._log_node_event 完成当前步骤需要的业务处理。
        self._log_node_event(
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event="completed",
            # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
            node_name=node_name,
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state=state,
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status="done",
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task=task,
            # 将 result_chars 的值保存下来，供后续流程判断或组装响应时使用。
            result_chars=len(node_result or ""),
            # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        # 等待异步操作完成，再继续推进当前业务流程。
        await self.sender.update_workflow_card(
            # 执行当前业务步骤，推动流程继续向下游推进。
            card_message_id,
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=task_id,
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query=query,
            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
            node_statuses=self._node_statuses_until(node_name, "waiting", state.get("node_results", [])),
            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
            current_node=node_name,
            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
            current_result=node_result,
            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
            buttons_node=node_name,
        )

        # 调用 task_store.mark_waiting 完成当前步骤需要的业务处理。
        self.task_store.mark_waiting(
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            card_message_id,
            # 调用 self._on_timeout 完成当前步骤需要的业务处理。
            lambda timeout_task_id, timeout_node_name: self._on_timeout(timeout_task_id, timeout_node_name),
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_result,
            # 执行当前业务步骤，推动流程继续向下游推进。
            diagnosis_state,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive node waiting for confirmation task_id=%s node=%s timeout_seconds=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.wait_seconds,
        )
        # 调用 self._log_node_event 完成当前步骤需要的业务处理。
        self._log_node_event(
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event="waiting",
            # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
            node_name=node_name,
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state=state,
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status="waiting",
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task=task,
            # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            timeout_seconds=self.wait_seconds,
            # 将 result_chars 的值保存下来，供后续流程判断或组装响应时使用。
            result_chars=len(node_result or ""),
        )

        # 将 resume_payload 的值保存下来，供后续流程判断或组装响应时使用。
        resume_payload = interrupt(
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "task_id": task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "node_name": node_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "card_message_id": card_message_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "timeout_seconds": self.wait_seconds,
            }
        )
        # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
        action = str((resume_payload or {}).get("action") or "next")
        # 根据 action not in {"next", "retry"} 判断当前流程该进入哪个处理分支。
        if action not in {"next", "retry"}:
            # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
            action = "next"
        # 将 retry_counts 的值保存下来，供后续流程判断或组装响应时使用。
        retry_counts = increment_retry(state, node_name) if action == "retry" else dict(state.get("retry_counts", {}))
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive node resumed task_id=%s node=%s action=%s source=%s retry_count=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            node_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            action,
            # 调用 get 完成当前步骤需要的业务处理。
            (resume_payload or {}).get("source"),
            # 调用 retry_counts.get 完成当前步骤需要的业务处理。
            retry_counts.get(node_name, 0),
        )
        # 调用 self._log_node_event 完成当前步骤需要的业务处理。
        self._log_node_event(
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event="resumed",
            # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
            node_name=node_name,
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state=state,
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status="resumed",
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task=task,
            # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
            action=action,
            # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
            source=(resume_payload or {}).get("source"),
            # 将 retry_count 的值保存下来，供后续流程判断或组装响应时使用。
            retry_count=retry_counts.get(node_name, 0),
        )
        # 根据 action == "retry" and retry_counts.get(node_name, 0)... 判断当前流程该进入哪个处理分支。
        if action == "retry" and retry_counts.get(node_name, 0) >= MAX_NODE_RETRIES:
            # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
            action = "skip"
            # 将 node_result 的值保存下来，供后续流程判断或组装响应时使用。
            node_result = f"{node_result}\n\n已重试 {MAX_NODE_RETRIES} 次，自动跳过当前节点并进入下一节点。"
            # 等待异步操作完成，再继续推进当前业务流程。
            await self.sender.update_workflow_card(
                # 执行当前业务步骤，推动流程继续向下游推进。
                card_message_id,
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id=task_id,
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=query,
                # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                node_statuses=self._node_statuses_until(node_name, "skipped", state.get("node_results", [])),
                # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                current_node=node_name,
                # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                current_result=node_result,
                # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                buttons_node=None,
            )
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive node retry limit reached task_id=%s node=%s max_retries=%s", task_id, node_name, MAX_NODE_RETRIES)
            # 调用 self._log_node_event 完成当前步骤需要的业务处理。
            self._log_node_event(
                # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
                event="retry_limit",
                # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
                node_name=node_name,
                # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
                state=state,
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status="skipped",
                # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
                task=task,
                # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
                action=action,
                # 将 max_retries 的值保存下来，供后续流程判断或组装响应时使用。
                max_retries=MAX_NODE_RETRIES,
                # 将 retry_count 的值保存下来，供后续流程判断或组装响应时使用。
                retry_count=retry_counts.get(node_name, 0),
            )
        # 将 update 的值保存下来，供后续流程判断或组装响应时使用。
        update: TopicFlowState = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "current_node": node_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "node_result": node_result,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "last_action": action,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "retry_counts": retry_counts,
            # 调用 append_node_result 完成当前步骤需要的业务处理。
            "node_results": append_node_result(state, node_name, node_result),
        }
        # 根据 action == "next" 判断当前流程该进入哪个处理分支。
        if action == "next":
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            update["diagnosis_state"] = diagnosis_state
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for state_key in ("tool_plan", "tool_results", "evidence_review"):
                # 根据 state_key in diagnosis_state 判断当前流程该进入哪个处理分支。
                if state_key in diagnosis_state:
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    update[state_key] = diagnosis_state[state_key]  # type: ignore[typeddict-item]
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return update

    # 定义 _drive 相关的处理逻辑，供流程或外部调用复用。
    async def _drive(self, task_id: str, graph_input: TopicFlowState | Command) -> None:
        # 驱动指定任务继续往前跑，直到下一个需要用户确认的节点或整个流程结束。
        app = self.compile()
        # 将 trace_id 的值保存下来，供后续流程判断或组装响应时使用。
        trace_id = self._trace_id_for_input(task_id, graph_input)
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = self._langfuse_metadata(task_id, graph_input)
        # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
        config = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "configurable": {"thread_id": task_id, "trace_id": trace_id, "task_id": task_id},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "recursion_limit": 50,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "metadata": metadata,
        }
        # 将 lock 的值保存下来，供后续流程判断或组装响应时使用。
        lock = await self._get_task_lock(task_id)
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive drive start task_id=%s input_type=%s", task_id, type(graph_input).__name__)
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self.llm.set_trace_context(**metadata) if self.llm is not None else None
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 进入异步上下文管理区域，确保异步资源按约定释放。
            async with lock:
                # 异步遍历上游数据流，边接收边处理每个结果。
                async for update in app.astream(graph_input, config=config, stream_mode="updates"):
                    # 记录关键运行信息，方便排查流程进展和异常现场。
                    logger.info("Interactive drive update task_id=%s nodes=%s", task_id, list(update.keys()))
                # 将 snapshot 的值保存下来，供后续流程判断或组装响应时使用。
                snapshot = await app.aget_state(config)
                # 将 next_nodes 的值保存下来，供后续流程判断或组装响应时使用。
                next_nodes = tuple(getattr(snapshot, "next", ()) or ())
                # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
                values = getattr(snapshot, "values", {}) or {}
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Interactive drive snapshot task_id=%s next_nodes=%s node_results=%s elapsed_ms=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    next_nodes,
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(values.get("node_results", []) or []),
                    # 调用 int 完成当前步骤需要的业务处理。
                    int((time.perf_counter() - started) * 1000),
                )
                # 根据 not next_nodes 判断当前流程该进入哪个处理分支。
                if not next_nodes:
                    # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
                    task = self.task_store.get_task(task_id)
                    # 根据 task is not None 判断当前流程该进入哪个处理分支。
                    if task is not None:
                        # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
                        final_text = self._build_final_text(values)
                        # 根据 monitor.enabled and task.created_at_monotonic 判断当前流程该进入哪个处理分支。
                        if monitor.enabled and task.created_at_monotonic:
                            # 调用 workflow_total_duration_seconds.labels 完成当前步骤需要的业务处理。
                            monitor.workflow_total_duration_seconds.labels(
                                # 将 workflow_type 的值保存下来，供后续流程判断或组装响应时使用。
                                workflow_type="interactive_topic",
                                # 将 group_id 的值保存下来，供后续流程判断或组装响应时使用。
                                group_id=task.chat_id or "unknown",
                            # 调用 observe 完成当前步骤需要的业务处理。
                            ).observe(time.perf_counter() - task.created_at_monotonic)
                        # 记录关键运行信息，方便排查流程进展和异常现场。
                        logger.info(
                            # 保存当前计算结果，供后续流程判断或组装响应时使用。
                            "Interactive topic final card update task_id=%s final_chars=%s node_results=%s",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            task_id,
                            # 调用 len 完成当前步骤需要的业务处理。
                            len(final_text),
                            # 调用 len 完成当前步骤需要的业务处理。
                            len(values.get("node_results", []) or []),
                        )
                        # 等待异步操作完成，再继续推进当前业务流程。
                        await self.sender.update_workflow_card(
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            task.card_message_id,
                            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                            task_id=task_id,
                            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                            query=task.query,
                            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                            node_statuses={step.node_name: "done" for step in WORKFLOW_STEPS},
                            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                            current_node=None,
                            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                            current_result=final_text,
                            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                            buttons_node=None,
                            # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
                            feedback_buttons=True,
                        )
                        # 根据 monitor.enabled 判断当前流程该进入哪个处理分支。
                        if monitor.enabled:
                            # 调用 current_workflow_active.labels 完成当前步骤需要的业务处理。
                            monitor.current_workflow_active.labels(
                                # 将 workflow_type 的值保存下来，供后续流程判断或组装响应时使用。
                                workflow_type="interactive_topic",
                                # 将 group_id 的值保存下来，供后续流程判断或组装响应时使用。
                                group_id=task.chat_id or "unknown",
                            # 调用 dec 完成当前步骤需要的业务处理。
                            ).dec()
                        # 调用 task_store.mark_feedback_waiting 完成当前步骤需要的业务处理。
                        self.task_store.mark_feedback_waiting(task_id, task.card_message_id)
        # 无论前面的流程是否成功，都执行资源清理或上下文恢复。
        finally:
            # 根据 self.llm is not None and token is not None 判断当前流程该进入哪个处理分支。
            if self.llm is not None and token is not None:
                # 调用 llm.reset_trace_context 完成当前步骤需要的业务处理。
                self.llm.reset_trace_context(token)

    # 定义 _on_timeout 相关的处理逻辑，供流程或外部调用复用。
    def _on_timeout(self, task_id: str, node_name: str) -> None:
        # 节点等待用户太久时自动选择“下一步”，避免诊断流程一直卡住。
        try:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic timeout fired task_id=%s node=%s", task_id, node_name)
            # 调用 self.handle_card_callback_sync 完成当前步骤需要的业务处理。
            self.handle_card_callback_sync(
                # 执行当前业务步骤，推动流程继续向下游推进。
                {"value": {"task_id": task_id, "node_name": node_name, "action": "next"}},
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source="timeout",
            )
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Interactive topic timeout handling failed task_id=%s node=%s", task_id, node_name)

    # 定义 _get_task_lock 相关的处理逻辑，供流程或外部调用复用。
    async def _get_task_lock(self, task_id: str) -> asyncio.Lock:
        # 为每个任务准备一把锁，防止多个按钮回调同时推进同一条流程。
        async with self._task_locks_guard:
            # 将 lock 的值保存下来，供后续流程判断或组装响应时使用。
            lock = self._task_locks.get(task_id)
            # 根据 lock is None 判断当前流程该进入哪个处理分支。
            if lock is None:
                # 将 lock 的值保存下来，供后续流程判断或组装响应时使用。
                lock = asyncio.Lock()
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                self._task_locks[task_id] = lock
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return lock

    # 定义 _route_after_confirm 相关的处理逻辑，供流程或外部调用复用。
    def _route_after_confirm(self, state: TopicFlowState) -> str:
        # 根据用户刚才点的是“重试”还是“下一步”，决定当前节点是否重新执行。
        if state.get("last_action") != "retry":
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "next"
        # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
        current_node = str(state.get("current_node") or "")
        # 将 retry_count 的值保存下来，供后续流程判断或组装响应时使用。
        retry_count = state.get("retry_counts", {}).get(current_node, 0)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "retry" if retry_count < MAX_NODE_RETRIES else "next"

    # 定义 _initial_diagnosis_state 相关的处理逻辑，供流程或外部调用复用。
    def _initial_diagnosis_state(self, *, chat_id: str, root_message_id: str, query: str, task_id: str) -> DiagnosisState:
        # 把飞书用户消息包装成标准诊断状态，让单卡片流程也能复用普通诊断节点。
        trace_id = trace_id_from_parts(chat_id, root_message_id)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "raw_alert": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "source": "feishu-user",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "level": "INFO",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "summary": query,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "details": query,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "raw_text": query,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "trigger_type": "user_message",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tags": ["feishu", "interactive-topic"],
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            "chat_id": chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "thread_root_message_id": root_message_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_thread_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_run_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "messages": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "evidence": [],
        }

    # 定义 _diagnosis_state 相关的处理逻辑，供流程或外部调用复用。
    def _diagnosis_state(self, state: TopicFlowState) -> DiagnosisState:
        # 从单卡片状态中取出标准诊断状态；缺失时用当前任务信息重新补一份。
        diagnosis_state = state.get("diagnosis_state")
        # 根据 diagnosis_state 判断当前流程该进入哪个处理分支。
        if diagnosis_state:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return diagnosis_state
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._initial_diagnosis_state(
            # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
            chat_id=state.get("chat_id", ""),
            # 将 root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            root_message_id=state.get("root_message_id", ""),
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query=state.get("query", ""),
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=state.get("task_id", ""),
        )

    # 定义 _merge_diagnosis_state 相关的处理逻辑，供流程或外部调用复用。
    def _merge_diagnosis_state(self, state: TopicFlowState, update: DiagnosisState) -> DiagnosisState:
        # 把某个节点产出的诊断增量合并回标准诊断状态。
        return {**self._diagnosis_state(state), **update}

    # 定义 _normalize_node_result 相关的处理逻辑，供流程或外部调用复用。
    def _normalize_node_result(self, result: NodeRunResult, state: TopicFlowState) -> tuple[str, DiagnosisState]:
        # 统一节点返回格式，确保每个节点都有展示文本和最新诊断状态。
        if isinstance(result, tuple):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return result
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return result, self._diagnosis_state(state)

    # 定义 _understand 相关的处理逻辑，供流程或外部调用复用。
    async def _understand(self, state: TopicFlowState) -> NodeRunResult:
        # 单卡片流程的第一步，把用户问题理解成后续检索和工具调用可用的告警摘要。
        if self.llm is None:
            # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
            diagnosis_state = self._merge_diagnosis_state(state, {"alert_summary": state.get("query", "")})
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return f"告警理解完成：{state.get('query', '')}", diagnosis_state
        # 将 update 的值保存下来，供后续流程判断或组装响应时使用。
        update = await understand_node(
            # 调用 self._diagnosis_state 完成当前步骤需要的业务处理。
            self._diagnosis_state(state),
            # 将 llm 的值保存下来，供后续流程判断或组装响应时使用。
            llm=self.llm,
            # 将 case_store 的值保存下来，供后续流程判断或组装响应时使用。
            case_store=None,
            # 将 cache_enabled 的值保存下来，供后续流程判断或组装响应时使用。
            cache_enabled=False,
        )
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, update)
        # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
        summary = str(diagnosis_state.get("alert_summary") or state.get("query", ""))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"告警理解完成：\n{summary}", diagnosis_state

    # 定义 _cache_check 相关的处理逻辑，供流程或外部调用复用。
    async def _cache_check(self, state: TopicFlowState) -> NodeRunResult:
        # 查询历史成功案例，看看当前问题是否能直接复用以前的处理经验。
        query = str(self._diagnosis_state(state).get("alert_summary") or state.get("query", ""))
        # 根据 self.case_store is None 判断当前流程该进入哪个处理分支。
        if self.case_store is None:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive cache check skipped task_id=%s reason=no_case_store", state.get("task_id"))
            # 调用 monitor.record_cache_miss 完成当前步骤需要的业务处理。
            monitor.record_cache_miss(state.get("chat_id"))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "缓存检查跳过：历史案例库未配置，请检查 RAG_PROVIDER 和 MILVUS_URI。", self._diagnosis_state(state)
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs = await self.case_store.search_similar_cases(query, top_k=RAG_DISPLAY_TOP_K)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive cache check completed task_id=%s hits=%s top_scores=%s",
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("task_id"),
            # 调用 len 完成当前步骤需要的业务处理。
            len(docs),
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores([doc.score for doc in docs]),
        )
        # 根据 not docs 判断当前流程该进入哪个处理分支。
        if not docs:
            # 调用 monitor.record_cache_miss 完成当前步骤需要的业务处理。
            monitor.record_cache_miss(state.get("chat_id"))
            # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
            diagnosis_state = self._merge_diagnosis_state(state, {"cache_candidates": [], "cache_hit": False})
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return f"缓存检查完成：未命中可复用诊断缓存，继续分析用户问题「{query}」。", diagnosis_state

        # 调用 monitor.record_cache_hit 完成当前步骤需要的业务处理。
        monitor.record_cache_hit(state.get("chat_id"))
        # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
        lines = [f"缓存检查完成：命中 {len(docs)} 条可复用历史成功案例。"]
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for index, doc in enumerate(docs, start=1):
            # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
            metadata = doc.metadata or {}
            # 将 plan 的值保存下来，供后续流程判断或组装响应时使用。
            plan = metadata.get("recommended_plan") or metadata.get("final_text") or ""
            # 将 plan_text 的值保存下来，供后续流程判断或组装响应时使用。
            plan_text = str(plan).replace("\n", " ")[:220] or "-"
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                f"{index}. {doc.title or doc.id} | score={doc.score:.4f}\n"
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                f"   case_id={doc.id}\n"
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                f"   历史方案={plan_text}"
            )
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(
            # 执行当前业务步骤，推动流程继续向下游推进。
            state,
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_candidates": [
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    {"id": doc.id, "title": doc.title, "score": doc.score, "metadata": doc.metadata}
                    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
                    for doc in docs
                ],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cache_hit": True,
            },
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "\n".join(lines), diagnosis_state

    # 定义 _rag_retrieve 相关的处理逻辑，供流程或外部调用复用。
    async def _rag_retrieve(self, state: TopicFlowState) -> NodeRunResult:
        # 从知识库和历史案例中召回背景资料，为后续判断补充上下文。
        query = str(self._diagnosis_state(state).get("alert_summary") or state.get("query", ""))
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive RAG retrieve start task_id=%s query_chars=%s", state.get("task_id"), len(query))
        # 调用 monitor.record_rag_retrieval 完成当前步骤需要的业务处理。
        monitor.record_rag_retrieval("interactive_hybrid", state.get("chat_id"))
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs = await self.retriever.retrieve(query) if self.retriever is not None else []
        # 将 history_docs 的值保存下来，供后续流程判断或组装响应时使用。
        history_docs = await self.case_store.search_similar_cases(query, top_k=RAG_DISPLAY_TOP_K) if self.case_store else []
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive RAG retrieve completed task_id=%s docs=%s history_docs=%s doc_scores=%s history_scores=%s elapsed_ms=%s",
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("task_id"),
            # 调用 len 完成当前步骤需要的业务处理。
            len(docs),
            # 调用 len 完成当前步骤需要的业务处理。
            len(history_docs),
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores([doc.score for doc in docs]),
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores([doc.score for doc in history_docs]),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )

        # 将 retrieved_docs 的值保存下来，供后续流程判断或组装响应时使用。
        retrieved_docs = [doc.to_prompt_text() for doc in docs]
        # 把一组结果合并到集合中，扩展后续可使用的数据范围。
        retrieved_docs.extend(doc.to_prompt_text() for doc in history_docs)
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, {"retrieved_docs": retrieved_docs})

        # 根据 not docs and not history_docs 判断当前流程该进入哪个处理分支。
        if not docs and not history_docs:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "RAG 检索完成：真实向量库未命中相关运维片段或历史案例。", diagnosis_state

        # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
        lines = ["RAG 检索完成："]
        # 根据 docs 判断当前流程该进入哪个处理分支。
        if docs:
            # 将 display_docs 的值保存下来，供后续流程判断或组装响应时使用。
            display_docs = docs[:RAG_DISPLAY_TOP_K]
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(f"\n**文档/混合召回 Top {len(display_docs)} / 共命中 {len(docs)} 条**")
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for index, doc in enumerate(display_docs, start=1):
                # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                metadata = doc.metadata or {}
                # 将 section 的值保存下来，供后续流程判断或组装响应时使用。
                section = metadata.get("section") or metadata.get("h2") or metadata.get("h1") or "-"
                # 将 category 的值保存下来，供后续流程判断或组装响应时使用。
                category = metadata.get("alert_category") or "-"
                # 将 severity 的值保存下来，供后续流程判断或组装响应时使用。
                severity = metadata.get("severity_level") or "-"
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source = doc.source_uri or metadata.get("source") or "-"
                # 将 preview 的值保存下来，供后续流程判断或组装响应时使用。
                preview = doc.text.replace("\n", " ")[:160]
                # 把当前结果追加到集合中，逐步构建最终输出。
                lines.append(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    f"{index}. {doc.title or doc.id} | score={doc.score:.4f} | category={category} | "
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    f"severity={severity}\n   section={section}\n   source={source}\n   {preview}"
                )
        # 处理前面条件都不满足时的默认分支。
        else:
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append("\n**文档/混合召回**：未命中")

        # 根据 history_docs 判断当前流程该进入哪个处理分支。
        if history_docs:
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(f"\n**历史案例召回 Top {len(history_docs)}**")
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for index, doc in enumerate(history_docs, start=1):
                # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                metadata = doc.metadata or {}
                # 将 plan 的值保存下来，供后续流程判断或组装响应时使用。
                plan = metadata.get("recommended_plan") or metadata.get("final_text") or ""
                # 将 plan_text 的值保存下来，供后续流程判断或组装响应时使用。
                plan_text = str(plan).replace("\n", " ")[:220] or "-"
                # 把当前结果追加到集合中，逐步构建最终输出。
                lines.append(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    f"{index}. {doc.title or doc.id} | score={doc.score:.4f}\n"
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    f"   case_id={doc.id}\n"
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    f"   历史方案={plan_text}"
                )
        # 处理前面条件都不满足时的默认分支。
        else:
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append("\n**历史案例召回**：未命中")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "\n".join(lines), diagnosis_state

    # 定义 _handle_feedback_callback 相关的处理逻辑，供流程或外部调用复用。
    async def _handle_feedback_callback(self, task_id: str, action: str, *, source: str) -> dict[str, str]:
        # 处理最终“结果有效/无效”反馈，有效时尝试把本次诊断写入历史案例库。
        if not task_id:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive feedback ignored missing task_id action=%s source=%s", action, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}
        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task = self.task_store.confirm_feedback(task_id, action, source=source)
        # 根据 task is None 判断当前流程该进入哪个处理分支。
        if task is None:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive feedback ignored stale task_id=%s action=%s source=%s", task_id, action, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}

        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive feedback handling start task_id=%s action=%s source=%s", task_id, action, source)
        # 将 app 的值保存下来，供后续流程判断或组装响应时使用。
        app = self.compile()
        # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
        config = {"configurable": {"thread_id": task_id}, "recursion_limit": 50}
        # 将 snapshot 的值保存下来，供后续流程判断或组装响应时使用。
        snapshot = await app.aget_state(config)
        # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
        values = getattr(snapshot, "values", {}) or {}
        # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
        final_text = self._build_final_text(values)
        # 将 saved_case_id 的值保存下来，供后续流程判断或组装响应时使用。
        saved_case_id = ""
        # 根据 action == "feedback_valid" 判断当前流程该进入哪个处理分支。
        if action == "feedback_valid":
            # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
            monitor.record_feedback(True, task.chat_id)
            # 根据 self.case_store is not None 判断当前流程该进入哪个处理分支。
            if self.case_store is not None:
                # 进入可能失败的处理块，便于后续统一捕获和恢复。
                try:
                    # 将 saved_case_id 的值保存下来，供后续流程判断或组装响应时使用。
                    saved_case_id = await self.case_store.save_case_to_history(
                        # 调用 self._build_feedback_state 完成当前步骤需要的业务处理。
                        self._build_feedback_state(task, values, final_text)
                    )
                    # 记录关键运行信息，方便排查流程进展和异常现场。
                    logger.info("Interactive feedback saved case task_id=%s case_id=%s", task_id, saved_case_id)
                # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
                except Exception as exc:
                    # 记录关键运行信息，方便排查流程进展和异常现场。
                    logger.exception("Interactive topic feedback save failed task_id=%s", task_id)
                    # 调用 task_store.reset_feedback_waiting 完成当前步骤需要的业务处理。
                    self.task_store.reset_feedback_waiting(task_id)
                    # 等待异步操作完成，再继续推进当前业务流程。
                    await self.sender.update_workflow_card(
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        task.card_message_id,
                        # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                        task_id=task_id,
                        # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                        query=task.query,
                        # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
                        node_statuses={step.node_name: "done" for step in WORKFLOW_STEPS},
                        # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
                        current_node=None,
                        # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
                        current_result=f"{final_text}\n\n❌ 入库失败：{exc}\n请修复后重试，或选择不存储。",
                        # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
                        buttons_node=None,
                        # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
                        feedback_buttons=True,
                    )
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return {"status": "error", "reason": "save_failed"}
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = f"{final_text}\n\n✅ 已存入历史案例知识库：{saved_case_id}"
            # 处理前面条件都不满足时的默认分支。
            else:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Interactive feedback valid but case_store missing task_id=%s", task_id)
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = f"{final_text}\n\n✅ 已确认有效；历史案例库未配置，未执行入库。"
        # 处理前面条件都不满足时的默认分支。
        else:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive feedback marked invalid task_id=%s", task_id)
            # 调用 monitor.record_feedback 完成当前步骤需要的业务处理。
            monitor.record_feedback(False, task.chat_id)
            # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
            result = f"{final_text}\n\n❌ 已标记为无效，本次诊断结果不入库。"

        # 等待异步操作完成，再继续推进当前业务流程。
        await self.sender.update_workflow_card(
            # 执行当前业务步骤，推动流程继续向下游推进。
            task.card_message_id,
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=task_id,
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query=task.query,
            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
            node_statuses={step.node_name: "done" for step in WORKFLOW_STEPS},
            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
            current_node=None,
            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
            current_result=result,
            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
            buttons_node=None,
            # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
            feedback_buttons=False,
        )
        # 调用 task_store.finish_task 完成当前步骤需要的业务处理。
        self.task_store.finish_task(task_id)
        # 进入异步上下文管理区域，确保异步资源按约定释放。
        async with self._task_locks_guard:
            # 调用 _task_locks.pop 完成当前步骤需要的业务处理。
            self._task_locks.pop(task_id, None)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Interactive feedback handling completed task_id=%s action=%s saved_case_id=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            action,
            # 执行当前业务步骤，推动流程继续向下游推进。
            saved_case_id,
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"status": "ok", "saved_case_id": saved_case_id}

    # 定义 _tool_call 相关的处理逻辑，供流程或外部调用复用。
    async def _tool_call(self, state: TopicFlowState) -> NodeRunResult:
        # 兼容旧版工具调用节点，一次性抓取指标、日志和拓扑等实时数据。
        update = await fetch_live_data_node(self._diagnosis_state(state))
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, update)
        # 将 live_data 的值保存下来，供后续流程判断或组装响应时使用。
        live_data = diagnosis_state.get("live_data", {})
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"工具调用完成：\n{_format_live_data(live_data)}", diagnosis_state


    # 定义 _tool_router 相关的处理逻辑，供流程或外部调用复用。
    async def _tool_router(self, state: TopicFlowState) -> NodeRunResult:
        # 根据当前问题和已有证据决定需要调用哪些只读工具，以及每个工具查什么。
        diagnosis_state = self._diagnosis_state(state)
        # 将 default_plan 的值保存下来，供后续流程判断或组装响应时使用。
        default_plan = _default_tool_plan(state, diagnosis_state)
        # 根据 self.llm is None 判断当前流程该进入哪个处理分支。
        if self.llm is None:
            # 将 tool_plan 的值保存下来，供后续流程判断或组装响应时使用。
            tool_plan = default_plan
        # 处理前面条件都不满足时的默认分支。
        else:
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt = _build_tool_router_prompt(state, diagnosis_state, default_plan)
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
                response = await self.llm.call(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    prompt,
                    # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_name="tool_router",
                    # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_variables={
                        # 调用 state.get 完成当前步骤需要的业务处理。
                        "query": state.get("query", ""),
                        # 调用 state.get 完成当前步骤需要的业务处理。
                        "node_results": state.get("node_results", []),
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "default_plan": default_plan,
                    },
                    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                    metadata={
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "node_name": "tool_router",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "generation_name": "tool_router",
                        # 调用 _allowed_tool_names 完成当前步骤需要的业务处理。
                        "available_tools": _allowed_tool_names(),
                        # 调用 current_trace_context 完成当前步骤需要的业务处理。
                        **current_trace_context(),
                    },
                )
                # 将 tool_plan 的值保存下来，供后续流程判断或组装响应时使用。
                tool_plan = _coerce_tool_plan(response, default_plan)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Interactive tool router LLM failed task_id=%s; using default plan", state.get("task_id"), exc_info=True)
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("tool_router_llm_error")
                # 将 tool_plan 的值保存下来，供后续流程判断或组装响应时使用。
                tool_plan = default_plan
        # 将 tool_plan 的值保存下来，供后续流程判断或组装响应时使用。
        tool_plan = _sanitize_tool_plan(tool_plan)
        # 将 selected_tools 的值保存下来，供后续流程判断或组装响应时使用。
        selected_tools = [str(item.get("tool") or "") for item in tool_plan.get("tool_calls", []) if isinstance(item, dict)]
        # 调用 self._log_agent_event 完成当前步骤需要的业务处理。
        self._log_agent_event(
            # 将 event_type 的值保存下来，供后续流程判断或组装响应时使用。
            event_type="agent_decision",
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event="tool_plan",
            # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
            node_name="tool_router",
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state=state,
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status="planned" if selected_tools else "skipped",
            # 将 selected_tools 的值保存下来，供后续流程判断或组装响应时使用。
            selected_tools=selected_tools,
            # 将 tool_call_count 的值保存下来，供后续流程判断或组装响应时使用。
            tool_call_count=len(selected_tools),
            # 将 reason 的值保存下来，供后续流程判断或组装响应时使用。
            reason=str(tool_plan.get("reason") or ""),
            # 将 need_tools 的值保存下来，供后续流程判断或组装响应时使用。
            need_tools=bool(tool_plan.get("need_tools")),
        )
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, {"tool_plan": tool_plan})
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _format_tool_plan(tool_plan), diagnosis_state

    # 定义 _tool_executor 相关的处理逻辑，供流程或外部调用复用。
    async def _tool_executor(self, state: TopicFlowState) -> NodeRunResult:
        # 按工具路由结果逐个执行 Prometheus、日志或拓扑查询，并汇总成现场证据。
        diagnosis_state = self._diagnosis_state(state)
        # 将 tool_plan 的值保存下来，供后续流程判断或组装响应时使用。
        tool_plan = _sanitize_tool_plan(state.get("tool_plan") or diagnosis_state.get("tool_plan") or {})
        # 根据 not tool_plan.get("need_tools") 判断当前流程该进入哪个处理分支。
        if not tool_plan.get("need_tools"):
            # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
            diagnosis_state = self._merge_diagnosis_state(state, {"tool_results": [], "live_data": {}})
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "Tool execution skipped: tool_router did not request tools.", diagnosis_state

        # 将 provider 的值保存下来，供后续流程判断或组装响应时使用。
        provider = _get_tools_provider()
        # 将 tool_results 的值保存下来，供后续流程判断或组装响应时使用。
        tool_results: list[dict[str, object]] = []
        # 将 live_data 的值保存下来，供后续流程判断或组装响应时使用。
        live_data: dict[str, object] = {}
        # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
        evidence = list(diagnosis_state.get("evidence", []))
        # 将 group_id 的值保存下来，供后续流程判断或组装响应时使用。
        group_id = str(state.get("chat_id") or "")
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for call in tool_plan.get("tool_calls", []):
            # 根据 not isinstance(call, dict) 判断当前流程该进入哪个处理分支。
            if not isinstance(call, dict):
                # 跳过当前项剩余逻辑，继续处理下一项数据。
                continue
            # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
            started = time.perf_counter()
            # 将 tool_name 的值保存下来，供后续流程判断或组装响应时使用。
            tool_name = str(call.get("tool") or "")
            # 将 arguments 的值保存下来，供后续流程判断或组装响应时使用。
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status = "success"
            # 将 error_message 的值保存下来，供后续流程判断或组装响应时使用。
            error_message = ""
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload: object = {}
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
                payload = await _execute_tool_call(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    provider,
                    # 将 tool_name 的值保存下来，供后续流程判断或组装响应时使用。
                    tool_name=tool_name,
                    # 将 arguments 的值保存下来，供后续流程判断或组装响应时使用。
                    arguments=arguments,
                    # 将 alert_summary 的值保存下来，供后续流程判断或组装响应时使用。
                    alert_summary=str(diagnosis_state.get("alert_summary") or state.get("query") or ""),
                    # 将 group_id 的值保存下来，供后续流程判断或组装响应时使用。
                    group_id=group_id,
                )
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                live_data[_live_data_key(tool_name)] = _compact_tool_payload(payload)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as exc:
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status = "failed"
                # 将 error_message 的值保存下来，供后续流程判断或组装响应时使用。
                error_message = _truncate_text(str(exc), 1000)
                # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
                payload = {"error": error_message}
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                live_data[_live_data_key(tool_name)] = _compact_tool_payload(payload)
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("tool_executor_error")
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Interactive tool execution failed task_id=%s tool=%s error=%s", state.get("task_id"), tool_name, exc)
            # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            # 将 result_count 的值保存下来，供后续流程判断或组装响应时使用。
            result_count = _tool_result_count(payload)
            # 将 record 的值保存下来，供后续流程判断或组装响应时使用。
            record = {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tool_name": tool_name,
                # 调用 str 完成当前步骤需要的业务处理。
                "purpose": str(call.get("purpose") or ""),
                # 调用 _redact_tool_arguments 完成当前步骤需要的业务处理。
                "arguments": _redact_tool_arguments(arguments),
                # 执行当前业务步骤，推动流程继续向下游推进。
                "status": status,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "result_count": result_count,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "elapsed_ms": elapsed_ms,
                # 调用 _tool_result_summary 完成当前步骤需要的业务处理。
                "summary": _tool_result_summary(payload),
            }
            # 根据 error_message 判断当前流程该进入哪个处理分支。
            if error_message:
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                record["error_message"] = error_message
            # 把当前结果追加到集合中，逐步构建最终输出。
            tool_results.append(record)
            # 把当前结果追加到集合中，逐步构建最终输出。
            evidence.append(f"Tool {tool_name} {status}: {record['summary']}")
            # 调用 self._log_agent_event 完成当前步骤需要的业务处理。
            self._log_agent_event(
                # 将 event_type 的值保存下来，供后续流程判断或组装响应时使用。
                event_type="tool_call",
                # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
                event="execute",
                # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
                node_name="tool_executor",
                # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
                state=state,
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status=status,
                # 将 tool_name 的值保存下来，供后续流程判断或组装响应时使用。
                tool_name=tool_name,
                # 将 purpose 的值保存下来，供后续流程判断或组装响应时使用。
                purpose=record["purpose"],
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=_tool_query(arguments),
                # 将 result_count 的值保存下来，供后续流程判断或组装响应时使用。
                result_count=result_count,
                # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
                elapsed_ms=elapsed_ms,
                # 将 error_message 的值保存下来，供后续流程判断或组装响应时使用。
                error_message=error_message,
            )
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, {"tool_results": tool_results, "live_data": live_data, "evidence": evidence})
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _format_tool_results(tool_results), diagnosis_state

    # 定义 _evidence_review 相关的处理逻辑，供流程或外部调用复用。
    async def _evidence_review(self, state: TopicFlowState) -> NodeRunResult:
        # 检查当前证据是否足够支撑诊断方案，不足时记录还缺哪些指标或日志。
        diagnosis_state = self._diagnosis_state(state)
        # 将 tool_results 的值保存下来，供后续流程判断或组装响应时使用。
        tool_results = [
            # 调用 in 完成当前步骤需要的业务处理。
            item for item in (state.get("tool_results") or diagnosis_state.get("tool_results") or [])
            # 根据 isinstance(item, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(item, dict)
        ]
        # 将 success_count 的值保存下来，供后续流程判断或组装响应时使用。
        success_count = sum(1 for item in tool_results if item.get("status") == "success")
        # 将 failure_count 的值保存下来，供后续流程判断或组装响应时使用。
        failure_count = sum(1 for item in tool_results if item.get("status") == "failed")
        # 将 evidence_count 的值保存下来，供后续流程判断或组装响应时使用。
        evidence_count = len(diagnosis_state.get("evidence", []))
        # 将 review 的值保存下来，供后续流程判断或组装响应时使用。
        review = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "is_sufficient": success_count > 0 or evidence_count > 0,
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "reason": f"tool_success_count={success_count}, tool_failure_count={failure_count}, evidence_count={evidence_count}",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "missing_evidence": [] if success_count > 0 else ["live_metrics_or_logs"],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tool_success_count": success_count,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tool_failure_count": failure_count,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "evidence_count": evidence_count,
        }
        # 根据 self.llm is not None 判断当前流程该进入哪个处理分支。
        if self.llm is not None:
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt = _build_evidence_review_prompt(state, diagnosis_state, tool_results, review)
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
                response = await self.llm.call(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    prompt,
                    # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_name="evidence_review",
                    # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_variables={
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tool_results": tool_results,
                        # 调用 diagnosis_state.get 完成当前步骤需要的业务处理。
                        "evidence": diagnosis_state.get("evidence", []),
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "default_review": review,
                    },
                    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                    metadata={
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "node_name": "evidence_review",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "generation_name": "evidence_review",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tool_success_count": success_count,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tool_failure_count": failure_count,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "evidence_count": evidence_count,
                        # 调用 current_trace_context 完成当前步骤需要的业务处理。
                        **current_trace_context(),
                    },
                )
                # 将 review 的值保存下来，供后续流程判断或组装响应时使用。
                review = _coerce_evidence_review(response, review)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Interactive evidence review LLM failed task_id=%s; using rule review", state.get("task_id"), exc_info=True)
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("evidence_review_llm_error")
        # 调用 self._log_agent_event 完成当前步骤需要的业务处理。
        self._log_agent_event(
            # 将 event_type 的值保存下来，供后续流程判断或组装响应时使用。
            event_type="evidence_review",
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event="review",
            # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
            node_name="evidence_review",
            # 将 state 的值保存下来，供后续流程判断或组装响应时使用。
            state=state,
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status="sufficient" if review.get("is_sufficient") else "insufficient",
            # 将 is_sufficient 的值保存下来，供后续流程判断或组装响应时使用。
            is_sufficient=bool(review.get("is_sufficient")),
            # 将 reason 的值保存下来，供后续流程判断或组装响应时使用。
            reason=str(review.get("reason") or ""),
            # 将 missing_evidence 的值保存下来，供后续流程判断或组装响应时使用。
            missing_evidence=review.get("missing_evidence") or [],
            # 将 tool_success_count 的值保存下来，供后续流程判断或组装响应时使用。
            tool_success_count=int(review.get("tool_success_count") or success_count),
            # 将 tool_failure_count 的值保存下来，供后续流程判断或组装响应时使用。
            tool_failure_count=int(review.get("tool_failure_count") or failure_count),
            # 将 evidence_count 的值保存下来，供后续流程判断或组装响应时使用。
            evidence_count=int(review.get("evidence_count") or evidence_count),
        )
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, {"evidence_review": review})
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _format_evidence_review(review), diagnosis_state

    # 定义 _generate_plan 相关的处理逻辑，供流程或外部调用复用。
    async def _generate_plan(self, state: TopicFlowState) -> NodeRunResult:
        # 基于摘要、RAG、工具结果和证据审查生成处理方案，模型失败时给出保守兜底方案。
        if self.llm is None:
            # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
            diagnosis_state = self._merge_diagnosis_state(
                # 执行当前业务步骤，推动流程继续向下游推进。
                state,
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "recommended_plan": {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "summary": "LLM 未配置，无法生成真实诊断方案。",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "actions": ["检查 LLM 配置后重试。"],
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "risk_level": "unknown",
                    },
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "need_human": True,
                },
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "方案生成跳过：LLM 未配置。", diagnosis_state
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 update 的值保存下来，供后续流程判断或组装响应时使用。
            update = await generate_plan_node(self._diagnosis_state(state), self.llm)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive generate_plan LLM failed task_id=%s; using evidence-aware fallback plan",
                # 调用 state.get 完成当前步骤需要的业务处理。
                state.get("task_id"),
                # 将 exc_info 的值保存下来，供后续流程判断或组装响应时使用。
                exc_info=True,
            )
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("interactive_generate_plan_llm_error")
            # 将 update 的值保存下来，供后续流程判断或组装响应时使用。
            update = _fallback_plan_update(self._diagnosis_state(state), exc)
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, update)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _format_plan_result(diagnosis_state.get("recommended_plan", {}), diagnosis_state.get("evidence", [])), diagnosis_state

    # 定义 _validate 相关的处理逻辑，供流程或外部调用复用。
    async def _validate(self, state: TopicFlowState) -> NodeRunResult:
        # 对生成的处理方案做安全校验，避免把高风险操作直接展示为可执行建议。
        if self.llm is None:
            # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
            diagnosis_state = self._merge_diagnosis_state(state, {"validation_result": False})
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "方案校验跳过：LLM 未配置。", diagnosis_state
        # 将 update 的值保存下来，供后续流程判断或组装响应时使用。
        update = await validate_plan_node(self._diagnosis_state(state), self.llm)
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state = self._merge_diagnosis_state(state, update)
        # 将 validation_result 的值保存下来，供后续流程判断或组装响应时使用。
        validation_result = bool(diagnosis_state.get("validation_result"))
        # 将 attempts 的值保存下来，供后续流程判断或组装响应时使用。
        attempts = diagnosis_state.get("validation_attempts", 0)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return (
            # 执行当前业务步骤，推动流程继续向下游推进。
            "方案校验完成："
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"{'通过' if validation_result else '未通过'}，"
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"校验次数={attempts}。"
        # 执行当前业务步骤，推动流程继续向下游推进。
        ), diagnosis_state

    # 定义 _summary 相关的处理逻辑，供流程或外部调用复用。
    async def _summary(self, state: TopicFlowState) -> str:
        # 把前面所有节点的结果压缩成飞书卡片上最终展示的中文诊断结论。
        if self.llm is not None:
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt = _build_summary_prompt(state)
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Interactive summary LLM start task_id=%s prompt_chars=%s",
                    # 调用 state.get 完成当前步骤需要的业务处理。
                    state.get("task_id"),
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(prompt),
                )
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary = (
                    # 等待异步操作完成，再继续推进当前业务流程。
                    await self.llm.call(
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        prompt,
                        # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
                        prompt_name="interactive_summary",
                        # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
                        prompt_variables={
                            # 调用 state.get 完成当前步骤需要的业务处理。
                            "query": state.get("query", ""),
                            # 调用 state.get 完成当前步骤需要的业务处理。
                            "node_results": state.get("node_results", []),
                        },
                        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                        metadata={"node_name": "interactive_summary", **current_trace_context()},
                    )
                # 调用 strip 完成当前步骤需要的业务处理。
                ).strip()
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Interactive summary LLM completed task_id=%s summary_chars=%s",
                    # 调用 state.get 完成当前步骤需要的业务处理。
                    state.get("task_id"),
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(summary),
                )
                # 根据 summary 判断当前流程该进入哪个处理分支。
                if summary:
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return summary
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.exception("Interactive summary LLM failed task_id=%s", state.get("task_id"))
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("interactive_summary_llm_error")
        # 等待异步操作完成，再继续推进当前业务流程。
        await asyncio.sleep(0.1)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "总结完成：建议先限流止血，扩容消费者，检查数据库慢查询，并持续观察错误率回落。"

    # 定义 _build_final_text 相关的处理逻辑，供流程或外部调用复用。
    def _build_final_text(self, state: dict[str, Any]) -> str:
        # 从流程状态中取出最终总结；没有总结时用最后一个节点结果兜底展示。
        results = state.get("node_results", [])
        # 将 last_result 的值保存下来，供后续流程判断或组装响应时使用。
        last_result = ""
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for item in results:
            # 根据 not isinstance(item, dict) 判断当前流程该进入哪个处理分支。
            if not isinstance(item, dict):
                # 跳过当前项剩余逻辑，继续处理下一项数据。
                continue
            # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
            result = str(item.get("result") or "")
            # 根据 result 判断当前流程该进入哪个处理分支。
            if result:
                # 将 last_result 的值保存下来，供后续流程判断或组装响应时使用。
                last_result = result
            # 根据 item.get("node_name") == "summary" and result 判断当前流程该进入哪个处理分支。
            if item.get("node_name") == "summary" and result:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return result
        # 根据 last_result 判断当前流程该进入哪个处理分支。
        if last_result:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return last_result
        # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
        lines = ["LangGraph 交互式话题流程执行完毕。"]
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for item in results:
            # 根据 isinstance(item, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(item, dict):
                # 把当前结果追加到集合中，逐步构建最终输出。
                lines.append(f"- {item.get('node_name')}: {item.get('result')}")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "\n".join(lines)

    # 定义 _trace_id_for_input 相关的处理逻辑，供流程或外部调用复用。
    def _trace_id_for_input(self, task_id: str, graph_input: TopicFlowState | Command) -> str:
        # 为单卡片任务生成追踪 ID，方便日志、模型调用和节点事件互相关联。
        if isinstance(graph_input, dict):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return str(
                # 调用 graph_input.get 完成当前步骤需要的业务处理。
                graph_input.get("trace_id")
                # 调用 trace_id_from_parts 完成当前步骤需要的业务处理。
                or trace_id_from_parts(graph_input.get("chat_id"), graph_input.get("root_message_id"))
            )
        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task = self.task_store.get_task(task_id)
        # 根据 task is not None 判断当前流程该进入哪个处理分支。
        if task is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return trace_id_from_parts(task.chat_id, task.root_message_id)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return task_id

    # 定义 _langfuse_metadata 相关的处理逻辑，供流程或外部调用复用。
    def _langfuse_metadata(self, task_id: str, graph_input: TopicFlowState | Command) -> dict[str, object]:
        # 整理写入 Langfuse 的任务元数据，让单卡片流程里的模型调用能串成一条链。
        if isinstance(graph_input, dict):
            # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
            chat_id = graph_input.get("chat_id")
            # 将 root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            root_message_id = graph_input.get("root_message_id")
            # 将 trace_id 的值保存下来，供后续流程判断或组装响应时使用。
            trace_id = self._trace_id_for_input(task_id, graph_input)
        # 处理前面条件都不满足时的默认分支。
        else:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self.task_store.get_task(task_id)
            # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
            chat_id = task.chat_id if task else None
            # 将 root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            root_message_id = task.root_message_id if task else None
            # 将 trace_id 的值保存下来，供后续流程判断或组装响应时使用。
            trace_id = trace_id_from_parts(chat_id, root_message_id) if task else task_id
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "business_trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "trace_name": "interactive_topic",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "task_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_thread_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_run_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "chat_id": chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "thread_root_message_id": root_message_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_type": "interactive_topic",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tags": ["agent-sentinel", "interactive-topic", "feishu"],
        }

    # 定义 _log_node_event 相关的处理逻辑，供流程或外部调用复用。
    def _log_node_event(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        event: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        state: TopicFlowState,
        # 执行当前业务步骤，推动流程继续向下游推进。
        status: str,
        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task: Any | None = None,
        # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
        elapsed_ms: int | None = None,
        # 将 error 的值保存下来，供后续流程判断或组装响应时使用。
        error: BaseException | None = None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        **extra: object,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 记录节点级 JSON 事件，方便从日志系统还原每个节点的执行状态。
        trace_id = str(
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("trace_id")
            # 调用 trace_id_from_parts 完成当前步骤需要的业务处理。
            or trace_id_from_parts(state.get("chat_id"), state.get("root_message_id"))
            # 执行当前业务步骤，推动流程继续向下游推进。
            or ""
        )
        # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
        task_id = str(state.get("task_id") or "")
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload: dict[str, object] = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "event_type": "node_event",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_type": "interactive_topic",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "event": event,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "status": status,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "business_trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "task_id": task_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "node": node_name,
            # 调用 str 完成当前步骤需要的业务处理。
            "chat_id": str(state.get("chat_id") or (task.chat_id if task else "") or ""),
            # 调用 str 完成当前步骤需要的业务处理。
            "root_message_id": str(state.get("root_message_id") or (task.root_message_id if task else "") or ""),
            # 调用 str 完成当前步骤需要的业务处理。
            "card_message_id": str((task.card_message_id if task else None) or ""),
            # 调用 int 完成当前步骤需要的业务处理。
            "retry_count": int(state.get("retry_counts", {}).get(node_name, 0)),
            # 调用 len 完成当前步骤需要的业务处理。
            "previous_results": len(state.get("node_results", [])),
        }
        # 根据 elapsed_ms is not None 判断当前流程该进入哪个处理分支。
        if elapsed_ms is not None:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            payload["elapsed_ms"] = elapsed_ms
        # 根据 error is not None 判断当前流程该进入哪个处理分支。
        if error is not None:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            payload["error_type"] = type(error).__name__
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            payload["error_message"] = str(error)
        # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
        payload.update({key: value for key, value in extra.items() if value is not None})
        # 将 payload_json 的值保存下来，供后续流程判断或组装响应时使用。
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("node_event %s", payload_json)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        node_event_json_logger.info(payload_json)

    # 定义 _log_agent_event 相关的处理逻辑，供流程或外部调用复用。
    def _log_agent_event(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        event_type: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        event: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        state: TopicFlowState,
        # 执行当前业务步骤，推动流程继续向下游推进。
        status: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        **extra: object,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 记录智能体决策或工具调用事件，用于后续审计“为什么调用了某个工具”。
        trace_id = str(
            # 调用 state.get 完成当前步骤需要的业务处理。
            state.get("trace_id")
            # 调用 trace_id_from_parts 完成当前步骤需要的业务处理。
            or trace_id_from_parts(state.get("chat_id"), state.get("root_message_id"))
            # 执行当前业务步骤，推动流程继续向下游推进。
            or ""
        )
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload: dict[str, object] = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "event_type": event_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "workflow_type": "interactive_topic",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "event": event,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "status": status,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "business_trace_id": trace_id,
            # 调用 str 完成当前步骤需要的业务处理。
            "task_id": str(state.get("task_id") or ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "node": node_name,
            # 调用 str 完成当前步骤需要的业务处理。
            "chat_id": str(state.get("chat_id") or ""),
            # 调用 str 完成当前步骤需要的业务处理。
            "root_message_id": str(state.get("root_message_id") or ""),
        }
        # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
        payload.update({key: value for key, value in extra.items() if value is not None})
        # 将 payload_json 的值保存下来，供后续流程判断或组装响应时使用。
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("%s %s", event_type, payload_json)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        node_event_json_logger.info(payload_json)

    # 定义 _build_feedback_state 相关的处理逻辑，供流程或外部调用复用。
    def _build_feedback_state(self, task: Any, values: dict[str, Any], final_text: str) -> dict[str, Any]:
        # 把一次有效的单卡片诊断整理成历史案例入库需要的结构。
        node_results = values.get("node_results", [])
        # 将 result_by_node 的值保存下来，供后续流程判断或组装响应时使用。
        result_by_node = {
            # 调用 str 完成当前步骤需要的业务处理。
            str(item.get("node_name")): str(item.get("result") or "")
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for item in node_results
            # 根据 isinstance(item, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(item, dict)
        }
        # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
        evidence = [str(item.get("result")) for item in node_results if isinstance(item, dict) and item.get("result")]
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "raw_alert": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "source": "interactive-topic",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "summary": task.query,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tags": ["interactive-topic"],
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            "alert_summary": task.query,
            # 调用 result_by_node.get 完成当前步骤需要的业务处理。
            "retrieved_docs": [result_by_node.get("rag_retrieve", "")],
            # 调用 result_by_node.get 完成当前步骤需要的业务处理。
            "live_data": {"tool_call": result_by_node.get("tool_call", "")},
            # 调用 result_by_node.get 完成当前步骤需要的业务处理。
            "recommended_plan": {"summary": result_by_node.get("summary") or final_text},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "evidence": evidence,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "validation_result": True,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "need_human": False,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "human_decision": "feedback_valid",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "final_text": final_text,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "messages": [{"role": "assistant", "content": final_text}],
        }

    # 定义 _node_results 相关的处理逻辑，供流程或外部调用复用。
    def _node_results(self, task_id: str) -> list[dict[str, str]]:
        # 读取某个任务已经完成的节点结果，用于卡片重放或最终反馈处理。
        app = self.compile()
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 snapshot 的值保存下来，供后续流程判断或组装响应时使用。
            snapshot = app.get_state({"configurable": {"thread_id": task_id}, "recursion_limit": 50})
            # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
            values = getattr(snapshot, "values", {}) or {}
            # 将 results 的值保存下来，供后续流程判断或组装响应时使用。
            results = values.get("node_results")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return results if isinstance(results, list) else []
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.debug("Failed to load node results task_id=%s", task_id, exc_info=True)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []

    # 定义 _confirmed_replay_result 相关的处理逻辑，供流程或外部调用复用。
    def _confirmed_replay_result(self, task: Any, node_name: str) -> str | None:
        # 当用户确认当前节点后，复用该节点的展示结果，避免重复执行同一步。
        if task is None:
            return None
        # 根据 task.current_node != node_name 判断当前流程该进入哪个处理分支。
        if task.current_node != node_name:
            return None
        # 根据 task.status not in {"confirmed", "retrying"} 判断当前流程该进入哪个处理分支。
        if task.status not in {"confirmed", "retrying"}:
            return None
        # 根据 task.pending_node_result is None 判断当前流程该进入哪个处理分支。
        if task.pending_node_result is None:
            return None
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return str(task.pending_node_result)

    # 定义 _node_statuses_until 相关的处理逻辑，供流程或外部调用复用。
    def _node_statuses_until(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        current_node: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        current_status: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_results: list[dict[str, str]],
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, str]:
        # 根据已完成节点和当前节点，计算飞书卡片里每个步骤该显示的状态。
        completed = {
            # 调用 str 完成当前步骤需要的业务处理。
            str(item.get("node_name"))
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for item in node_results
            # 根据 isinstance(item, dict) and item.get("node_name") 判断当前流程该进入哪个处理分支。
            if isinstance(item, dict) and item.get("node_name")
        }
        # 将 statuses 的值保存下来，供后续流程判断或组装响应时使用。
        statuses: dict[str, str] = {}
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for step in WORKFLOW_STEPS:
            # 根据 step.node_name in completed 判断当前流程该进入哪个处理分支。
            if step.node_name in completed:
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                statuses[step.node_name] = "done"
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        statuses[current_node] = current_status
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return statuses

    # 定义 _extract_card_value 相关的处理逻辑，供流程或外部调用复用。
    def _extract_card_value(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 兼容飞书不同回调结构，从 payload 中取出真正的按钮 value。
        action = payload.get("action")
        # 根据 isinstance(action, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(action, dict):
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = action.get("value")
            # 根据 isinstance(value, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(value, dict):
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return value
        # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
        event = payload.get("event")
        # 根据 isinstance(event, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(event, dict):
            # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
            action = event.get("action")
            # 根据 isinstance(action, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(action, dict):
                # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
                value = action.get("value")
                # 根据 isinstance(value, dict) 判断当前流程该进入哪个处理分支。
                if isinstance(value, dict):
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return value
        # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
        value = payload.get("value")
        # 根据 isinstance(value, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(value, dict):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return value
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return payload

    # 定义 _submit 相关的处理逻辑，供流程或外部调用复用。
    def _submit(self, coro: Awaitable[Any]) -> concurrent.futures.Future[Any]:
        # 把协程提交到单卡片流程自己的后台事件循环执行。
        loop = self._ensure_background_loop()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return asyncio.run_coroutine_threadsafe(coro, loop)

    # 定义 _ensure_background_loop 相关的处理逻辑，供流程或外部调用复用。
    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        # 确保后台事件循环已经启动，用来串行管理单卡片任务。
        with self._loop_guard:
            # 根据 self._loop is not None and self._loop.is_running() 判断当前流程该进入哪个处理分支。
            if self._loop is not None and self._loop.is_running():
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return self._loop
            # 调用 _loop_started.clear 完成当前步骤需要的业务处理。
            self._loop_started.clear()
            # 将 self._loop_thread 的值保存下来，供后续流程判断或组装响应时使用。
            self._loop_thread = threading.Thread(
                # 将 target 的值保存下来，供后续流程判断或组装响应时使用。
                target=self._run_background_loop,
                # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
                name="interactive-topic-loop",
                # 将 daemon 的值保存下来，供后续流程判断或组装响应时使用。
                daemon=True,
            )
            # 调用 _loop_thread.start 完成当前步骤需要的业务处理。
            self._loop_thread.start()
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        self._loop_started.wait(timeout=5)
        # 根据 self._loop is None 判断当前流程该进入哪个处理分支。
        if self._loop is None:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("Interactive topic background loop failed to start.")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._loop

    # 定义 _run_background_loop 相关的处理逻辑，供流程或外部调用复用。
    def _run_background_loop(self) -> None:
        # 在线程里启动 asyncio 事件循环，让同步回调也能安全驱动异步工作流。
        loop = asyncio.new_event_loop()
        # 调用 asyncio.set_event_loop 完成当前步骤需要的业务处理。
        asyncio.set_event_loop(loop)
        # 将 self._loop 的值保存下来，供后续流程判断或组装响应时使用。
        self._loop = loop
        # 调用 _loop_started.set 完成当前步骤需要的业务处理。
        self._loop_started.set()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Interactive topic background event loop started")
        # 调用 loop.run_forever 完成当前步骤需要的业务处理。
        loop.run_forever()


# 定义 _node_title 相关的处理逻辑，供流程或外部调用复用。
def _node_title(node_name: str) -> str:
    # 把内部节点名转换成飞书卡片上更容易理解的步骤标题。
    for step in WORKFLOW_STEPS:
        # 根据 step.node_name == node_name 判断当前流程该进入哪个处理分支。
        if step.node_name == node_name:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return step.title
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return node_name


# 定义 _format_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 把相似度分数格式化成短文本，方便写入日志和卡片。
    if not scores:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + (", ..." if len(scores) > limit else "") + "]"


# 定义 _allowed_tool_names 相关的处理逻辑，供流程或外部调用复用。
def _allowed_tool_names() -> list[str]:
    # 列出当前允许模型选择的只读工具，防止它请求未知或危险工具。
    return ["prometheus_query_metrics", "aliyun_sls_query_logs", "topology_query"]


# 定义 _default_tool_plan 相关的处理逻辑，供流程或外部调用复用。
def _default_tool_plan(state: TopicFlowState, diagnosis_state: DiagnosisState) -> dict[str, object]:
    # 当模型不可用或判断失败时，给出默认的安全工具查询计划。
    trace_id = str(state.get("trace_id") or trace_id_from_parts(state.get("chat_id"), state.get("root_message_id")) or "")
    # 将 alert_summary 的值保存下来，供后续流程判断或组装响应时使用。
    alert_summary = str(diagnosis_state.get("alert_summary") or state.get("query") or "")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "need_tools": True,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reason": "Default controlled ReAct plan: query metrics and logs before generating a plan.",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "tool_calls": [
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tool": "prometheus_query_metrics",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "purpose": "Check Agent-Sentinel service and workflow metrics.",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "arguments": {"promql": "up"},
            },
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tool": "aliyun_sls_query_logs",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "purpose": "Query structured node events for the current business trace.",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "arguments": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "query": f'event_type: node_event AND business_trace_id: "{trace_id}"',
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "alert_summary": alert_summary,
                },
            },
        ],
    }


# 定义 _build_tool_router_prompt 相关的处理逻辑，供流程或外部调用复用。
def _build_tool_router_prompt(state: TopicFlowState, diagnosis_state: DiagnosisState, default_plan: dict[str, object]) -> str:
    # 生成工具路由提示词，告诉模型只能选择哪些工具以及参数格式。
    context = {
        # 调用 state.get 完成当前步骤需要的业务处理。
        "query": state.get("query", ""),
        # 调用 state.get 完成当前步骤需要的业务处理。
        "business_trace_id": state.get("trace_id") or trace_id_from_parts(state.get("chat_id"), state.get("root_message_id")),
        # 调用 diagnosis_state.get 完成当前步骤需要的业务处理。
        "alert_summary": diagnosis_state.get("alert_summary"),
        # 调用 state.get 完成当前步骤需要的业务处理。
        "node_results": state.get("node_results", []),
        # 调用 _allowed_tool_names 完成当前步骤需要的业务处理。
        "available_tools": _allowed_tool_names(),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "default_plan": default_plan,
    }
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return (
        # 执行当前业务步骤，推动流程继续向下游推进。
        "You are the tool router for an AIOps alert investigation.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Return JSON only. Choose at most 3 read-only tools.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Allowed tools:\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- prometheus_query_metrics: query Prometheus with promql or alert_summary.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- aliyun_sls_query_logs: query SLS logs with query or alert_summary.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- topology_query: inspect service dependency topology.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Rules:\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- For prometheus_query_metrics, pass exactly one valid PromQL expression per tool call; never comma-separate multiple expressions.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- For aliyun_sls_query_logs, prefer indexed Agent-Sentinel fields such as business_trace_id, event_type, node, status, tool_name, and query.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- Keep query strings concise and read-only.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Schema: {\"need_tools\": boolean, \"reason\": string, "
        # 执行当前业务步骤，推动流程继续向下游推进。
        "\"tool_calls\": [{\"tool\": string, \"purpose\": string, \"arguments\": object}]}.\n"
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        f"Context:\n{json.dumps(context, ensure_ascii=False, default=str)}"
    )


# 定义 _build_evidence_review_prompt 相关的处理逻辑，供流程或外部调用复用。
def _build_evidence_review_prompt(
    # 执行当前业务步骤，推动流程继续向下游推进。
    state: TopicFlowState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    diagnosis_state: DiagnosisState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    tool_results: list[dict[str, object]],
    # 执行当前业务步骤，推动流程继续向下游推进。
    default_review: dict[str, object],
# 执行当前业务步骤，推动流程继续向下游推进。
) -> str:
    # 生成证据审查提示词，让模型判断当前证据是否足够支撑方案。
    context = {
        # 调用 state.get 完成当前步骤需要的业务处理。
        "query": state.get("query", ""),
        # 调用 state.get 完成当前步骤需要的业务处理。
        "business_trace_id": state.get("trace_id") or trace_id_from_parts(state.get("chat_id"), state.get("root_message_id")),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "tool_results": tool_results,
        # 调用 diagnosis_state.get 完成当前步骤需要的业务处理。
        "evidence": diagnosis_state.get("evidence", []),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "default_review": default_review,
    }
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return (
        # 执行当前业务步骤，推动流程继续向下游推进。
        "You are reviewing evidence for an AIOps alert investigation.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Return JSON only. Decide if evidence is sufficient for a safe diagnosis plan.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Schema: {\"is_sufficient\": boolean, \"reason\": string, "
        # 执行当前业务步骤，推动流程继续向下游推进。
        "\"missing_evidence\": [string], \"tool_success_count\": integer, "
        # 执行当前业务步骤，推动流程继续向下游推进。
        "\"tool_failure_count\": integer, \"evidence_count\": integer}.\n"
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        f"Context:\n{json.dumps(context, ensure_ascii=False, default=str)}"
    )


# 定义 _coerce_tool_plan 相关的处理逻辑，供流程或外部调用复用。
def _coerce_tool_plan(response: object, default_plan: dict[str, object]) -> dict[str, object]:
    # 把模型返回内容解析成工具计划；解析失败时使用默认计划。
    parsed = _extract_json_object(response)
    # 根据 not isinstance(parsed, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(parsed, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default_plan
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return parsed


# 定义 _coerce_evidence_review 相关的处理逻辑，供流程或外部调用复用。
def _coerce_evidence_review(response: object, default_review: dict[str, object]) -> dict[str, object]:
    # 把模型返回内容解析成证据审查结果，并补齐默认字段。
    parsed = _extract_json_object(response)
    # 根据 not isinstance(parsed, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(parsed, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default_review
    # 将 review 的值保存下来，供后续流程判断或组装响应时使用。
    review = dict(default_review)
    # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
    review.update(parsed)
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    review["is_sufficient"] = bool(review.get("is_sufficient"))
    # 将 missing 的值保存下来，供后续流程判断或组装响应时使用。
    missing = review.get("missing_evidence")
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    review["missing_evidence"] = missing if isinstance(missing, list) else []
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return review


# 定义 _extract_json_object 相关的处理逻辑，供流程或外部调用复用。
def _extract_json_object(value: object) -> dict[str, object] | None:
    # 从模型文本里提取 JSON 对象，兼容模型在 JSON 外多说了几句话的情况。
    if isinstance(value, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return value
    # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
    text = str(value or "").strip()
    # 根据 not text 判断当前流程该进入哪个处理分支。
    if not text:
        return None
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = json.loads(text)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except json.JSONDecodeError:
        # 将 start 的值保存下来，供后续流程判断或组装响应时使用。
        start = text.find("{")
        # 将 end 的值保存下来，供后续流程判断或组装响应时使用。
        end = text.rfind("}")
        # 根据 start < 0 or end <= start 判断当前流程该进入哪个处理分支。
        if start < 0 or end <= start:
            return None
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
            data = json.loads(text[start : end + 1])
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except json.JSONDecodeError:
            return None
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return data if isinstance(data, dict) else None


# 定义 _sanitize_tool_plan 相关的处理逻辑，供流程或外部调用复用。
def _sanitize_tool_plan(plan: object) -> dict[str, object]:
    # 清洗工具计划，只保留白名单工具和安全参数，最多执行有限数量的调用。
    if not isinstance(plan, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"need_tools": False, "reason": "Invalid tool plan.", "tool_calls": []}
    # 将 allowed 的值保存下来，供后续流程判断或组装响应时使用。
    allowed = set(_allowed_tool_names())
    # 将 calls 的值保存下来，供后续流程判断或组装响应时使用。
    calls = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for item in plan.get("tool_calls", []):
        # 根据 not isinstance(item, dict) 判断当前流程该进入哪个处理分支。
        if not isinstance(item, dict):
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue
        # 将 tool 的值保存下来，供后续流程判断或组装响应时使用。
        tool = str(item.get("tool") or "")
        # 根据 tool not in allowed 判断当前流程该进入哪个处理分支。
        if tool not in allowed:
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue
        # 将 arguments 的值保存下来，供后续流程判断或组装响应时使用。
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        # 把当前结果追加到集合中，逐步构建最终输出。
        calls.append(
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tool": tool,
                # 调用 str 完成当前步骤需要的业务处理。
                "purpose": str(item.get("purpose") or ""),
                # 调用 _redact_tool_arguments 完成当前步骤需要的业务处理。
                "arguments": _redact_tool_arguments(arguments),
            }
        )
        # 根据 len(calls) >= MAX_ROUTED_TOOL_CALLS 判断当前流程该进入哪个处理分支。
        if len(calls) >= MAX_ROUTED_TOOL_CALLS:
            # 满足停止条件后退出循环，避免继续执行无效处理。
            break
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 调用 bool 完成当前步骤需要的业务处理。
        "need_tools": bool(plan.get("need_tools", bool(calls))) and bool(calls),
        # 调用 str 完成当前步骤需要的业务处理。
        "reason": str(plan.get("reason") or ""),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "tool_calls": calls,
    }


# 定义 _execute_tool_call 相关的处理逻辑，供流程或外部调用复用。
async def _execute_tool_call(
    # 执行当前业务步骤，推动流程继续向下游推进。
    provider: Any,
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    tool_name: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    arguments: dict[str, object],
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_summary: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    group_id: str | None,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> object:
    # 根据工具名分发到对应 provider，实际执行指标、日志或拓扑查询。
    if tool_name == "prometheus_query_metrics":
        # 将 metrics 的值保存下来，供后续流程判断或组装响应时使用。
        metrics = getattr(provider, "metrics")
        # 根据 "promql" in arguments and hasattr(metrics, "_client"... 判断当前流程该进入哪个处理分支。
        if "promql" in arguments and hasattr(metrics, "_client") and hasattr(metrics, "_config"):
            # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
            result = await metrics._client.call_tool(  # type: ignore[attr-defined]
                # 执行当前业务步骤，推动流程继续向下游推进。
                metrics._config.prometheus_tool,  # type: ignore[attr-defined]
                {
                    # 调用 str 完成当前步骤需要的业务处理。
                    "promql": str(arguments.get("promql") or ""),
                    # 调用 getattr 完成当前步骤需要的业务处理。
                    "prometheus_base_url": getattr(metrics._config, "prometheus_base_url", None),  # type: ignore[attr-defined]
                    # 调用 int 完成当前步骤需要的业务处理。
                    "timeout_seconds": int(arguments.get("timeout_seconds") or 10),
                },
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return _extract_mcp_json(result)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await metrics.get_metrics(str(arguments.get("alert_summary") or alert_summary), group_id=group_id)
    # 根据 tool_name == "aliyun_sls_query_logs" 判断当前流程该进入哪个处理分支。
    if tool_name == "aliyun_sls_query_logs":
        # 将 logs 的值保存下来，供后续流程判断或组装响应时使用。
        logs = getattr(provider, "logs")
        # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
        summary = str(arguments.get("query") or arguments.get("alert_summary") or alert_summary)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await logs.query_logs(summary, group_id=group_id)
    # 根据 tool_name == "topology_query" 判断当前流程该进入哪个处理分支。
    if tool_name == "topology_query":
        # 将 topology 的值保存下来，供后续流程判断或组装响应时使用。
        topology = getattr(provider, "topology")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await topology.get_topology(str(arguments.get("service") or alert_summary), group_id=group_id)
    # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
    raise ValueError(f"Unsupported tool: {tool_name}")


# 定义 _extract_mcp_json 相关的处理逻辑，供流程或外部调用复用。
def _extract_mcp_json(result: object) -> object:
    # 从 MCP 工具返回结构里取出 JSON 内容，其他格式则原样返回。
    if not isinstance(result, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return result
    # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
    content = result.get("content")
    # 根据 isinstance(content, list) and content and isinstance... 判断当前流程该进入哪个处理分支。
    if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "json":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return content[0].get("json")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return result


# 定义 _redact_tool_arguments 相关的处理逻辑，供流程或外部调用复用。
def _redact_tool_arguments(arguments: object) -> dict[str, object]:
    # 记录工具参数前先脱敏，避免密钥、令牌、密码进入日志。
    if not isinstance(arguments, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {}
    # 将 redacted 的值保存下来，供后续流程判断或组装响应时使用。
    redacted: dict[str, object] = {}
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for key, value in arguments.items():
        # 根据 any(secret in str(key).lower() for secret in ("key",... 判断当前流程该进入哪个处理分支。
        if any(secret in str(key).lower() for secret in ("key", "secret", "token", "password")):
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            redacted[str(key)] = "***"
        # 根据 isinstance(value, str) 判断当前流程该进入哪个处理分支。
        elif isinstance(value, str):
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            redacted[str(key)] = value[:500]
        # 处理前面条件都不满足时的默认分支。
        else:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            redacted[str(key)] = value
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return redacted


# 定义 _tool_query 相关的处理逻辑，供流程或外部调用复用。
def _tool_query(arguments: object) -> str:
    # 从工具参数中提取最核心的查询语句，用于日志和审计展示。
    if not isinstance(arguments, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return str(arguments.get("promql") or arguments.get("query") or arguments.get("alert_summary") or "")[:500]


# 定义 _live_data_key 相关的处理逻辑，供流程或外部调用复用。
def _live_data_key(tool_name: str) -> str:
    # 把工具名映射成 live_data 里的字段名，例如 metrics、logs、topology。
    if tool_name == "prometheus_query_metrics":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "metrics"
    # 根据 tool_name == "aliyun_sls_query_logs" 判断当前流程该进入哪个处理分支。
    if tool_name == "aliyun_sls_query_logs":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "logs"
    # 根据 tool_name == "topology_query" 判断当前流程该进入哪个处理分支。
    if tool_name == "topology_query":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "topology"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return tool_name


# 定义 _tool_result_count 相关的处理逻辑，供流程或外部调用复用。
def _tool_result_count(payload: object) -> int:
    # 估算工具返回了多少条结果，便于判断查询是否真的命中数据。
    if isinstance(payload, dict):
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for key in ("series", "matches", "dependencies"):
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = payload.get(key)
            # 根据 isinstance(value, list) 判断当前流程该进入哪个处理分支。
            if isinstance(value, list):
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return len(value)
        # 将 total 的值保存下来，供后续流程判断或组装响应时使用。
        total = payload.get("total")
        # 根据 isinstance(total, int) 判断当前流程该进入哪个处理分支。
        if isinstance(total, int):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return total
    # 根据 isinstance(payload, list) 判断当前流程该进入哪个处理分支。
    if isinstance(payload, list):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return len(payload)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return 0


# 定义 _tool_result_summary 相关的处理逻辑，供流程或外部调用复用。
def _tool_result_summary(payload: object) -> str:
    # 把工具返回内容压缩成一行摘要，避免卡片和日志被大段原始数据淹没。
    if isinstance(payload, dict):
        # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
        summary = payload.get("summary")
        # 根据 summary 判断当前流程该进入哪个处理分支。
        if summary:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return str(summary)[:500]
        # 将 error 的值保存下来，供后续流程判断或组装响应时使用。
        error = payload.get("error")
        # 根据 error 判断当前流程该进入哪个处理分支。
        if error:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return f"error={error}"[:500]
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return json.dumps({key: payload.get(key) for key in list(payload.keys())[:5]}, ensure_ascii=False, default=str)[:500]
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return str(payload)[:500]


# 定义 _compact_tool_payload 相关的处理逻辑，供流程或外部调用复用。
def _compact_tool_payload(payload: object) -> object:
    # 压缩工具原始返回，只保留后续诊断真正需要的关键字段。
    if isinstance(payload, dict):
        # 将 compact 的值保存下来，供后续流程判断或组装响应时使用。
        compact: dict[str, object] = {}
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for key in ("provider", "tool", "query", "summary", "error", "total", "status"):
            # 根据 key in payload 判断当前流程该进入哪个处理分支。
            if key in payload:
                # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
                value = payload.get(key)
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                compact[key] = _truncate_text(value, 500) if isinstance(value, str) else value
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for key in ("series", "matches", "dependencies"):
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = payload.get(key)
            # 根据 isinstance(value, list) 判断当前流程该进入哪个处理分支。
            if isinstance(value, list):
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                compact[key] = value[:3]
        # 根据 compact 判断当前流程该进入哪个处理分支。
        if compact:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return compact
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 调用 str 完成当前步骤需要的业务处理。
            str(key): _truncate_text(value, 500) if isinstance(value, str) else value
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for key, value in list(payload.items())[:5]
        }
    # 根据 isinstance(payload, list) 判断当前流程该进入哪个处理分支。
    if isinstance(payload, list):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return payload[:3]
    # 根据 isinstance(payload, str) 判断当前流程该进入哪个处理分支。
    if isinstance(payload, str):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _truncate_text(payload, 500)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return payload


# 定义 _truncate_text 相关的处理逻辑，供流程或外部调用复用。
def _truncate_text(value: object, limit: int) -> str:
    # 截断超长文本，避免日志、卡片和状态字段过大。
    text = str(value or "")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


# 定义 _fallback_plan_update 相关的处理逻辑，供流程或外部调用复用。
def _fallback_plan_update(state: DiagnosisState, error: BaseException) -> DiagnosisState:
    # 模型生成方案失败时构造保守方案，提醒先补证据、避免高风险操作。
    review = state.get("evidence_review") if isinstance(state.get("evidence_review"), dict) else {}
    # 将 missing 的值保存下来，供后续流程判断或组装响应时使用。
    missing = review.get("missing_evidence") if isinstance(review, dict) else []
    # 将 missing_items 的值保存下来，供后续流程判断或组装响应时使用。
    missing_items = [str(item) for item in missing] if isinstance(missing, list) else []
    # 将 actions 的值保存下来，供后续流程判断或组装响应时使用。
    actions = [
        # 执行当前业务步骤，推动流程继续向下游推进。
        "先修正失败的 Prometheus/SLS 查询，补齐关键指标和错误日志证据。",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "在证据不足前避免直接执行重启、回滚、扩容等高风险动作。",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "优先检查服务拓扑中异常或高延迟的下游依赖，并结合新证据重新生成方案。",
    ]
    # 根据 missing_items 判断当前流程该进入哪个处理分支。
    if missing_items:
        # 调用 actions.insert 完成当前步骤需要的业务处理。
        actions.insert(0, "补齐缺失证据：" + "；".join(missing_items[:3]))
    # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
    evidence = list(state.get("evidence", []))
    # 把当前结果追加到集合中，逐步构建最终输出。
    evidence.append(f"generate_plan fallback because LLM failed: {_truncate_text(error, 300)}")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "recommended_plan": {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "summary": "当前证据不足，已生成保守排查方案。",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "actions": actions,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "risk_level": "medium",
        },
        # 执行当前业务步骤，推动流程继续向下游推进。
        "evidence": evidence,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "need_human": True,
    }


# 定义 _format_tool_plan 相关的处理逻辑，供流程或外部调用复用。
def _format_tool_plan(plan: dict[str, object]) -> str:
    # 把工具计划格式化成用户能看懂的卡片文本。
    calls = [item for item in plan.get("tool_calls", []) if isinstance(item, dict)]
    # 根据 not calls 判断当前流程该进入哪个处理分支。
    if not calls:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"Tool router skipped: {plan.get('reason') or 'no tools needed'}"
    # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
    lines = [f"Tool router selected {len(calls)} tool(s): {plan.get('reason') or '-'}"]
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, call in enumerate(calls, start=1):
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append(f"{index}. {call.get('tool')} - {call.get('purpose') or '-'}")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "\n".join(lines)


# 定义 _format_tool_results 相关的处理逻辑，供流程或外部调用复用。
def _format_tool_results(results: list[dict[str, object]]) -> str:
    # 把每次工具执行的状态、数量、耗时和摘要整理成展示文本。
    if not results:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "Tool executor completed: no tools were executed."
    # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
    lines = ["Tool executor completed:"]
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for item in results:
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"- {item.get('tool_name')} status={item.get('status')} "
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"count={item.get('result_count')} elapsed_ms={item.get('elapsed_ms')} summary={item.get('summary')}"
        )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "\n".join(lines)


# 定义 _format_evidence_review 相关的处理逻辑，供流程或外部调用复用。
def _format_evidence_review(review: dict[str, object]) -> str:
    # 把证据审查结果翻译成卡片文本，说明证据是否充足以及缺什么。
    status = "sufficient" if review.get("is_sufficient") else "insufficient"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return (
        # 执行当前业务步骤，推动流程继续向下游推进。
        f"Evidence review completed: {status}.\n"
        # 调用 review.get 完成当前步骤需要的业务处理。
        f"Reason: {review.get('reason') or '-'}\n"
        # 调用 review.get 完成当前步骤需要的业务处理。
        f"Missing evidence: {review.get('missing_evidence') or []}"
    )


# 定义 _format_live_data 相关的处理逻辑，供流程或外部调用复用。
def _format_live_data(live_data: dict[str, Any]) -> str:
    # 把实时指标、日志和拓扑压缩成简短中文摘要。
    if not live_data:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "未获得实时指标、日志或拓扑数据。"
    # 将 metrics 的值保存下来，供后续流程判断或组装响应时使用。
    metrics = live_data.get("metrics") if isinstance(live_data.get("metrics"), dict) else {}
    # 将 logs 的值保存下来，供后续流程判断或组装响应时使用。
    logs = live_data.get("logs") if isinstance(live_data.get("logs"), dict) else {}
    # 将 topology 的值保存下来，供后续流程判断或组装响应时使用。
    topology = live_data.get("topology") if isinstance(live_data.get("topology"), dict) else {}
    # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
    lines = []
    # 根据 metrics 判断当前流程该进入哪个处理分支。
    if metrics:
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "指标："
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"p95={metrics.get('latency_p95_ms', '-') }ms，"
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"错误率={metrics.get('error_rate', '-') }，"
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"CPU={metrics.get('cpu_usage', '-') }"
        )
    # 将 matches 的值保存下来，供后续流程判断或组装响应时使用。
    matches = logs.get("matches") if isinstance(logs, dict) else None
    # 根据 matches 判断当前流程该进入哪个处理分支。
    if matches:
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append("日志：" + "；".join(str(item) for item in list(matches)[:2]))
    # 根据 topology 判断当前流程该进入哪个处理分支。
    if topology:
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "拓扑："
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"服务={topology.get('service', '-')}，"
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            f"依赖={', '.join(str(item) for item in topology.get('dependencies', [])[:5])}"
        )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "\n".join(lines) if lines else str(live_data)


# 定义 _format_plan_result 相关的处理逻辑，供流程或外部调用复用。
def _format_plan_result(plan: object, evidence: list[str]) -> str:
    # 把推荐方案和最近证据整理成飞书卡片可展示的中文结果。
    if not isinstance(plan, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"方案生成完成：{plan}"
    # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
    summary = str(plan.get("summary") or plan.get("title") or "已生成诊断方案。")
    # 将 actions 的值保存下来，供后续流程判断或组装响应时使用。
    actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    # 将 risk_level 的值保存下来，供后续流程判断或组装响应时使用。
    risk_level = str(plan.get("risk_level") or plan.get("risk") or "unknown")
    # 将 lines 的值保存下来，供后续流程判断或组装响应时使用。
    lines = [f"方案生成完成：{summary}", f"风险级别：{risk_level}"]
    # 根据 actions 判断当前流程该进入哪个处理分支。
    if actions:
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append("建议动作：")
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for index, action in enumerate(actions[:5], start=1):
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(f"{index}. {action}")
    # 根据 evidence 判断当前流程该进入哪个处理分支。
    if evidence:
        # 把当前结果追加到集合中，逐步构建最终输出。
        lines.append("证据摘要：")
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for item in evidence[-3:]:
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(f"- {item}")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "\n".join(lines)


# 定义 _build_summary_prompt 相关的处理逻辑，供流程或外部调用复用。
def _build_summary_prompt(state: TopicFlowState) -> str:
    # 生成最终总结提示词，要求模型按故障判断、证据、建议和风险输出中文结论。
    node_results: list[str] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for item in state.get("node_results", []):
        # 根据 not isinstance(item, dict) 判断当前流程该进入哪个处理分支。
        if not isinstance(item, dict):
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue
        # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
        node_name = str(item.get("node_name") or "unknown")
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = str(item.get("result") or "")
        # 把当前结果追加到集合中，逐步构建最终输出。
        node_results.append(f"## {node_name}\n{result[:3000]}")

    # 将 context 的值保存下来，供后续流程判断或组装响应时使用。
    context = "\n\n".join(node_results) or "No previous node results."
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return (
        # 执行当前业务步骤，推动流程继续向下游推进。
        "You are an AIOps incident response assistant for a Feishu interactive alert workflow.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "请只使用中文 Markdown 输出，内容要适合直接展示在飞书卡片中。\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "不要输出 JSON，不要输出代码块，不要复述原始字典，不要编造不存在的证据。\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "如果证据不足，请在关键证据中明确写“当前证据不足”。\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "处置建议必须可执行，按优先级从高到低排列，并避免危险命令。\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "请严格按以下标题输出：\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "## 故障判断\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "简洁说明最可能的问题。\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "## 关键证据\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- 列出来自缓存、RAG、实时工具调用中的关键证据。\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "## 处置建议\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "1. 给出优先级最高的操作。\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "2. 给出后续排查或恢复操作。\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "## 风险与观察项\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- 说明执行动作的风险。\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "- 说明后续需要观察的指标、日志或告警。\n\n"
        # 调用 state.get 完成当前步骤需要的业务处理。
        f"用户问题:\n{state.get('query', '')}\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        f"节点结果:\n{context}"
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return (
        # 执行当前业务步骤，推动流程继续向下游推进。
        "You are an AIOps incident response assistant for a Feishu interactive alert workflow.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Respond in concise Chinese Markdown.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Use the cache check, RAG retrieval, and live tool results below to produce a practical final diagnosis.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Do not invent evidence. If evidence is weak, say so briefly.\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "Output exactly these sections:\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "1. 故障判断\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "2. 关键证据\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "3. 处置建议\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        "4. 风险与观察项\n\n"
        # 调用 state.get 完成当前步骤需要的业务处理。
        f"用户问题:\n{state.get('query', '')}\n\n"
        # 执行当前业务步骤，推动流程继续向下游推进。
        f"节点结果:\n{context}"
    )
