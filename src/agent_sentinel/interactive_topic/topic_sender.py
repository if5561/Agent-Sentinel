from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from agent_sentinel.feishu_app import FeishuBotClient

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
# 定义 WorkflowStep 组件，集中管理这个模块的状态和行为。
class WorkflowStep:
    # 执行当前业务步骤，推动流程继续向下游推进。
    node_name: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    title: str


# 将 WORKFLOW_STEPS 的值保存下来，供后续流程判断或组装响应时使用。
WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("understand", "告警理解"),
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("cache_check", "告警理解（缓存查找）"),
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("rag_retrieve", "RAG检索"),
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("tool_call", "实时数据"),
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("generate_plan", "生成方案"),
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("validate", "方案校验"),
    # 调用 WorkflowStep 完成当前步骤需要的业务处理。
    WorkflowStep("summary", "方案生成"),
)

# 将 STATUS_TEXT 的值保存下来，供后续流程判断或组装响应时使用。
STATUS_TEXT = {
    # 执行当前业务步骤，推动流程继续向下游推进。
    "done": "✅ 已完成",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "running": "🔄 正在执行",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "waiting": "⏸ 等待确认",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "skipped": "⏭ 已跳过",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "failed": "❌ 失败",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "pending": "⏸ 等待中",
}


# 定义 InteractiveTopicSender 组件，集中管理这个模块的状态和行为。
class InteractiveTopicSender:
    """Send and update a single Feishu card for an interactive workflow."""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, client: FeishuBotClient | None, wait_seconds: int = 5) -> None:
        # 保存飞书客户端和等待确认时间，并为每张卡片准备独立更新锁。
        self.client = client
        # 将 self.wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        self.wait_seconds = wait_seconds
        # 将 self._update_locks 的值保存下来，供后续流程判断或组装响应时使用。
        self._update_locks: dict[str, asyncio.Lock] = {}
        # 将 self._update_locks_guard 的值保存下来，供后续流程判断或组装响应时使用。
        self._update_locks_guard = asyncio.Lock()

    # 定义 send_workflow_card 相关的处理逻辑，供流程或外部调用复用。
    async def send_workflow_card(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        root_message_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        task_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        query: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> str | None:
        # 创建第一张工作流卡片，把诊断任务展示在原始飞书话题下。
        if not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive workflow card skipped chat_id=%s task_id=%s", chat_id, task_id)
            return None
        # 将 card 的值保存下来，供后续流程判断或组装响应时使用。
        card = build_workflow_card(
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=task_id,
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query=query,
            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
            node_statuses={},
            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
            current_result="工作流已创建，等待开始执行。",
            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
            current_node=None,
            # 将 wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            wait_seconds=self.wait_seconds,
            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
            buttons_node=None,
            # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
            feedback_buttons=False,
        )
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
            started = time.perf_counter()
            # 将 message_id 的值保存下来，供后续流程判断或组装响应时使用。
            message_id = await asyncio.to_thread(
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.client.send_topic_card,
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                card,
            )
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive workflow card sent task_id=%s chat_id=%s root_message_id=%s message_id=%s elapsed_ms=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                message_id,
                # 调用 int 完成当前步骤需要的业务处理。
                int((time.perf_counter() - started) * 1000),
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return message_id
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Failed to send interactive workflow card chat_id=%s root=%s task_id=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
            )
            return None

    # 定义 update_workflow_card 相关的处理逻辑，供流程或外部调用复用。
    async def update_workflow_card(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        message_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        task_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        query: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_statuses: dict[str, str],
        # 执行当前业务步骤，推动流程继续向下游推进。
        current_result: str,
        # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
        current_node: str | None = None,
        # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
        buttons_node: str | None = None,
        # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
        feedback_buttons: bool = False,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 刷新同一张工作流卡片，让用户看到当前节点状态、结果和可点击按钮。
        if not message_id or not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive workflow card update skipped task_id=%s current_node=%s result=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                current_node,
                # 执行当前业务步骤，推动流程继续向下游推进。
                current_result,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False

        # 将 card 的值保存下来，供后续流程判断或组装响应时使用。
        card = build_workflow_card(
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=task_id,
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query=query,
            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
            node_statuses=node_statuses,
            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
            current_result=current_result,
            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
            current_node=current_node,
            # 将 wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            wait_seconds=self.wait_seconds,
            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
            buttons_node=buttons_node,
            # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
            feedback_buttons=feedback_buttons,
        )
        # 将 lock 的值保存下来，供后续流程判断或组装响应时使用。
        lock = await self._get_update_lock(message_id)
        # 进入异步上下文管理区域，确保异步资源按约定释放。
        async with lock:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
                started = time.perf_counter()
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = await self.client.update_message_card_async(message_id, card)
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Interactive workflow card updated task_id=%s message_id=%s current_node=%s buttons_node=%s feedback_buttons=%s statuses=%s elapsed_ms=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    message_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    current_node,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    buttons_node,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    feedback_buttons,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    node_statuses,
                    # 调用 int 完成当前步骤需要的业务处理。
                    int((time.perf_counter() - started) * 1000),
                )
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return result
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.exception("Failed to update interactive workflow card message_id=%s", message_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return False

    # Backward-compatible wrappers kept for older call sites/tests.
    # 定义 send_topic_text 相关的处理逻辑，供流程或外部调用复用。
    async def send_topic_text(self, chat_id: str, root_message_id: str, text: str) -> str | None:
        # 兼容旧调用方式，在飞书原话题下发送一条文本回复。
        if not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic text skipped chat_id=%s text=%s", chat_id, text)
            return None
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return await asyncio.to_thread(self.client.send_topic_text, chat_id, root_message_id, text)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to send interactive topic text chat_id=%s root=%s", chat_id, root_message_id)
            return None

    # 定义 send_topic_card 相关的处理逻辑，供流程或外部调用复用。
    async def send_topic_card(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        root_message_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        task_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_result: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> str | None:
        # 兼容旧调用方式，把单节点结果也转成完整工作流卡片发送。
        return await self.send_workflow_card(chat_id, root_message_id, task_id, node_result)

    # 定义 update_topic_card_status 相关的处理逻辑，供流程或外部调用复用。
    async def update_topic_card_status(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        message_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        task_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_result: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        status_text: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 兼容旧调用方式，把某个节点的状态更新映射到新的工作流卡片结构。
        return await self.update_workflow_card(
            # 执行当前业务步骤，推动流程继续向下游推进。
            message_id,
            # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
            task_id=task_id,
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query="",
            # 将 node_statuses 的值保存下来，供后续流程判断或组装响应时使用。
            node_statuses={node_name: "done"},
            # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
            current_node=node_name,
            # 将 current_result 的值保存下来，供后续流程判断或组装响应时使用。
            current_result=status_text or node_result,
            # 将 buttons_node 的值保存下来，供后续流程判断或组装响应时使用。
            buttons_node=None,
            # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
            feedback_buttons=False,
        )

    # 定义 _get_update_lock 相关的处理逻辑，供流程或外部调用复用。
    async def _get_update_lock(self, message_id: str) -> asyncio.Lock:
        # 为同一张飞书卡片复用同一把异步锁，避免并发 PATCH 互相覆盖。
        async with self._update_locks_guard:
            # 将 lock 的值保存下来，供后续流程判断或组装响应时使用。
            lock = self._update_locks.get(message_id)
            # 根据 lock is None 判断当前流程该进入哪个处理分支。
            if lock is None:
                # 将 lock 的值保存下来，供后续流程判断或组装响应时使用。
                lock = asyncio.Lock()
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                self._update_locks[message_id] = lock
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return lock


# 定义 build_workflow_card 相关的处理逻辑，供流程或外部调用复用。
def build_workflow_card(
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    task_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    query: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    node_statuses: dict[str, str],
    # 执行当前业务步骤，推动流程继续向下游推进。
    current_result: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    current_node: str | None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    wait_seconds: int,
    # 执行当前业务步骤，推动流程继续向下游推进。
    buttons_node: str | None,
    # 将 feedback_buttons 的值保存下来，供后续流程判断或组装响应时使用。
    feedback_buttons: bool = False,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> dict[str, object]:
    # 组装工作流进度卡片，把节点列表、当前结果和确认/反馈按钮放到同一张卡片里。
    steps = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, step in enumerate(WORKFLOW_STEPS, start=1):
        # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
        status = node_statuses.get(step.node_name, "pending")
        # 把当前结果追加到集合中，逐步构建最终输出。
        steps.append(f"{index}. {STATUS_TEXT.get(status, STATUS_TEXT['pending'])} {step.title}")

    # 将 elements 的值保存下来，供后续流程判断或组装响应时使用。
    elements: list[dict[str, object]] = [
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tag": "markdown",
            # 调用 join 完成当前步骤需要的业务处理。
            "content": "\n".join(steps),
        },
        # 执行当前业务步骤，推动流程继续向下游推进。
        {"tag": "hr"},
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tag": "markdown",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "content": (
                # 执行当前业务步骤，推动流程继续向下游推进。
                f"**任务 ID**：{task_id}\n"
                # 调用 _truncate 完成当前步骤需要的业务处理。
                f"**用户问题**：{_truncate(query or '-', 500)}\n"
                # 调用 _step_title 完成当前步骤需要的业务处理。
                f"**当前节点**：{_step_title(current_node) if current_node else '-'}\n\n"
                # 调用 _truncate 完成当前步骤需要的业务处理。
                f"**当前节点结果**：\n{_truncate(current_result or '-', 1800)}"
            ),
        },
    ]

    # 根据 buttons_node 判断当前流程该进入哪个处理分支。
    if buttons_node:
        # 把一组结果合并到集合中，扩展后续可使用的数据范围。
        elements.extend(
            [
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "tag": "note",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "elements": [
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "tag": "plain_text",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "content": f"请确认当前节点结果；{wait_seconds} 秒无操作将按现有逻辑自动继续。",
                        }
                    ],
                },
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "tag": "action",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "actions": [
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "tag": "button",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "text": {"tag": "plain_text", "content": "同意"},
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "type": "primary",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "value": {"task_id": task_id, "node_name": buttons_node, "action": "next"},
                        },
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "tag": "button",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "text": {"tag": "plain_text", "content": "拒绝"},
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "type": "danger",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "value": {"task_id": task_id, "node_name": buttons_node, "action": "retry"},
                        },
                    ],
                },
            ]
        )

    # 根据 feedback_buttons 判断当前流程该进入哪个处理分支。
    if feedback_buttons:
        # 把一组结果合并到集合中，扩展后续可使用的数据范围。
        elements.extend(
            [
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "tag": "note",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "elements": [
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "tag": "plain_text",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "content": "请判断当前诊断结果是否有效；有效后会写入历史案例知识库。",
                        }
                    ],
                },
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "tag": "action",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "actions": [
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "tag": "button",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "text": {"tag": "plain_text", "content": "✅ 有效，存入知识库"},
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "type": "primary",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "value": {
                                # 执行当前业务步骤，推动流程继续向下游推进。
                                "task_id": task_id,
                                # 执行当前业务步骤，推动流程继续向下游推进。
                                "node_name": "feedback_learning",
                                # 执行当前业务步骤，推动流程继续向下游推进。
                                "action": "feedback_valid",
                            },
                        },
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "tag": "button",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "text": {"tag": "plain_text", "content": "❌ 无效，不存储"},
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "type": "danger",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "value": {
                                # 执行当前业务步骤，推动流程继续向下游推进。
                                "task_id": task_id,
                                # 执行当前业务步骤，推动流程继续向下游推进。
                                "node_name": "feedback_learning",
                                # 执行当前业务步骤，推动流程继续向下游推进。
                                "action": "feedback_invalid",
                            },
                        },
                    ],
                },
            ]
        )

    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "config": {"wide_screen_mode": True, "update_multi": True},
        # 执行当前业务步骤，推动流程继续向下游推进。
        "header": {
            # 调用 _header_template 完成当前步骤需要的业务处理。
            "template": _header_template(node_statuses),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title": {"tag": "plain_text", "content": "智能诊断工作流"},
        },
        # 执行当前业务步骤，推动流程继续向下游推进。
        "elements": elements,
    }


# 定义 _step_title 相关的处理逻辑，供流程或外部调用复用。
def _step_title(node_name: str | None) -> str:
    # 把内部节点名翻译成用户能看懂的中文阶段名称。
    for step in WORKFLOW_STEPS:
        # 根据 step.node_name == node_name 判断当前流程该进入哪个处理分支。
        if step.node_name == node_name:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return step.title
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return node_name or "-"


# 定义 _header_template 相关的处理逻辑，供流程或外部调用复用。
def _header_template(node_statuses: dict[str, str]) -> str:
    # 根据节点状态选择飞书卡片头部颜色，让失败、运行中和完成状态一眼可见。
    if any(status == "failed" for status in node_statuses.values()):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "red"
    # 根据 all(node_statuses.get(step.node_name) == "done" for ... 判断当前流程该进入哪个处理分支。
    if all(node_statuses.get(step.node_name) == "done" for step in WORKFLOW_STEPS):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "green"
    # 根据 any(status == "running" for status in node_statuses.... 判断当前流程该进入哪个处理分支。
    if any(status == "running" for status in node_statuses.values()):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "blue"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "wathet"


# 定义 _truncate 相关的处理逻辑，供流程或外部调用复用。
def _truncate(text: str, limit: int) -> str:
    # 限制卡片中的长文本长度，防止一次诊断结果把飞书卡片撑得过长。
    if len(text) <= limit:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return text
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return f"{text[:limit]}..."
