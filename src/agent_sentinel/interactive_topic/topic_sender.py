from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from agent_sentinel.feishu_app import FeishuBotClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    node_name: str
    title: str


WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep("understand", "告警理解"),
    WorkflowStep("cache_check", "告警理解（缓存查找）"),
    WorkflowStep("rag_retrieve", "RAG检索"),
    WorkflowStep("tool_call", "实时数据"),
    WorkflowStep("generate_plan", "生成方案"),
    WorkflowStep("validate", "方案校验"),
    WorkflowStep("summary", "方案生成"),
)

STATUS_TEXT = {
    "done": "✅ 已完成",
    "running": "🔄 正在执行",
    "waiting": "⏸ 等待确认",
    "skipped": "⏭ 已跳过",
    "failed": "❌ 失败",
    "pending": "⏸ 等待中",
}


class InteractiveTopicSender:
    """Send and update a single Feishu card for an interactive workflow."""

    def __init__(self, client: FeishuBotClient | None, wait_seconds: int = 5) -> None:
        # 保存飞书客户端和等待确认时间，并为每张卡片准备独立更新锁。
        self.client = client
        self.wait_seconds = wait_seconds
        self._update_locks: dict[str, asyncio.Lock] = {}
        self._update_locks_guard = asyncio.Lock()

    async def send_workflow_card(
        self,
        chat_id: str,
        root_message_id: str,
        task_id: str,
        query: str,
    ) -> str | None:
        # 创建第一张工作流卡片，把诊断任务展示在原始飞书话题下。
        if not self.client or not self.client.is_configured():
            logger.info("Interactive workflow card skipped chat_id=%s task_id=%s", chat_id, task_id)
            return None
        card = build_workflow_card(
            task_id=task_id,
            query=query,
            node_statuses={},
            current_result="工作流已创建，等待开始执行。",
            current_node=None,
            wait_seconds=self.wait_seconds,
            buttons_node=None,
            feedback_buttons=False,
        )
        try:
            started = time.perf_counter()
            message_id = await asyncio.to_thread(
                self.client.send_topic_card,
                chat_id,
                root_message_id,
                card,
            )
            logger.info(
                "Interactive workflow card sent task_id=%s chat_id=%s root_message_id=%s message_id=%s elapsed_ms=%s",
                task_id,
                chat_id,
                root_message_id,
                message_id,
                int((time.perf_counter() - started) * 1000),
            )
            return message_id
        except Exception:
            logger.exception(
                "Failed to send interactive workflow card chat_id=%s root=%s task_id=%s",
                chat_id,
                root_message_id,
                task_id,
            )
            return None

    async def update_workflow_card(
        self,
        message_id: str | None,
        *,
        task_id: str,
        query: str,
        node_statuses: dict[str, str],
        current_result: str,
        current_node: str | None = None,
        buttons_node: str | None = None,
        feedback_buttons: bool = False,
    ) -> bool:
        # 刷新同一张工作流卡片，让用户看到当前节点状态、结果和可点击按钮。
        if not message_id or not self.client or not self.client.is_configured():
            logger.info(
                "Interactive workflow card update skipped task_id=%s current_node=%s result=%s",
                task_id,
                current_node,
                current_result,
            )
            return False

        card = build_workflow_card(
            task_id=task_id,
            query=query,
            node_statuses=node_statuses,
            current_result=current_result,
            current_node=current_node,
            wait_seconds=self.wait_seconds,
            buttons_node=buttons_node,
            feedback_buttons=feedback_buttons,
        )
        lock = await self._get_update_lock(message_id)
        async with lock:
            try:
                started = time.perf_counter()
                result = await self.client.update_message_card_async(message_id, card)
                logger.info(
                    "Interactive workflow card updated task_id=%s message_id=%s current_node=%s buttons_node=%s feedback_buttons=%s statuses=%s elapsed_ms=%s",
                    task_id,
                    message_id,
                    current_node,
                    buttons_node,
                    feedback_buttons,
                    node_statuses,
                    int((time.perf_counter() - started) * 1000),
                )
                return result
            except Exception:
                logger.exception("Failed to update interactive workflow card message_id=%s", message_id)
                return False

    # Backward-compatible wrappers kept for older call sites/tests.
    async def send_topic_text(self, chat_id: str, root_message_id: str, text: str) -> str | None:
        # 兼容旧调用方式，在飞书原话题下发送一条文本回复。
        if not self.client or not self.client.is_configured():
            logger.info("Interactive topic text skipped chat_id=%s text=%s", chat_id, text)
            return None
        try:
            return await asyncio.to_thread(self.client.send_topic_text, chat_id, root_message_id, text)
        except Exception:
            logger.exception("Failed to send interactive topic text chat_id=%s root=%s", chat_id, root_message_id)
            return None

    async def send_topic_card(
        self,
        chat_id: str,
        root_message_id: str,
        task_id: str,
        node_name: str,
        node_result: str,
    ) -> str | None:
        # 兼容旧调用方式，把单节点结果也转成完整工作流卡片发送。
        return await self.send_workflow_card(chat_id, root_message_id, task_id, node_result)

    async def update_topic_card_status(
        self,
        message_id: str | None,
        task_id: str,
        node_name: str,
        node_result: str,
        status_text: str,
    ) -> bool:
        # 兼容旧调用方式，把某个节点的状态更新映射到新的工作流卡片结构。
        return await self.update_workflow_card(
            message_id,
            task_id=task_id,
            query="",
            node_statuses={node_name: "done"},
            current_node=node_name,
            current_result=status_text or node_result,
            buttons_node=None,
            feedback_buttons=False,
        )

    async def _get_update_lock(self, message_id: str) -> asyncio.Lock:
        # 为同一张飞书卡片复用同一把异步锁，避免并发 PATCH 互相覆盖。
        async with self._update_locks_guard:
            lock = self._update_locks.get(message_id)
            if lock is None:
                lock = asyncio.Lock()
                self._update_locks[message_id] = lock
            return lock


def build_workflow_card(
    *,
    task_id: str,
    query: str,
    node_statuses: dict[str, str],
    current_result: str,
    current_node: str | None,
    wait_seconds: int,
    buttons_node: str | None,
    feedback_buttons: bool = False,
) -> dict[str, object]:
    # 组装工作流进度卡片，把节点列表、当前结果和确认/反馈按钮放到同一张卡片里。
    steps = []
    for index, step in enumerate(WORKFLOW_STEPS, start=1):
        status = node_statuses.get(step.node_name, "pending")
        steps.append(f"{index}. {STATUS_TEXT.get(status, STATUS_TEXT['pending'])} {step.title}")

    elements: list[dict[str, object]] = [
        {
            "tag": "markdown",
            "content": "\n".join(steps),
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": (
                f"**任务 ID**：{task_id}\n"
                f"**用户问题**：{_truncate(query or '-', 500)}\n"
                f"**当前节点**：{_step_title(current_node) if current_node else '-'}\n\n"
                f"**当前节点结果**：\n{_truncate(current_result or '-', 1800)}"
            ),
        },
    ]

    if buttons_node:
        elements.extend(
            [
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"请确认当前节点结果；{wait_seconds} 秒无操作将按现有逻辑自动继续。",
                        }
                    ],
                },
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "同意"},
                            "type": "primary",
                            "value": {"task_id": task_id, "node_name": buttons_node, "action": "next"},
                        },
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "拒绝"},
                            "type": "danger",
                            "value": {"task_id": task_id, "node_name": buttons_node, "action": "retry"},
                        },
                    ],
                },
            ]
        )

    if feedback_buttons:
        elements.extend(
            [
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": "请判断当前诊断结果是否有效；有效后会写入历史案例知识库。",
                        }
                    ],
                },
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "✅ 有效，存入知识库"},
                            "type": "primary",
                            "value": {
                                "task_id": task_id,
                                "node_name": "feedback_learning",
                                "action": "feedback_valid",
                            },
                        },
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "❌ 无效，不存储"},
                            "type": "danger",
                            "value": {
                                "task_id": task_id,
                                "node_name": "feedback_learning",
                                "action": "feedback_invalid",
                            },
                        },
                    ],
                },
            ]
        )

    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "template": _header_template(node_statuses),
            "title": {"tag": "plain_text", "content": "智能诊断工作流"},
        },
        "elements": elements,
    }


def _step_title(node_name: str | None) -> str:
    # 把内部节点名翻译成用户能看懂的中文阶段名称。
    for step in WORKFLOW_STEPS:
        if step.node_name == node_name:
            return step.title
    return node_name or "-"


def _header_template(node_statuses: dict[str, str]) -> str:
    # 根据节点状态选择飞书卡片头部颜色，让失败、运行中和完成状态一眼可见。
    if any(status == "failed" for status in node_statuses.values()):
        return "red"
    if all(node_statuses.get(step.node_name) == "done" for step in WORKFLOW_STEPS):
        return "green"
    if any(status == "running" for status in node_statuses.values()):
        return "blue"
    return "wathet"


def _truncate(text: str, limit: int) -> str:
    # 限制卡片中的长文本长度，防止一次诊断结果把飞书卡片撑得过长。
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
