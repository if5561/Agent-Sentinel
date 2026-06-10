from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

logger = logging.getLogger(__name__)

TimeoutCallback = Callable[[str, str], None]


@dataclass(slots=True)
class TopicTask:
    task_id: str
    chat_id: str
    root_message_id: str
    query: str
    status: str = "running"
    created_at_monotonic: float = 0.0
    current_node: str = ""
    card_message_id: str | None = None
    last_action: str | None = None
    pending_node_result: str | None = None
    pending_diagnosis_state: dict[str, Any] | None = None
    timeout_timer: threading.Timer | None = None


class TopicTaskStore:
    """Thread-safe in-memory task pool for card decisions and timeouts."""

    def __init__(self, wait_seconds: int) -> None:
        # 准备线程安全的任务池，保存每个飞书话题正在等待哪个节点确认。
        self.wait_seconds = wait_seconds
        self._tasks: dict[str, TopicTask] = {}
        self._source_index: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    def create_task(self, task_id: str, chat_id: str, root_message_id: str, query: str) -> TopicTask:
        # 为一次飞书话题诊断创建任务；同一会话同一根消息重复进入时复用旧任务。
        with self._lock:
            source_key = self._source_key(chat_id, root_message_id)
            existing_task = self._task_by_source_key_locked(source_key)
            if existing_task is not None:
                logger.info(
                    "Interactive topic task reused source duplicate task_id=%s chat_id=%s root_message_id=%s",
                    existing_task.task_id,
                    chat_id,
                    root_message_id,
                )
                return replace(existing_task, timeout_timer=None)
            task = TopicTask(
                task_id=task_id,
                chat_id=chat_id,
                root_message_id=root_message_id,
                query=query,
                created_at_monotonic=time.perf_counter(),
            )
            self._tasks[task_id] = task
            self._source_index[source_key] = task_id
            logger.info("Interactive topic task created task_id=%s chat_id=%s", task_id, chat_id)
            return replace(task, timeout_timer=None)

    def set_card_message_id(self, task_id: str, card_message_id: str | None) -> TopicTask | None:
        # 记录当前任务对应的飞书卡片消息 ID，后续节点可以更新同一张卡片。
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning("Cannot set card id for missing interactive topic task task_id=%s", task_id)
                return None
            task.card_message_id = card_message_id
            return replace(task, timeout_timer=None)

    def mark_waiting(
        self,
        task_id: str,
        node_name: str,
        card_message_id: str | None,
        on_timeout: TimeoutCallback,
        node_result: str | None = None,
        diagnosis_state: dict[str, Any] | None = None,
    ) -> TopicTask | None:
        # 把任务切到“等待用户确认”状态，并启动超时计时器防止流程永久挂起。
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning("Cannot mark missing interactive topic task waiting task_id=%s", task_id)
                return None
            self._cancel_timer_locked(task)
            timer = threading.Timer(self.wait_seconds, on_timeout, args=(task_id, node_name))
            timer.daemon = True
            task.status = "pending"
            task.current_node = node_name
            task.card_message_id = card_message_id
            task.last_action = None
            task.pending_node_result = node_result
            task.pending_diagnosis_state = diagnosis_state
            task.timeout_timer = timer
            timer.start()
            logger.info(
                "Interactive topic task waiting task_id=%s node=%s timeout_seconds=%s",
                task_id,
                node_name,
                self.wait_seconds,
            )
            return replace(task, timeout_timer=None)

    def confirm_action(
        self,
        task_id: str,
        node_name: str,
        action: str,
        *,
        source: str,
    ) -> TopicTask | None:
        # 接收用户对当前节点的“继续/重试”选择，只接受仍在等待中的最新节点。
        normalized_action = "retry" if action == "retry" else "next"
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.info("Ignoring action for missing interactive topic task task_id=%s", task_id)
                return None
            if task.status != "pending":
                logger.info(
                    "Ignoring duplicate interactive topic action task_id=%s node=%s action=%s status=%s source=%s",
                    task_id,
                    node_name,
                    normalized_action,
                    task.status,
                    source,
                )
                return None
            if task.current_node != node_name:
                logger.info(
                    "Ignoring stale interactive topic action task_id=%s expected_node=%s actual_node=%s action=%s",
                    task_id,
                    task.current_node,
                    node_name,
                    normalized_action,
                )
                return None
            self._cancel_timer_locked(task)
            task.status = "retrying" if normalized_action == "retry" else "confirmed"
            task.last_action = normalized_action
            logger.info(
                "Interactive topic action accepted task_id=%s node=%s action=%s source=%s",
                task_id,
                node_name,
                normalized_action,
                source,
            )
            return replace(task, timeout_timer=None)

    def mark_feedback_waiting(self, task_id: str, card_message_id: str | None) -> TopicTask | None:
        # 诊断结束后把任务切到反馈等待状态，等待用户评价结果是否有效。
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning("Cannot mark missing interactive topic task feedback task_id=%s", task_id)
                return None
            self._cancel_timer_locked(task)
            task.status = "pending_feedback"
            task.current_node = "feedback_learning"
            task.card_message_id = card_message_id
            task.last_action = None
            logger.info("Interactive topic task waiting for feedback task_id=%s", task_id)
            return replace(task, timeout_timer=None)

    def confirm_feedback(self, task_id: str, action: str, *, source: str) -> TopicTask | None:
        # 记录用户对最终结果的反馈，并忽略重复或过期的反馈点击。
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.info("Ignoring feedback for missing interactive topic task task_id=%s", task_id)
                return None
            if task.status != "pending_feedback" or task.current_node != "feedback_learning":
                logger.info(
                    "Ignoring duplicate interactive topic feedback task_id=%s action=%s status=%s source=%s",
                    task_id,
                    action,
                    task.status,
                    source,
                )
                return None
            task.status = "feedback_received"
            task.last_action = action
            logger.info("Interactive topic feedback accepted task_id=%s action=%s source=%s", task_id, action, source)
            return replace(task, timeout_timer=None)

    def reset_feedback_waiting(self, task_id: str) -> TopicTask | None:
        # 反馈处理失败或需要重试时，把任务重新放回等待反馈状态。
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = "pending_feedback"
            task.current_node = "feedback_learning"
            task.last_action = None
            logger.info("Interactive topic feedback reset to pending task_id=%s", task_id)
            return replace(task, timeout_timer=None)

    def get_task(self, task_id: str) -> TopicTask | None:
        # 按任务 ID 查询任务快照，返回副本以免外部直接改内部状态。
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return replace(task, timeout_timer=None)

    def get_task_by_source(self, chat_id: str, root_message_id: str) -> TopicTask | None:
        # 按飞书会话和根消息查找任务，用来识别同一话题是否已经在运行。
        with self._lock:
            task = self._task_by_source_key_locked(self._source_key(chat_id, root_message_id))
            if task is None:
                return None
            return replace(task, timeout_timer=None)

    def finish_task(self, task_id: str) -> None:
        # 任务完成后清理任务池、来源索引和定时器，释放这次话题占用的状态。
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task is not None:
                self._source_index.pop(self._source_key(task.chat_id, task.root_message_id), None)
                self._cancel_timer_locked(task)
                logger.info("Interactive topic task finished task_id=%s", task_id)

    def _task_by_source_key_locked(self, source_key: tuple[str, str]) -> TopicTask | None:
        # 在持锁状态下通过来源索引找任务，并顺手清理已经失效的索引。
        task_id = self._source_index.get(source_key)
        if not task_id:
            return None
        task = self._tasks.get(task_id)
        if task is None:
            self._source_index.pop(source_key, None)
            return None
        return task

    def _source_key(self, chat_id: str, root_message_id: str) -> tuple[str, str]:
        # 把会话 ID 和根消息 ID 标准化成任务来源键，用于判断重复话题。
        return (str(chat_id or "").strip(), str(root_message_id or "").strip())

    def _cancel_timer_locked(self, task: TopicTask) -> None:
        # 取消任务上一次等待动作的超时计时器，避免用户已确认后又触发超时逻辑。
        timer = task.timeout_timer
        task.timeout_timer = None
        if timer is None:
            return
        try:
            timer.cancel()
        except Exception:
            logger.exception("Failed to cancel interactive topic timer task_id=%s", task.task_id)
