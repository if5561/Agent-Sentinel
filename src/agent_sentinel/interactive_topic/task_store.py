from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 将 TimeoutCallback 的值保存下来，供后续流程判断或组装响应时使用。
TimeoutCallback = Callable[[str, str], None]


@dataclass(slots=True)
# 定义 TopicTask 组件，集中管理这个模块的状态和行为。
class TopicTask:
    # 执行当前业务步骤，推动流程继续向下游推进。
    task_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    chat_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    root_message_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    query: str
    # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
    status: str = "running"
    # 将 created_at_monotonic 的值保存下来，供后续流程判断或组装响应时使用。
    created_at_monotonic: float = 0.0
    # 将 current_node 的值保存下来，供后续流程判断或组装响应时使用。
    current_node: str = ""
    # 将 card_message_id 的值保存下来，供后续流程判断或组装响应时使用。
    card_message_id: str | None = None
    # 将 last_action 的值保存下来，供后续流程判断或组装响应时使用。
    last_action: str | None = None
    # 将 pending_node_result 的值保存下来，供后续流程判断或组装响应时使用。
    pending_node_result: str | None = None
    # 将 pending_diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
    pending_diagnosis_state: dict[str, Any] | None = None
    # 将 timeout_timer 的值保存下来，供后续流程判断或组装响应时使用。
    timeout_timer: threading.Timer | None = None


# 定义 TopicTaskStore 组件，集中管理这个模块的状态和行为。
class TopicTaskStore:
    """Thread-safe in-memory task pool for card decisions and timeouts."""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, wait_seconds: int) -> None:
        # 准备线程安全的任务池，保存每个飞书话题正在等待哪个节点确认。
        self.wait_seconds = wait_seconds
        # 将 self._tasks 的值保存下来，供后续流程判断或组装响应时使用。
        self._tasks: dict[str, TopicTask] = {}
        # 将 self._source_index 的值保存下来，供后续流程判断或组装响应时使用。
        self._source_index: dict[tuple[str, str], str] = {}
        # 将 self._lock 的值保存下来，供后续流程判断或组装响应时使用。
        self._lock = threading.RLock()

    # 定义 create_task 相关的处理逻辑，供流程或外部调用复用。
    def create_task(self, task_id: str, chat_id: str, root_message_id: str, query: str) -> TopicTask:
        # 为一次飞书话题诊断创建任务；同一会话同一根消息重复进入时复用旧任务。
        with self._lock:
            # 将 source_key 的值保存下来，供后续流程判断或组装响应时使用。
            source_key = self._source_key(chat_id, root_message_id)
            # 将 existing_task 的值保存下来，供后续流程判断或组装响应时使用。
            existing_task = self._task_by_source_key_locked(source_key)
            # 根据 existing_task is not None 判断当前流程该进入哪个处理分支。
            if existing_task is not None:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Interactive topic task reused source duplicate task_id=%s chat_id=%s root_message_id=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    existing_task.task_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    chat_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    root_message_id,
                )
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return replace(existing_task, timeout_timer=None)
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = TopicTask(
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id=task_id,
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=chat_id,
                # 将 root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                root_message_id=root_message_id,
                # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                query=query,
                # 将 created_at_monotonic 的值保存下来，供后续流程判断或组装响应时使用。
                created_at_monotonic=time.perf_counter(),
            )
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            self._tasks[task_id] = task
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            self._source_index[source_key] = task_id
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic task created task_id=%s chat_id=%s", task_id, chat_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 set_card_message_id 相关的处理逻辑，供流程或外部调用复用。
    def set_card_message_id(self, task_id: str, card_message_id: str | None) -> TopicTask | None:
        # 记录当前任务对应的飞书卡片消息 ID，后续节点可以更新同一张卡片。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Cannot set card id for missing interactive topic task task_id=%s", task_id)
                return None
            # 将 task.card_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            task.card_message_id = card_message_id
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 mark_waiting 相关的处理逻辑，供流程或外部调用复用。
    def mark_waiting(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        task_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        card_message_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        on_timeout: TimeoutCallback,
        # 将 node_result 的值保存下来，供后续流程判断或组装响应时使用。
        node_result: str | None = None,
        # 将 diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
        diagnosis_state: dict[str, Any] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> TopicTask | None:
        # 把任务切到“等待用户确认”状态，并启动超时计时器防止流程永久挂起。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Cannot mark missing interactive topic task waiting task_id=%s", task_id)
                return None
            # 调用 self._cancel_timer_locked 完成当前步骤需要的业务处理。
            self._cancel_timer_locked(task)
            # 将 timer 的值保存下来，供后续流程判断或组装响应时使用。
            timer = threading.Timer(self.wait_seconds, on_timeout, args=(task_id, node_name))
            # 将 timer.daemon 的值保存下来，供后续流程判断或组装响应时使用。
            timer.daemon = True
            # 将 task.status 的值保存下来，供后续流程判断或组装响应时使用。
            task.status = "pending"
            # 将 task.current_node 的值保存下来，供后续流程判断或组装响应时使用。
            task.current_node = node_name
            # 将 task.card_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            task.card_message_id = card_message_id
            # 将 task.last_action 的值保存下来，供后续流程判断或组装响应时使用。
            task.last_action = None
            # 将 task.pending_node_result 的值保存下来，供后续流程判断或组装响应时使用。
            task.pending_node_result = node_result
            # 将 task.pending_diagnosis_state 的值保存下来，供后续流程判断或组装响应时使用。
            task.pending_diagnosis_state = diagnosis_state
            # 将 task.timeout_timer 的值保存下来，供后续流程判断或组装响应时使用。
            task.timeout_timer = timer
            # 调用 timer.start 完成当前步骤需要的业务处理。
            timer.start()
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive topic task waiting task_id=%s node=%s timeout_seconds=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                node_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.wait_seconds,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 confirm_action 相关的处理逻辑，供流程或外部调用复用。
    def confirm_action(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        task_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        node_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        action: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> TopicTask | None:
        # 接收用户对当前节点的“继续/重试”选择，只接受仍在等待中的最新节点。
        normalized_action = "retry" if action == "retry" else "next"
        # 进入上下文管理器保护的区域，自动处理资源生命周期。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Ignoring action for missing interactive topic task task_id=%s", task_id)
                return None
            # 根据 task.status != "pending" 判断当前流程该进入哪个处理分支。
            if task.status != "pending":
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Ignoring duplicate interactive topic action task_id=%s node=%s action=%s status=%s source=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    node_name,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    normalized_action,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task.status,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    source,
                )
                return None
            # 根据 task.current_node != node_name 判断当前流程该进入哪个处理分支。
            if task.current_node != node_name:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Ignoring stale interactive topic action task_id=%s expected_node=%s actual_node=%s action=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task.current_node,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    node_name,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    normalized_action,
                )
                return None
            # 调用 self._cancel_timer_locked 完成当前步骤需要的业务处理。
            self._cancel_timer_locked(task)
            # 将 task.status 的值保存下来，供后续流程判断或组装响应时使用。
            task.status = "retrying" if normalized_action == "retry" else "confirmed"
            # 将 task.last_action 的值保存下来，供后续流程判断或组装响应时使用。
            task.last_action = normalized_action
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Interactive topic action accepted task_id=%s node=%s action=%s source=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                task_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                node_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                normalized_action,
                # 执行当前业务步骤，推动流程继续向下游推进。
                source,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 mark_feedback_waiting 相关的处理逻辑，供流程或外部调用复用。
    def mark_feedback_waiting(self, task_id: str, card_message_id: str | None) -> TopicTask | None:
        # 诊断结束后把任务切到反馈等待状态，等待用户评价结果是否有效。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Cannot mark missing interactive topic task feedback task_id=%s", task_id)
                return None
            # 调用 self._cancel_timer_locked 完成当前步骤需要的业务处理。
            self._cancel_timer_locked(task)
            # 将 task.status 的值保存下来，供后续流程判断或组装响应时使用。
            task.status = "pending_feedback"
            # 将 task.current_node 的值保存下来，供后续流程判断或组装响应时使用。
            task.current_node = "feedback_learning"
            # 将 task.card_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            task.card_message_id = card_message_id
            # 将 task.last_action 的值保存下来，供后续流程判断或组装响应时使用。
            task.last_action = None
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic task waiting for feedback task_id=%s", task_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 confirm_feedback 相关的处理逻辑，供流程或外部调用复用。
    def confirm_feedback(self, task_id: str, action: str, *, source: str) -> TopicTask | None:
        # 记录用户对最终结果的反馈，并忽略重复或过期的反馈点击。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Ignoring feedback for missing interactive topic task task_id=%s", task_id)
                return None
            # 根据 task.status != "pending_feedback" or task.current_no... 判断当前流程该进入哪个处理分支。
            if task.status != "pending_feedback" or task.current_node != "feedback_learning":
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Ignoring duplicate interactive topic feedback task_id=%s action=%s status=%s source=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    action,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    task.status,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    source,
                )
                return None
            # 将 task.status 的值保存下来，供后续流程判断或组装响应时使用。
            task.status = "feedback_received"
            # 将 task.last_action 的值保存下来，供后续流程判断或组装响应时使用。
            task.last_action = action
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic feedback accepted task_id=%s action=%s source=%s", task_id, action, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 reset_feedback_waiting 相关的处理逻辑，供流程或外部调用复用。
    def reset_feedback_waiting(self, task_id: str) -> TopicTask | None:
        # 反馈处理失败或需要重试时，把任务重新放回等待反馈状态。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                return None
            # 将 task.status 的值保存下来，供后续流程判断或组装响应时使用。
            task.status = "pending_feedback"
            # 将 task.current_node 的值保存下来，供后续流程判断或组装响应时使用。
            task.current_node = "feedback_learning"
            # 将 task.last_action 的值保存下来，供后续流程判断或组装响应时使用。
            task.last_action = None
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic feedback reset to pending task_id=%s", task_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 get_task 相关的处理逻辑，供流程或外部调用复用。
    def get_task(self, task_id: str) -> TopicTask | None:
        # 按任务 ID 查询任务快照，返回副本以免外部直接改内部状态。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.get(task_id)
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                return None
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 get_task_by_source 相关的处理逻辑，供流程或外部调用复用。
    def get_task_by_source(self, chat_id: str, root_message_id: str) -> TopicTask | None:
        # 按飞书会话和根消息查找任务，用来识别同一话题是否已经在运行。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._task_by_source_key_locked(self._source_key(chat_id, root_message_id))
            # 根据 task is None 判断当前流程该进入哪个处理分支。
            if task is None:
                return None
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return replace(task, timeout_timer=None)

    # 定义 finish_task 相关的处理逻辑，供流程或外部调用复用。
    def finish_task(self, task_id: str) -> None:
        # 任务完成后清理任务池、来源索引和定时器，释放这次话题占用的状态。
        with self._lock:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = self._tasks.pop(task_id, None)
            # 根据 task is not None 判断当前流程该进入哪个处理分支。
            if task is not None:
                # 调用 _source_index.pop 完成当前步骤需要的业务处理。
                self._source_index.pop(self._source_key(task.chat_id, task.root_message_id), None)
                # 调用 self._cancel_timer_locked 完成当前步骤需要的业务处理。
                self._cancel_timer_locked(task)
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Interactive topic task finished task_id=%s", task_id)

    # 定义 _task_by_source_key_locked 相关的处理逻辑，供流程或外部调用复用。
    def _task_by_source_key_locked(self, source_key: tuple[str, str]) -> TopicTask | None:
        # 在持锁状态下通过来源索引找任务，并顺手清理已经失效的索引。
        task_id = self._source_index.get(source_key)
        # 根据 not task_id 判断当前流程该进入哪个处理分支。
        if not task_id:
            return None
        # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
        task = self._tasks.get(task_id)
        # 根据 task is None 判断当前流程该进入哪个处理分支。
        if task is None:
            # 调用 _source_index.pop 完成当前步骤需要的业务处理。
            self._source_index.pop(source_key, None)
            return None
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return task

    # 定义 _source_key 相关的处理逻辑，供流程或外部调用复用。
    def _source_key(self, chat_id: str, root_message_id: str) -> tuple[str, str]:
        # 把会话 ID 和根消息 ID 标准化成任务来源键，用于判断重复话题。
        return (str(chat_id or "").strip(), str(root_message_id or "").strip())

    # 定义 _cancel_timer_locked 相关的处理逻辑，供流程或外部调用复用。
    def _cancel_timer_locked(self, task: TopicTask) -> None:
        # 取消任务上一次等待动作的超时计时器，避免用户已确认后又触发超时逻辑。
        timer = task.timeout_timer
        # 将 task.timeout_timer 的值保存下来，供后续流程判断或组装响应时使用。
        task.timeout_timer = None
        # 根据 timer is None 判断当前流程该进入哪个处理分支。
        if timer is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 调用 timer.cancel 完成当前步骤需要的业务处理。
            timer.cancel()
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to cancel interactive topic timer task_id=%s", task.task_id)
