from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - keeps local test env usable before dependencies are installed.
    # 测试或轻量环境可能暂未安装 prometheus_client；noop 指标保证业务代码仍可导入运行。
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def labels(self, **_: str) -> "_NoopMetric":
            # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
            return self

        def inc(self, _: float = 1.0) -> None:
            # 方法说明：记录可观测事件或监控指标，便于后续追踪。
            return None

        def dec(self, _: float = 1.0) -> None:
            # 方法说明：记录可观测事件或监控指标，便于后续追踪。
            return None

        def observe(self, _: float) -> None:
            # 方法说明：记录可观测事件或监控指标，便于后续追踪。
            return None

    class Counter(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            # 方法说明：初始化对象，并保存后续调用需要的状态。
            return None

    class Gauge(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            # 方法说明：初始化对象，并保存后续调用需要的状态。
            return None

    class Histogram(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            # 方法说明：初始化对象，并保存后续调用需要的状态。
            return None

    def generate_latest() -> bytes:  # type: ignore[no-redef]
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        return b"# prometheus_client is not installed\n"


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
DEFAULT_GROUP_ID = "unknown"
DEFAULT_WORKFLOW_TYPE = "diagnosis"


class Monitor:
    """Manual Prometheus metrics for the AIOps LangGraph and Feishu workflow."""

    def __init__(self) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.enabled = True

        # Node execution latency by LangGraph node name and group id.
        # Histogram 用于后续通过 histogram_quantile 计算 P50/P95/P99 节点耗时。
        self.workflow_node_duration_seconds = Histogram(
            "workflow_node_duration_seconds",
            "LangGraph node execution duration in seconds.",
            ["node_name", "group_id"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        )
        self.workflow_total_duration_seconds = Histogram(
            "workflow_total_duration_seconds",
            "End-to-end workflow duration in seconds.",
            ["workflow_type", "group_id"],
            buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
        )
        self.llm_call_duration_seconds = Histogram(
            "llm_call_duration_seconds",
            "LLM call duration in seconds.",
            ["model"],
            buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60),
        )

        # Counters for workflow volume, cache quality, retrieval and errors.
        # Counter 只增不减，适合记录节点调用量、缓存命中和错误次数等累计事件。
        self.workflow_node_count_total = Counter(
            "workflow_node_count_total",
            "Total LangGraph node invocations.",
            ["node_name", "group_id", "status"],
        )
        self.workflow_cache_hit_total = Counter(
            "workflow_cache_hit_total",
            "Similar-case cache hit count.",
            ["group_id"],
        )
        self.workflow_cache_miss_total = Counter(
            "workflow_cache_miss_total",
            "Similar-case cache miss count.",
            ["group_id"],
        )
        self.rag_retrieval_count_total = Counter(
            "rag_retrieval_count_total",
            "RAG retrieval request count.",
            ["retriever", "group_id"],
        )
        self.tool_call_count_total = Counter(
            "tool_call_count_total",
            "Tool call count.",
            ["tool_name", "group_id", "status"],
        )
        self.feedback_positive_total = Counter(
            "feedback_positive_total",
            "Positive user feedback count.",
            ["group_id"],
        )
        self.feedback_negative_total = Counter(
            "feedback_negative_total",
            "Negative user feedback count.",
            ["group_id"],
        )
        self.error_count_total = Counter(
            "error_count_total",
            "Application error count by error type.",
            ["error_type"],
        )
        self.token_consumption_total = Counter(
            "token_consumption_total",
            "LLM token consumption.",
            ["model", "token_type"],
        )
        self.current_workflow_active = Gauge(
            "current_workflow_active",
            "Currently active workflows.",
            ["workflow_type", "group_id"],
        )

    def configure(self, *, enabled: bool) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.enabled = enabled

    @contextmanager
    def track_node(self, node_name: str, group_id: str | None = None) -> Iterator[None]:
        """Context manager example: `with monitor.track_node("retrieve", group_id): ...`."""
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if not self.enabled:
            yield
            return
        labels = {"node_name": node_name, "group_id": _label(group_id)}
        started = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            # 异常会继续向外抛出；这里仅补充错误指标，不能吞掉业务失败。
            status = "error"
            self.record_error(f"{node_name}_error")
            raise
        finally:
            # finally 中记录耗时，确保成功和失败路径都能反映真实节点开销。
            self.workflow_node_duration_seconds.labels(**labels).observe(time.perf_counter() - started)
            self.workflow_node_count_total.labels(**labels, status=status).inc()

    def track_node_async(self, node_name: str) -> Callable[[F], F]:
        """Async decorator example: `@monitor.track_node_async("understand")`."""
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。

        def decorator(func: F) -> F:
            @functools.wraps(func)
            # 方法说明：记录可观测事件或监控指标，便于后续追踪。
            async def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
                # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
                with self.track_node(node_name, group_id_from_state(state)):
                    return await func(state, *args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    @contextmanager
    def track_workflow(self, workflow_type: str, group_id: str | None = None) -> Iterator[None]:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if not self.enabled:
            yield
            return
        labels = {"workflow_type": workflow_type, "group_id": _label(group_id)}
        started = time.perf_counter()
        # Gauge 表示当前活跃工作流数，开始时加一，结束时必须减一。
        self.current_workflow_active.labels(**labels).inc()
        try:
            yield
        except Exception:
            self.record_error(f"{workflow_type}_workflow_error")
            raise
        finally:
            self.workflow_total_duration_seconds.labels(**labels).observe(time.perf_counter() - started)
            self.current_workflow_active.labels(**labels).dec()

    def record_llm_call(self, model: str, duration_seconds: float) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if self.enabled:
            self.llm_call_duration_seconds.labels(model=_label(model)).observe(duration_seconds)

    def record_tokens(self, model: str, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if not self.enabled:
            return
        self.token_consumption_total.labels(model=_label(model), token_type="prompt").inc(max(prompt_tokens, 0))
        self.token_consumption_total.labels(model=_label(model), token_type="completion").inc(max(completion_tokens, 0))

    def record_cache_hit(self, group_id: str | None = None) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if self.enabled:
            self.workflow_cache_hit_total.labels(group_id=_label(group_id)).inc()

    def record_cache_miss(self, group_id: str | None = None) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if self.enabled:
            self.workflow_cache_miss_total.labels(group_id=_label(group_id)).inc()

    def record_rag_retrieval(self, retriever: str, group_id: str | None = None) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if self.enabled:
            self.rag_retrieval_count_total.labels(retriever=_label(retriever), group_id=_label(group_id)).inc()

    def record_tool_call(self, tool_name: str, group_id: str | None = None, *, status: str = "success") -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if self.enabled:
            self.tool_call_count_total.labels(tool_name=_label(tool_name), group_id=_label(group_id), status=_label(status)).inc()

    def record_feedback(self, positive: bool, group_id: str | None = None) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if not self.enabled:
            return
        if positive:
            self.feedback_positive_total.labels(group_id=_label(group_id)).inc()
        else:
            self.feedback_negative_total.labels(group_id=_label(group_id)).inc()

    def record_error(self, error_type: str) -> None:
        # 方法说明：记录可观测事件或监控指标，便于后续追踪。
        if self.enabled:
            self.error_count_total.labels(error_type=_label(error_type)).inc()

    def render_latest(self) -> bytes:
        # 方法说明：读取并返回当前流程需要的数据。
        return generate_latest()


monitor = Monitor()


def configure_monitoring(*, enabled: bool) -> None:
    # 方法说明：初始化对象，并保存后续调用需要的状态。
    monitor.configure(enabled=enabled)


def group_id_from_state(state: dict[str, Any] | None) -> str:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if not isinstance(state, dict):
        return DEFAULT_GROUP_ID
    return _label(state.get("chat_id"))


def trace_id_from_parts(chat_id: str | None, message_id: str | None) -> str:
    # trace_id 用于日志串联单次诊断，刻意不进入 Prometheus label，避免高基数时间序列。
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if chat_id and message_id:
        return f"{chat_id}_{message_id}"
    return chat_id or message_id or "unknown"


def trace_id_from_state(state: dict[str, Any] | None) -> str:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if not isinstance(state, dict):
        return "unknown"
    existing = str(state.get("trace_id") or "").strip()
    if existing:
        return existing
    return trace_id_from_parts(str(state.get("chat_id") or ""), str(state.get("thread_root_message_id") or ""))


def _label(value: object) -> str:
    # Prometheus label 不接受空值；统一归一化可以减少每个打点处的防御代码。
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    text = str(value or DEFAULT_GROUP_ID).strip()
    return text or DEFAULT_GROUP_ID
