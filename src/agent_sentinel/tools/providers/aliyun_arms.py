from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.utils.retry_utils import async_retry

logger = logging.getLogger(__name__)

# 告警级别映射
ALERT_SEVERITY_MAP = {
    "CRITICAL": "严重",
    "ERROR": "错误",
    "WARN": "警告",
    "WARNING": "警告",
    "INFO": "信息",
}


class AliyunARMSClient:
    """阿里云 ARMS 客户端

    封装 alibabacloud-arms20190808 SDK，提供异步接口。
    用于查询应用监控指标（QPS、延迟、错误率等）。
    """

    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        region_id: str = "cn-hangzhou",
        app_id: str = "",
        timeout_seconds: int = 10,
    ) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self._access_key_id = access_key_id
        self._access_key_secret = access_key_secret
        self._region_id = region_id
        self._app_id = app_id
        self._timeout_seconds = timeout_seconds
        self._client = None

    def _get_client(self):
        """延迟初始化 ARMS 客户端"""
        # 方法说明：首次查询指标时才创建 ARMS SDK 客户端，避免服务启动阶段就依赖云 SDK。
        if self._client is None:
            try:
                from alibabacloud_arms20190808.client import Client as ARMSClient
                from alibabacloud_tea_openapi import models as open_api_models

                config = open_api_models.Config(
                    access_key_id=self._access_key_id,
                    access_key_secret=self._access_key_secret,
                    region_id=self._region_id,
                )
                self._client = ARMSClient(config)
            except ImportError:
                raise ImportError(
                    "alibabacloud-arms20190808 is required for ARMS provider. "
                    "Install it with: pip install alibabacloud-arms20190808"
                )
        return self._client

    @async_retry(max_attempts=2)
    async def get_app_metrics(
        self,
        app_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> dict[str, Any]:
        """获取应用监控指标

        Args:
            app_id: ARMS 应用 ID，默认使用配置的 app_id
            start_time: 开始时间戳（毫秒），默认为 1 小时前
            end_time: 结束时间戳（毫秒），默认为当前时间

        Returns:
            包含 QPS、延迟、错误率等指标的字典
        """
        # 方法说明：查询指定应用最近一段时间的核心监控指标，例如 QPS、延迟、错误率和资源使用率。
        import time

        client = self._get_client()
        app_id = app_id or self._app_id

        if not app_id:
            logger.warning("ARMS app_id not configured, returning empty metrics")
            return {}

        # 默认查询最近 1 小时
        now = int(time.time() * 1000)
        start_time = start_time or (now - 3600 * 1000)
        end_time = end_time or now

        try:
            async with asyncio.timeout(self._timeout_seconds):
                loop = asyncio.get_event_loop()
                metrics = await loop.run_in_executor(
                    None,
                    lambda: self._query_app_metrics(client, app_id, start_time, end_time),
                )
                return metrics

        except Exception as e:
            logger.error("ARMS metrics query failed: %s", e)
            raise

    def _query_app_metrics(
        self,
        client,
        app_id: str,
        start_time: int,
        end_time: int,
    ) -> dict[str, Any]:
        """查询应用指标的具体实现

        查询多个维度的指标：
        1. 应用概览（QPS、延迟、错误率）
        2. JVM 指标（堆内存、GC）
        3. 系统指标（CPU、内存）
        """
        # 方法说明：同步调用 ARMS 多个接口，把应用概览、JVM 和线程池指标汇总成一个字典。
        result = {}

        try:
            from alibabacloud_arms20190808 import models as arms_models

            # 1. 查询应用概览指标
            try:
                request = arms_models.QueryAppOverviewRequest(
                    app_id=app_id,
                    start_time=start_time,
                    end_time=end_time,
                )
                response = client.query_app_overview(request)
                overview = response.to_map().get("body", {})

                result.update({
                    "qps": overview.get("qps", 0),
                    "latency_p95_ms": overview.get("rt_p95", 0),
                    "latency_avg_ms": overview.get("rt_avg", 0),
                    "error_rate": overview.get("error_rate", 0),
                    "error_count": overview.get("error_count", 0),
                    "total_requests": overview.get("total_count", 0),
                    "cpu_usage": overview.get("cpu_usage", 0),
                    "memory_usage": overview.get("memory_usage", 0),
                })
            except Exception as e:
                logger.warning("Failed to query app overview: %s", e)

            # 2. 查询 JVM 指标
            try:
                jvm_metrics = self._query_jvm_metrics(client, app_id, start_time, end_time)
                result.update(jvm_metrics)
            except Exception as e:
                logger.warning("Failed to query JVM metrics: %s", e)

            # 3. 查询线程池指标
            try:
                thread_metrics = self._query_thread_metrics(client, app_id, start_time, end_time)
                result.update(thread_metrics)
            except Exception as e:
                logger.warning("Failed to query thread metrics: %s", e)

            return result

        except Exception as e:
            logger.warning("Failed to query app metrics: %s, returning empty", e)
            return {}

    def _query_jvm_metrics(
        self,
        client,
        app_id: str,
        start_time: int,
        end_time: int,
    ) -> dict[str, Any]:
        """查询 JVM 指标"""
        # 方法说明：查询 JVM 堆、非堆和 GC 指标，用来判断是否存在内存或 GC 压力。
        from alibabacloud_arms20190808 import models as arms_models

        # 查询 JVM 堆内存
        request = arms_models.QueryMetricRequest(
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            metric="jvm_heap_memory_used",
        )
        response = client.query_metric(request)
        heap_used = response.to_map().get("body", {}).get("values", [{}])

        # 查询 JVM 非堆内存
        request = arms_models.QueryMetricRequest(
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            metric="jvm_non_heap_memory_used",
        )
        response = client.query_metric(request)
        non_heap_used = response.to_map().get("body", {}).get("values", [{}])

        # 查询 GC 次数
        request = arms_models.QueryMetricRequest(
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            metric="jvm_gc_count",
        )
        response = client.query_metric(request)
        gc_count = response.to_map().get("body", {}).get("values", [{}])

        return {
            "jvm_heap_used_mb": self._extract_latest_value(heap_used),
            "jvm_non_heap_used_mb": self._extract_latest_value(non_heap_used),
            "jvm_gc_count": self._extract_latest_value(gc_count),
        }

    def _query_thread_metrics(
        self,
        client,
        app_id: str,
        start_time: int,
        end_time: int,
    ) -> dict[str, Any]:
        """查询线程池指标"""
        # 方法说明：查询活跃线程数和线程池大小，用来判断线程池是否接近打满。
        from alibabacloud_arms20190808 import models as arms_models

        # 查询活跃线程数
        request = arms_models.QueryMetricRequest(
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            metric="thread_active_count",
        )
        response = client.query_metric(request)
        active_threads = response.to_map().get("body", {}).get("values", [{}])

        # 查询线程池大小
        request = arms_models.QueryMetricRequest(
            app_id=app_id,
            start_time=start_time,
            end_time=end_time,
            metric="thread_pool_size",
        )
        response = client.query_metric(request)
        pool_size = response.to_map().get("body", {}).get("values", [{}])

        return {
            "thread_active_count": self._extract_latest_value(active_threads),
            "thread_pool_size": self._extract_latest_value(pool_size),
        }

    def _extract_latest_value(self, values: list) -> float:
        """从指标值列表中提取最新值"""
        # 方法说明：从时间序列里取最新点并转成数字，缺失或格式异常时返回 0。
        if not values:
            return 0.0
        try:
            return float(values[-1].get("value", 0))
        except (ValueError, TypeError):
            return 0.0

    @async_retry(max_attempts=2)
    async def get_alerts(
        self,
        app_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """获取应用告警信息

        Args:
            app_id: ARMS 应用 ID
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）

        Returns:
            告警列表
        """
        # 方法说明：查询 ARMS 最近告警事件，作为诊断时判断服务健康状态的补充证据。
        client = self._get_client()
        app_id = app_id or self._app_id

        if not app_id:
            return []

        # 默认查询最近 1 小时
        now = int(time.time() * 1000)
        start_time = start_time or (now - 3600 * 1000)
        end_time = end_time or now

        try:
            async with asyncio.timeout(self._timeout_seconds):
                loop = asyncio.get_event_loop()
                alerts = await loop.run_in_executor(
                    None,
                    lambda: self._query_alerts(client, app_id, start_time, end_time),
                )
                return alerts

        except Exception as e:
            logger.error("ARMS alerts query failed: %s", e)
            return []

    def _query_alerts(
        self,
        client,
        app_id: str,
        start_time: int,
        end_time: int,
    ) -> list[dict[str, Any]]:
        """查询告警的具体实现"""
        # 方法说明：同步调用 ARMS 告警接口，并把云厂商字段整理成项目内部使用的告警结构。
        try:
            from alibabacloud_arms20190808 import models as arms_models

            # 查询告警事件
            request = arms_models.QueryAlertEventsRequest(
                app_id=app_id,
                start_time=start_time,
                end_time=end_time,
            )
            response = client.query_alert_events(request)
            events = response.to_map().get("body", {}).get("events", [])

            alerts = []
            for event in events:
                alerts.append({
                    "alert_name": event.get("alert_name", ""),
                    "severity": ALERT_SEVERITY_MAP.get(event.get("severity", ""), event.get("severity", "")),
                    "status": event.get("status", ""),
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "message": event.get("message", ""),
                    "tags": event.get("tags", {}),
                })

            return alerts

        except Exception as e:
            logger.warning("Failed to query alerts: %s", e)
            return []


class ARMSMetricsProvider:
    """基于 ARMS 的监控指标提供者"""

    def __init__(self, client: AliyunARMSClient) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self._client = client

    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """获取监控指标

        Args:
            alert_summary: 告警摘要文本
            group_id: 飞书群组 ID（可用于映射到不同的 ARMS 应用）

        Returns:
            包含监控指标、告警信息和健康状态的字典
        """
        # 方法说明：并发查询应用指标和告警，再计算健康状态和可能瓶颈，供诊断方案参考。
        logger.info("Fetching ARMS metrics summary_chars=%s group_id=%s", len(alert_summary), group_id)

        # 并发查询指标和告警
        metrics_task = self._client.get_app_metrics()
        alerts_task = self._client.get_alerts()

        metrics, alerts = await asyncio.gather(
            metrics_task,
            alerts_task,
            return_exceptions=True,
        )

        # 处理异常
        if isinstance(metrics, Exception):
            logger.warning("Failed to get metrics: %s", metrics)
            metrics = {}

        if isinstance(alerts, Exception):
            logger.warning("Failed to get alerts: %s", alerts)
            alerts = []

        monitor.record_tool_call("get_metrics", group_id)

        # 计算健康状态
        health_status = self._calculate_health_status(metrics, alerts)

        # 分析性能瓶颈
        bottlenecks = self._analyze_bottlenecks(metrics)

        return {
            **metrics,
            "alerts": alerts,
            "alert_count": len(alerts),
            "health_status": health_status,
            "bottlenecks": bottlenecks,
            "provider": "aliyun_arms",
        }

    def _calculate_health_status(self, metrics: dict, alerts: list) -> str:
        """计算应用健康状态

        根据指标和告警综合评估应用健康状态。
        """
        # 方法说明：用严重告警、错误率、延迟、CPU 和内存等信号粗略判断应用是否健康。
        # 检查是否有严重告警
        critical_alerts = [a for a in alerts if a.get("severity") in ["严重", "错误"]]
        if critical_alerts:
            return "CRITICAL"

        # 检查关键指标
        error_rate = metrics.get("error_rate", 0)
        latency_p95 = metrics.get("latency_p95_ms", 0)
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)

        # 错误率超过 5% 视为不健康
        if error_rate > 0.05:
            return "UNHEALTHY"

        # 延迟超过 3 秒视为警告
        if latency_p95 > 3000:
            return "WARNING"

        # CPU 或内存超过 80% 视为警告
        if cpu_usage > 0.8 or memory_usage > 0.8:
            return "WARNING"

        # 有普通告警
        if alerts:
            return "WARNING"

        return "HEALTHY"

    def _analyze_bottlenecks(self, metrics: dict) -> list[str]:
        """分析性能瓶颈"""
        # 方法说明：从指标中提取可读的瓶颈描述，例如 CPU 高、延迟高或线程池打满。
        bottlenecks = []

        # 检查 CPU
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > 0.8:
            bottlenecks.append(f"CPU 使用率过高: {cpu_usage:.1%}")

        # 检查内存
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > 0.8:
            bottlenecks.append(f"内存使用率过高: {memory_usage:.1%}")

        # 检查延迟
        latency_p95 = metrics.get("latency_p95_ms", 0)
        if latency_p95 > 2000:
            bottlenecks.append(f"P95 延迟过高: {latency_p95:.0f}ms")

        # 检查错误率
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.01:
            bottlenecks.append(f"错误率过高: {error_rate:.2%}")

        # 检查线程池
        thread_active = metrics.get("thread_active_count", 0)
        thread_pool_size = metrics.get("thread_pool_size", 0)
        if thread_pool_size > 0 and thread_active / thread_pool_size > 0.8:
            bottlenecks.append(f"线程池使用率过高: {thread_active}/{thread_pool_size}")

        # 检查 JVM 堆内存
        jvm_heap_used = metrics.get("jvm_heap_used_mb", 0)
        if jvm_heap_used > 1024:  # 超过 1GB
            bottlenecks.append(f"JVM 堆内存使用过高: {jvm_heap_used:.0f}MB")

        return bottlenecks
