from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.utils.retry_utils import async_retry

logger = logging.getLogger(__name__)


class MockMetricsProvider:
    """Mock 监控指标提供者"""

    @async_retry(max_attempts=2)
    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 方法说明：返回一组固定的模拟指标，方便本地演示诊断流程而不依赖真实监控系统。
        async with asyncio.timeout(5):
            logger.info("Fetching mock metrics summary_chars=%s", len(alert_summary))
            await asyncio.sleep(0.05)
            monitor.record_tool_call("get_metrics", group_id)
            return {
                # 应用概览指标
                "qps": 1250.5,
                "latency_p95_ms": 1850,
                "latency_avg_ms": 450,
                "error_rate": 0.087,
                "error_count": 108,
                "total_requests": 125000,
                "cpu_usage": 0.64,
                "memory_usage": 0.72,
                # JVM 指标
                "jvm_heap_used_mb": 512,
                "jvm_non_heap_used_mb": 128,
                "jvm_gc_count": 45,
                # 线程池指标
                "thread_active_count": 45,
                "thread_pool_size": 100,
                # 告警信息
                "alerts": [
                    {
                        "alert_name": "订单同步超时",
                        "severity": "错误",
                        "status": "已触发",
                        "start_time": 1716441600000,
                        "end_time": None,
                        "message": "order-sync timeout after 3 retries",
                        "tags": {"service": "order-sync", "env": "production"},
                    },
                    {
                        "alert_name": "支付接口响应慢",
                        "severity": "警告",
                        "status": "已触发",
                        "start_time": 1716441500000,
                        "end_time": None,
                        "message": "payment-api response time > 3s",
                        "tags": {"service": "payment-api", "env": "production"},
                    },
                ],
                "alert_count": 2,
                # 健康状态
                "health_status": "UNHEALTHY",
                "bottlenecks": [
                    "错误率过高: 8.70%",
                    "P95 延迟过高: 1850ms",
                ],
                "provider": "mock",
            }


class MockLogsProvider:
    """Mock 日志查询提供者"""

    @async_retry(max_attempts=2)
    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 方法说明：从配置的后端或数据集中检索匹配内容。
        async with asyncio.timeout(5):
            logger.info("Querying mock logs summary_chars=%s", len(alert_summary))
            await asyncio.sleep(0.05)
            monitor.record_tool_call("query_logs", group_id)
            return {
                "matches": [
                    {
                        "timestamp": 1716441600,
                        "level": "ERROR",
                        "service": "order-sync",
                        "message": "timeout after 3 retries, downstream payment-api unavailable",
                        "trace_id": "abc123def456",
                    },
                    {
                        "timestamp": 1716441590,
                        "level": "WARN",
                        "service": "payment-api",
                        "message": "slow response detected, latency > 3000ms",
                        "trace_id": "abc123def457",
                    },
                    {
                        "timestamp": 1716441580,
                        "level": "ERROR",
                        "service": "order-sync",
                        "message": "connection pool exhausted, max_connections=50",
                        "trace_id": "abc123def458",
                    },
                    {
                        "timestamp": 1716441570,
                        "level": "INFO",
                        "service": "order-sync",
                        "message": "retrying connection attempt 3/3",
                        "trace_id": "abc123def458",
                    },
                    {
                        "timestamp": 1716441560,
                        "level": "ERROR",
                        "service": "redis-cache",
                        "message": "READONLY You can't write against a read only replica",
                        "trace_id": "abc123def459",
                    },
                ],
                "total": 5,
                "level_stats": {
                    "ERROR": 3,
                    "WARN": 1,
                    "INFO": 1,
                },
                "error_messages": [
                    "timeout after 3 retries, downstream payment-api unavailable",
                    "connection pool exhausted, max_connections=50",
                    "READONLY You can't write against a read only replica",
                ],
                "query": 'level: ERROR AND service: "order-sync"',
                "provider": "mock",
            }


class MockTopologyProvider:
    """Mock 拓扑信息提供者"""

    @async_retry(max_attempts=2)
    async def get_topology(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 方法说明：返回一组固定的模拟服务依赖，方便本地演示拓扑分析。
        async with asyncio.timeout(5):
            logger.info("Fetching mock topology summary_chars=%s", len(alert_summary))
            await asyncio.sleep(0.05)
            monitor.record_tool_call("get_topology", group_id)
            return {"service": "order-sync", "dependencies": ["payment-api", "mysql-primary", "redis-cache"]}


class MockToolsProvider:
    """Mock 工具提供者，组合所有 Mock 子提供者"""

    def __init__(self) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self._metrics = MockMetricsProvider()
        self._logs = MockLogsProvider()
        self._topology = MockTopologyProvider()

    @property
    def metrics(self) -> MockMetricsProvider:
        # 方法说明：返回 mock 指标 provider，供上层按统一工具接口调用。
        return self._metrics

    @property
    def logs(self) -> MockLogsProvider:
        # 方法说明：返回 mock 日志 provider，供上层按统一工具接口调用。
        return self._logs

    @property
    def topology(self) -> MockTopologyProvider:
        # 方法说明：返回 mock 拓扑 provider，供上层按统一工具接口调用。
        return self._topology

    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """并发获取所有数据"""
        # 方法说明：并发返回模拟指标、日志和拓扑，保持和真实 provider 相同的数据结构。
        metrics, logs, topology = await asyncio.gather(
            self._metrics.get_metrics(alert_summary, group_id),
            self._logs.query_logs(alert_summary, group_id),
            self._topology.get_topology(alert_summary, group_id),
        )
        return {"metrics": metrics, "logs": logs, "topology": topology}


# 向后兼容：保留原有函数接口
async def get_metrics(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 方法说明：向后兼容旧函数调用方式，内部转到 MockMetricsProvider。
    return await MockMetricsProvider().get_metrics(alert_summary, group_id)


async def query_logs(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 方法说明：从配置的后端或数据集中检索匹配内容。
    return await MockLogsProvider().query_logs(alert_summary, group_id)


async def get_topology(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 方法说明：向后兼容旧函数调用方式，内部转到 MockTopologyProvider。
    return await MockTopologyProvider().get_topology(alert_summary, group_id)


async def fetch_all_live_data(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 方法说明：向后兼容旧函数调用方式，一次性返回 mock 现场数据。
    return await MockToolsProvider().fetch_all(alert_summary, group_id)
