from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.utils.retry_utils import async_retry

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 MockMetricsProvider 组件，集中管理这个模块的状态和行为。
class MockMetricsProvider:
    """Mock 监控指标提供者"""

    @async_retry(max_attempts=2)
    # 定义 get_metrics 相关的处理逻辑，供流程或外部调用复用。
    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 返回一组固定的模拟指标，方便本地演示诊断流程而不依赖真实监控系统。
        async with asyncio.timeout(5):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Fetching mock metrics summary_chars=%s", len(alert_summary))
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.sleep(0.05)
            # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
            monitor.record_tool_call("get_metrics", group_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 应用概览指标
                "qps": 1250.5,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "latency_p95_ms": 1850,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "latency_avg_ms": 450,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "error_rate": 0.087,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "error_count": 108,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "total_requests": 125000,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "cpu_usage": 0.64,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "memory_usage": 0.72,
                # JVM 指标
                "jvm_heap_used_mb": 512,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "jvm_non_heap_used_mb": 128,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "jvm_gc_count": 45,
                # 线程池指标
                "thread_active_count": 45,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "thread_pool_size": 100,
                # 告警信息
                "alerts": [
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "alert_name": "订单同步超时",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "severity": "错误",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "status": "已触发",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "start_time": 1716441600000,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "end_time": None,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "message": "order-sync timeout after 3 retries",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tags": {"service": "order-sync", "env": "production"},
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "alert_name": "支付接口响应慢",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "severity": "警告",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "status": "已触发",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "start_time": 1716441500000,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "end_time": None,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "message": "payment-api response time > 3s",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tags": {"service": "payment-api", "env": "production"},
                    },
                ],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "alert_count": 2,
                # 健康状态
                "health_status": "UNHEALTHY",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "bottlenecks": [
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "错误率过高: 8.70%",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "P95 延迟过高: 1850ms",
                ],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "provider": "mock",
            }


# 定义 MockLogsProvider 组件，集中管理这个模块的状态和行为。
class MockLogsProvider:
    """Mock 日志查询提供者"""

    @async_retry(max_attempts=2)
    # 定义 query_logs 相关的处理逻辑，供流程或外部调用复用。
    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 返回一组固定的模拟错误日志，方便本地演示日志证据如何进入诊断流程。
        async with asyncio.timeout(5):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Querying mock logs summary_chars=%s", len(alert_summary))
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.sleep(0.05)
            # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
            monitor.record_tool_call("query_logs", group_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "matches": [
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "timestamp": 1716441600,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "level": "ERROR",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "service": "order-sync",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "message": "timeout after 3 retries, downstream payment-api unavailable",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "trace_id": "abc123def456",
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "timestamp": 1716441590,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "level": "WARN",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "service": "payment-api",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "message": "slow response detected, latency > 3000ms",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "trace_id": "abc123def457",
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "timestamp": 1716441580,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "level": "ERROR",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "service": "order-sync",
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        "message": "connection pool exhausted, max_connections=50",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "trace_id": "abc123def458",
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "timestamp": 1716441570,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "level": "INFO",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "service": "order-sync",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "message": "retrying connection attempt 3/3",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "trace_id": "abc123def458",
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "timestamp": 1716441560,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "level": "ERROR",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "service": "redis-cache",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "message": "READONLY You can't write against a read only replica",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "trace_id": "abc123def459",
                    },
                ],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "total": 5,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "level_stats": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "ERROR": 3,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "WARN": 1,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "INFO": 1,
                },
                # 执行当前业务步骤，推动流程继续向下游推进。
                "error_messages": [
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "timeout after 3 retries, downstream payment-api unavailable",
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "connection pool exhausted, max_connections=50",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "READONLY You can't write against a read only replica",
                ],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "query": 'level: ERROR AND service: "order-sync"',
                # 执行当前业务步骤，推动流程继续向下游推进。
                "provider": "mock",
            }


# 定义 MockTopologyProvider 组件，集中管理这个模块的状态和行为。
class MockTopologyProvider:
    """Mock 拓扑信息提供者"""

    @async_retry(max_attempts=2)
    # 定义 get_topology 相关的处理逻辑，供流程或外部调用复用。
    async def get_topology(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 返回一组固定的模拟服务依赖，方便本地演示拓扑分析。
        async with asyncio.timeout(5):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Fetching mock topology summary_chars=%s", len(alert_summary))
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.sleep(0.05)
            # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
            monitor.record_tool_call("get_topology", group_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"service": "order-sync", "dependencies": ["payment-api", "mysql-primary", "redis-cache"]}


# 定义 MockToolsProvider 组件，集中管理这个模块的状态和行为。
class MockToolsProvider:
    """Mock 工具提供者，组合所有 Mock 子提供者"""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self) -> None:
        # 组装 mock 指标、日志和拓扑 provider，让本地环境也能一次性获取完整现场数据。
        self._metrics = MockMetricsProvider()
        # 将 self._logs 的值保存下来，供后续流程判断或组装响应时使用。
        self._logs = MockLogsProvider()
        # 将 self._topology 的值保存下来，供后续流程判断或组装响应时使用。
        self._topology = MockTopologyProvider()

    @property
    # 定义 metrics 相关的处理逻辑，供流程或外部调用复用。
    def metrics(self) -> MockMetricsProvider:
        # 返回 mock 指标 provider，供上层按统一工具接口调用。
        return self._metrics

    @property
    # 定义 logs 相关的处理逻辑，供流程或外部调用复用。
    def logs(self) -> MockLogsProvider:
        # 返回 mock 日志 provider，供上层按统一工具接口调用。
        return self._logs

    @property
    # 定义 topology 相关的处理逻辑，供流程或外部调用复用。
    def topology(self) -> MockTopologyProvider:
        # 返回 mock 拓扑 provider，供上层按统一工具接口调用。
        return self._topology

    # 定义 fetch_all 相关的处理逻辑，供流程或外部调用复用。
    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """并发获取所有数据"""
        # 并发返回模拟指标、日志和拓扑，保持和真实 provider 相同的数据结构。
        metrics, logs, topology = await asyncio.gather(
            # 调用 _metrics.get_metrics 完成当前步骤需要的业务处理。
            self._metrics.get_metrics(alert_summary, group_id),
            # 调用 _logs.query_logs 完成当前步骤需要的业务处理。
            self._logs.query_logs(alert_summary, group_id),
            # 调用 _topology.get_topology 完成当前步骤需要的业务处理。
            self._topology.get_topology(alert_summary, group_id),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"metrics": metrics, "logs": logs, "topology": topology}


# 向后兼容：保留原有函数接口
async def get_metrics(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 向后兼容旧函数调用方式，内部转到 MockMetricsProvider。
    return await MockMetricsProvider().get_metrics(alert_summary, group_id)


# 定义 query_logs 相关的处理逻辑，供流程或外部调用复用。
async def query_logs(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 向后兼容旧函数调用方式，内部转到 MockLogsProvider 查询模拟日志。
    return await MockLogsProvider().query_logs(alert_summary, group_id)


# 定义 get_topology 相关的处理逻辑，供流程或外部调用复用。
async def get_topology(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 向后兼容旧函数调用方式，内部转到 MockTopologyProvider。
    return await MockTopologyProvider().get_topology(alert_summary, group_id)


# 定义 fetch_all_live_data 相关的处理逻辑，供流程或外部调用复用。
async def fetch_all_live_data(alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
    # 向后兼容旧函数调用方式，一次性返回 mock 现场数据。
    return await MockToolsProvider().fetch_all(alert_summary, group_id)
