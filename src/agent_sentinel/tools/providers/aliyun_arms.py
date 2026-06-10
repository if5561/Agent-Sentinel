from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.utils.retry_utils import async_retry

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 告警级别映射
ALERT_SEVERITY_MAP = {
    # 执行当前业务步骤，推动流程继续向下游推进。
    "CRITICAL": "严重",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "ERROR": "错误",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "WARN": "警告",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "WARNING": "警告",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "INFO": "信息",
}


# 定义 AliyunARMSClient 组件，集中管理这个模块的状态和行为。
class AliyunARMSClient:
    """阿里云 ARMS 客户端

    封装 alibabacloud-arms20190808 SDK，提供异步接口。
    用于查询应用监控指标（QPS、延迟、错误率等）。
    """

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        access_key_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        access_key_secret: str,
        # 将 region_id 的值保存下来，供后续流程判断或组装响应时使用。
        region_id: str = "cn-hangzhou",
        # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
        app_id: str = "",
        # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        timeout_seconds: int = 10,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 保存 ARMS 鉴权、地域、应用 ID 和超时配置，SDK 客户端等到首次查询时再创建。
        self._access_key_id = access_key_id
        # 将 self._access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
        self._access_key_secret = access_key_secret
        # 将 self._region_id 的值保存下来，供后续流程判断或组装响应时使用。
        self._region_id = region_id
        # 将 self._app_id 的值保存下来，供后续流程判断或组装响应时使用。
        self._app_id = app_id
        # 将 self._timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        self._timeout_seconds = timeout_seconds
        # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
        self._client = None

    # 定义 _get_client 相关的处理逻辑，供流程或外部调用复用。
    def _get_client(self):
        """延迟初始化 ARMS 客户端"""
        # 首次查询指标时才创建 ARMS SDK 客户端，避免服务启动阶段就依赖云 SDK。
        if self._client is None:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                from alibabacloud_arms20190808.client import Client as ARMSClient
                from alibabacloud_tea_openapi import models as open_api_models

                # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
                config = open_api_models.Config(
                    # 将 access_key_id 的值保存下来，供后续流程判断或组装响应时使用。
                    access_key_id=self._access_key_id,
                    # 将 access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
                    access_key_secret=self._access_key_secret,
                    # 将 region_id 的值保存下来，供后续流程判断或组装响应时使用。
                    region_id=self._region_id,
                )
                # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
                self._client = ARMSClient(config)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except ImportError:
                # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
                raise ImportError(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "alibabacloud-arms20190808 is required for ARMS provider. "
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Install it with: pip install alibabacloud-arms20190808"
                )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._client

    @async_retry(max_attempts=2)
    # 定义 get_app_metrics 相关的处理逻辑，供流程或外部调用复用。
    async def get_app_metrics(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
        app_id: str | None = None,
        # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
        start_time: int | None = None,
        # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
        end_time: int | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, Any]:
        """获取应用监控指标

        Args:
            app_id: ARMS 应用 ID，默认使用配置的 app_id
            start_time: 开始时间戳（毫秒），默认为 1 小时前
            end_time: 结束时间戳（毫秒），默认为当前时间

        Returns:
            包含 QPS、延迟、错误率等指标的字典
        """
        # 查询指定应用最近一段时间的核心监控指标，例如 QPS、延迟、错误率和资源使用率。
        import time

        # 将 client 的值保存下来，供后续流程判断或组装响应时使用。
        client = self._get_client()
        # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
        app_id = app_id or self._app_id

        # 根据 not app_id 判断当前流程该进入哪个处理分支。
        if not app_id:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("ARMS app_id not configured, returning empty metrics")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {}

        # 默认查询最近 1 小时
        now = int(time.time() * 1000)
        # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
        start_time = start_time or (now - 3600 * 1000)
        # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
        end_time = end_time or now

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 进入异步上下文管理区域，确保异步资源按约定释放。
            async with asyncio.timeout(self._timeout_seconds):
                # 将 loop 的值保存下来，供后续流程判断或组装响应时使用。
                loop = asyncio.get_event_loop()
                # 将 metrics 的值保存下来，供后续流程判断或组装响应时使用。
                metrics = await loop.run_in_executor(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    None,
                    # 调用 self._query_app_metrics 完成当前步骤需要的业务处理。
                    lambda: self._query_app_metrics(client, app_id, start_time, end_time),
                )
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return metrics

        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as e:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.error("ARMS metrics query failed: %s", e)
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise

    # 定义 _query_app_metrics 相关的处理逻辑，供流程或外部调用复用。
    def _query_app_metrics(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        client,
        # 执行当前业务步骤，推动流程继续向下游推进。
        app_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        start_time: int,
        # 执行当前业务步骤，推动流程继续向下游推进。
        end_time: int,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, Any]:
        """查询应用指标的具体实现

        查询多个维度的指标：
        1. 应用概览（QPS、延迟、错误率）
        2. JVM 指标（堆内存、GC）
        3. 系统指标（CPU、内存）
        """
        # 同步调用 ARMS 多个接口，把应用概览、JVM 和线程池指标汇总成一个字典。
        result = {}

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            from alibabacloud_arms20190808 import models as arms_models

            # 1. 查询应用概览指标
            try:
                # 将 request 的值保存下来，供后续流程判断或组装响应时使用。
                request = arms_models.QueryAppOverviewRequest(
                    # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
                    app_id=app_id,
                    # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
                    start_time=start_time,
                    # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
                    end_time=end_time,
                )
                # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
                response = client.query_app_overview(request)
                # 将 overview 的值保存下来，供后续流程判断或组装响应时使用。
                overview = response.to_map().get("body", {})

                # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
                result.update({
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "qps": overview.get("qps", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "latency_p95_ms": overview.get("rt_p95", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "latency_avg_ms": overview.get("rt_avg", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "error_rate": overview.get("error_rate", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "error_count": overview.get("error_count", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "total_requests": overview.get("total_count", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "cpu_usage": overview.get("cpu_usage", 0),
                    # 调用 overview.get 完成当前步骤需要的业务处理。
                    "memory_usage": overview.get("memory_usage", 0),
                })
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as e:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Failed to query app overview: %s", e)

            # 2. 查询 JVM 指标
            try:
                # 将 jvm_metrics 的值保存下来，供后续流程判断或组装响应时使用。
                jvm_metrics = self._query_jvm_metrics(client, app_id, start_time, end_time)
                # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
                result.update(jvm_metrics)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as e:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Failed to query JVM metrics: %s", e)

            # 3. 查询线程池指标
            try:
                # 将 thread_metrics 的值保存下来，供后续流程判断或组装响应时使用。
                thread_metrics = self._query_thread_metrics(client, app_id, start_time, end_time)
                # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
                result.update(thread_metrics)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as e:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Failed to query thread metrics: %s", e)

            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return result

        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as e:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Failed to query app metrics: %s, returning empty", e)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {}

    # 定义 _query_jvm_metrics 相关的处理逻辑，供流程或外部调用复用。
    def _query_jvm_metrics(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        client,
        # 执行当前业务步骤，推动流程继续向下游推进。
        app_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        start_time: int,
        # 执行当前业务步骤，推动流程继续向下游推进。
        end_time: int,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, Any]:
        """查询 JVM 指标"""
        # 查询 JVM 堆、非堆和 GC 指标，用来判断是否存在内存或 GC 压力。
        from alibabacloud_arms20190808 import models as arms_models

        # 查询 JVM 堆内存
        request = arms_models.QueryMetricRequest(
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=app_id,
            # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
            start_time=start_time,
            # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
            end_time=end_time,
            # 将 metric 的值保存下来，供后续流程判断或组装响应时使用。
            metric="jvm_heap_memory_used",
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = client.query_metric(request)
        # 将 heap_used 的值保存下来，供后续流程判断或组装响应时使用。
        heap_used = response.to_map().get("body", {}).get("values", [{}])

        # 查询 JVM 非堆内存
        request = arms_models.QueryMetricRequest(
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=app_id,
            # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
            start_time=start_time,
            # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
            end_time=end_time,
            # 将 metric 的值保存下来，供后续流程判断或组装响应时使用。
            metric="jvm_non_heap_memory_used",
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = client.query_metric(request)
        # 将 non_heap_used 的值保存下来，供后续流程判断或组装响应时使用。
        non_heap_used = response.to_map().get("body", {}).get("values", [{}])

        # 查询 GC 次数
        request = arms_models.QueryMetricRequest(
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=app_id,
            # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
            start_time=start_time,
            # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
            end_time=end_time,
            # 将 metric 的值保存下来，供后续流程判断或组装响应时使用。
            metric="jvm_gc_count",
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = client.query_metric(request)
        # 将 gc_count 的值保存下来，供后续流程判断或组装响应时使用。
        gc_count = response.to_map().get("body", {}).get("values", [{}])

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 调用 self._extract_latest_value 完成当前步骤需要的业务处理。
            "jvm_heap_used_mb": self._extract_latest_value(heap_used),
            # 调用 self._extract_latest_value 完成当前步骤需要的业务处理。
            "jvm_non_heap_used_mb": self._extract_latest_value(non_heap_used),
            # 调用 self._extract_latest_value 完成当前步骤需要的业务处理。
            "jvm_gc_count": self._extract_latest_value(gc_count),
        }

    # 定义 _query_thread_metrics 相关的处理逻辑，供流程或外部调用复用。
    def _query_thread_metrics(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        client,
        # 执行当前业务步骤，推动流程继续向下游推进。
        app_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        start_time: int,
        # 执行当前业务步骤，推动流程继续向下游推进。
        end_time: int,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, Any]:
        """查询线程池指标"""
        # 查询活跃线程数和线程池大小，用来判断线程池是否接近打满。
        from alibabacloud_arms20190808 import models as arms_models

        # 查询活跃线程数
        request = arms_models.QueryMetricRequest(
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=app_id,
            # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
            start_time=start_time,
            # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
            end_time=end_time,
            # 将 metric 的值保存下来，供后续流程判断或组装响应时使用。
            metric="thread_active_count",
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = client.query_metric(request)
        # 将 active_threads 的值保存下来，供后续流程判断或组装响应时使用。
        active_threads = response.to_map().get("body", {}).get("values", [{}])

        # 查询线程池大小
        request = arms_models.QueryMetricRequest(
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=app_id,
            # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
            start_time=start_time,
            # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
            end_time=end_time,
            # 将 metric 的值保存下来，供后续流程判断或组装响应时使用。
            metric="thread_pool_size",
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = client.query_metric(request)
        # 将 pool_size 的值保存下来，供后续流程判断或组装响应时使用。
        pool_size = response.to_map().get("body", {}).get("values", [{}])

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 调用 self._extract_latest_value 完成当前步骤需要的业务处理。
            "thread_active_count": self._extract_latest_value(active_threads),
            # 调用 self._extract_latest_value 完成当前步骤需要的业务处理。
            "thread_pool_size": self._extract_latest_value(pool_size),
        }

    # 定义 _extract_latest_value 相关的处理逻辑，供流程或外部调用复用。
    def _extract_latest_value(self, values: list) -> float:
        """从指标值列表中提取最新值"""
        # 从时间序列里取最新点并转成数字，缺失或格式异常时返回 0。
        if not values:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return 0.0
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return float(values[-1].get("value", 0))
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except (ValueError, TypeError):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return 0.0

    @async_retry(max_attempts=2)
    # 定义 get_alerts 相关的处理逻辑，供流程或外部调用复用。
    async def get_alerts(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
        app_id: str | None = None,
        # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
        start_time: int | None = None,
        # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
        end_time: int | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[dict[str, Any]]:
        """获取应用告警信息

        Args:
            app_id: ARMS 应用 ID
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）

        Returns:
            告警列表
        """
        # 查询 ARMS 最近告警事件，作为诊断时判断服务健康状态的补充证据。
        client = self._get_client()
        # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
        app_id = app_id or self._app_id

        # 根据 not app_id 判断当前流程该进入哪个处理分支。
        if not app_id:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []

        # 默认查询最近 1 小时
        now = int(time.time() * 1000)
        # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
        start_time = start_time or (now - 3600 * 1000)
        # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
        end_time = end_time or now

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 进入异步上下文管理区域，确保异步资源按约定释放。
            async with asyncio.timeout(self._timeout_seconds):
                # 将 loop 的值保存下来，供后续流程判断或组装响应时使用。
                loop = asyncio.get_event_loop()
                # 将 alerts 的值保存下来，供后续流程判断或组装响应时使用。
                alerts = await loop.run_in_executor(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    None,
                    # 调用 self._query_alerts 完成当前步骤需要的业务处理。
                    lambda: self._query_alerts(client, app_id, start_time, end_time),
                )
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return alerts

        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as e:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.error("ARMS alerts query failed: %s", e)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []

    # 定义 _query_alerts 相关的处理逻辑，供流程或外部调用复用。
    def _query_alerts(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        client,
        # 执行当前业务步骤，推动流程继续向下游推进。
        app_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        start_time: int,
        # 执行当前业务步骤，推动流程继续向下游推进。
        end_time: int,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[dict[str, Any]]:
        """查询告警的具体实现"""
        # 同步调用 ARMS 告警接口，并把云厂商字段整理成项目内部使用的告警结构。
        try:
            from alibabacloud_arms20190808 import models as arms_models

            # 查询告警事件
            request = arms_models.QueryAlertEventsRequest(
                # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
                app_id=app_id,
                # 将 start_time 的值保存下来，供后续流程判断或组装响应时使用。
                start_time=start_time,
                # 将 end_time 的值保存下来，供后续流程判断或组装响应时使用。
                end_time=end_time,
            )
            # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
            response = client.query_alert_events(request)
            # 将 events 的值保存下来，供后续流程判断或组装响应时使用。
            events = response.to_map().get("body", {}).get("events", [])

            # 将 alerts 的值保存下来，供后续流程判断或组装响应时使用。
            alerts = []
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for event in events:
                # 把当前结果追加到集合中，逐步构建最终输出。
                alerts.append({
                    # 调用 event.get 完成当前步骤需要的业务处理。
                    "alert_name": event.get("alert_name", ""),
                    # 调用 ALERT_SEVERITY_MAP.get 完成当前步骤需要的业务处理。
                    "severity": ALERT_SEVERITY_MAP.get(event.get("severity", ""), event.get("severity", "")),
                    # 调用 event.get 完成当前步骤需要的业务处理。
                    "status": event.get("status", ""),
                    # 调用 event.get 完成当前步骤需要的业务处理。
                    "start_time": event.get("start_time"),
                    # 调用 event.get 完成当前步骤需要的业务处理。
                    "end_time": event.get("end_time"),
                    # 调用 event.get 完成当前步骤需要的业务处理。
                    "message": event.get("message", ""),
                    # 调用 event.get 完成当前步骤需要的业务处理。
                    "tags": event.get("tags", {}),
                })

            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return alerts

        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as e:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Failed to query alerts: %s", e)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []


# 定义 ARMSMetricsProvider 组件，集中管理这个模块的状态和行为。
class ARMSMetricsProvider:
    """基于 ARMS 的监控指标提供者"""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, client: AliyunARMSClient) -> None:
        # 绑定 ARMS 客户端，把云监控查询能力包装成项目统一的指标 provider。
        self._client = client

    # 定义 get_metrics 相关的处理逻辑，供流程或外部调用复用。
    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """获取监控指标

        Args:
            alert_summary: 告警摘要文本
            group_id: 飞书群组 ID（可用于映射到不同的 ARMS 应用）

        Returns:
            包含监控指标、告警信息和健康状态的字典
        """
        # 并发查询应用指标和告警，再计算健康状态和可能瓶颈，供诊断方案参考。
        logger.info("Fetching ARMS metrics summary_chars=%s group_id=%s", len(alert_summary), group_id)

        # 并发查询指标和告警
        metrics_task = self._client.get_app_metrics()
        # 将 alerts_task 的值保存下来，供后续流程判断或组装响应时使用。
        alerts_task = self._client.get_alerts()

        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        metrics, alerts = await asyncio.gather(
            # 执行当前业务步骤，推动流程继续向下游推进。
            metrics_task,
            # 执行当前业务步骤，推动流程继续向下游推进。
            alerts_task,
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return_exceptions=True,
        )

        # 处理异常
        if isinstance(metrics, Exception):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Failed to get metrics: %s", metrics)
            # 将 metrics 的值保存下来，供后续流程判断或组装响应时使用。
            metrics = {}

        # 根据 isinstance(alerts, Exception) 判断当前流程该进入哪个处理分支。
        if isinstance(alerts, Exception):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Failed to get alerts: %s", alerts)
            # 将 alerts 的值保存下来，供后续流程判断或组装响应时使用。
            alerts = []

        # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
        monitor.record_tool_call("get_metrics", group_id)

        # 计算健康状态
        health_status = self._calculate_health_status(metrics, alerts)

        # 分析性能瓶颈
        bottlenecks = self._analyze_bottlenecks(metrics)

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            **metrics,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "alerts": alerts,
            # 调用 len 完成当前步骤需要的业务处理。
            "alert_count": len(alerts),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "health_status": health_status,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "bottlenecks": bottlenecks,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "provider": "aliyun_arms",
        }

    # 定义 _calculate_health_status 相关的处理逻辑，供流程或外部调用复用。
    def _calculate_health_status(self, metrics: dict, alerts: list) -> str:
        """计算应用健康状态

        根据指标和告警综合评估应用健康状态。
        """
        # 用严重告警、错误率、延迟、CPU 和内存等信号粗略判断应用是否健康。
        # 检查是否有严重告警
        critical_alerts = [a for a in alerts if a.get("severity") in ["严重", "错误"]]
        # 根据 critical_alerts 判断当前流程该进入哪个处理分支。
        if critical_alerts:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "CRITICAL"

        # 检查关键指标
        error_rate = metrics.get("error_rate", 0)
        # 将 latency_p95 的值保存下来，供后续流程判断或组装响应时使用。
        latency_p95 = metrics.get("latency_p95_ms", 0)
        # 将 cpu_usage 的值保存下来，供后续流程判断或组装响应时使用。
        cpu_usage = metrics.get("cpu_usage", 0)
        # 将 memory_usage 的值保存下来，供后续流程判断或组装响应时使用。
        memory_usage = metrics.get("memory_usage", 0)

        # 错误率超过 5% 视为不健康
        if error_rate > 0.05:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "UNHEALTHY"

        # 延迟超过 3 秒视为警告
        if latency_p95 > 3000:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "WARNING"

        # CPU 或内存超过 80% 视为警告
        if cpu_usage > 0.8 or memory_usage > 0.8:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "WARNING"

        # 有普通告警
        if alerts:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "WARNING"

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "HEALTHY"

    # 定义 _analyze_bottlenecks 相关的处理逻辑，供流程或外部调用复用。
    def _analyze_bottlenecks(self, metrics: dict) -> list[str]:
        """分析性能瓶颈"""
        # 从指标中提取可读的瓶颈描述，例如 CPU 高、延迟高或线程池打满。
        bottlenecks = []

        # 检查 CPU
        cpu_usage = metrics.get("cpu_usage", 0)
        # 根据 cpu_usage > 0.8 判断当前流程该进入哪个处理分支。
        if cpu_usage > 0.8:
            # 把当前结果追加到集合中，逐步构建最终输出。
            bottlenecks.append(f"CPU 使用率过高: {cpu_usage:.1%}")

        # 检查内存
        memory_usage = metrics.get("memory_usage", 0)
        # 根据 memory_usage > 0.8 判断当前流程该进入哪个处理分支。
        if memory_usage > 0.8:
            # 把当前结果追加到集合中，逐步构建最终输出。
            bottlenecks.append(f"内存使用率过高: {memory_usage:.1%}")

        # 检查延迟
        latency_p95 = metrics.get("latency_p95_ms", 0)
        # 根据 latency_p95 > 2000 判断当前流程该进入哪个处理分支。
        if latency_p95 > 2000:
            # 把当前结果追加到集合中，逐步构建最终输出。
            bottlenecks.append(f"P95 延迟过高: {latency_p95:.0f}ms")

        # 检查错误率
        error_rate = metrics.get("error_rate", 0)
        # 根据 error_rate > 0.01 判断当前流程该进入哪个处理分支。
        if error_rate > 0.01:
            # 把当前结果追加到集合中，逐步构建最终输出。
            bottlenecks.append(f"错误率过高: {error_rate:.2%}")

        # 检查线程池
        thread_active = metrics.get("thread_active_count", 0)
        # 将 thread_pool_size 的值保存下来，供后续流程判断或组装响应时使用。
        thread_pool_size = metrics.get("thread_pool_size", 0)
        # 根据 thread_pool_size > 0 and thread_active / thread_pool... 判断当前流程该进入哪个处理分支。
        if thread_pool_size > 0 and thread_active / thread_pool_size > 0.8:
            # 把当前结果追加到集合中，逐步构建最终输出。
            bottlenecks.append(f"线程池使用率过高: {thread_active}/{thread_pool_size}")

        # 检查 JVM 堆内存
        jvm_heap_used = metrics.get("jvm_heap_used_mb", 0)
        # 根据 jvm_heap_used > 1024: # 超过 1GB 判断当前流程该进入哪个处理分支。
        if jvm_heap_used > 1024:  # 超过 1GB
            # 把当前结果追加到集合中，逐步构建最终输出。
            bottlenecks.append(f"JVM 堆内存使用过高: {jvm_heap_used:.0f}MB")

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return bottlenecks
