from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_sentinel.tools.base import LogsProvider, MetricsProvider, TopologyProvider

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 CompositeToolsProvider 组件，集中管理这个模块的状态和行为。
class CompositeToolsProvider:
    """组合工具提供者，支持部分降级

    可以将不同的 MetricsProvider、LogsProvider、TopologyProvider 组合在一起。
    未提供的子 provider 会在工厂函数中使用 Mock 实现。
    """

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        metrics_provider: MetricsProvider,
        # 执行当前业务步骤，推动流程继续向下游推进。
        logs_provider: LogsProvider,
        # 执行当前业务步骤，推动流程继续向下游推进。
        topology_provider: TopologyProvider,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 把指标、日志和拓扑三个 provider 组合到一起，允许它们来自不同真实或 mock 后端。
        self._metrics = metrics_provider
        # 将 self._logs 的值保存下来，供后续流程判断或组装响应时使用。
        self._logs = logs_provider
        # 将 self._topology 的值保存下来，供后续流程判断或组装响应时使用。
        self._topology = topology_provider

    @property
    # 定义 metrics 相关的处理逻辑，供流程或外部调用复用。
    def metrics(self) -> MetricsProvider:
        # 返回当前组合 provider 使用的指标查询实现。
        return self._metrics

    @property
    # 定义 logs 相关的处理逻辑，供流程或外部调用复用。
    def logs(self) -> LogsProvider:
        # 返回当前组合 provider 使用的日志查询实现。
        return self._logs

    @property
    # 定义 topology 相关的处理逻辑，供流程或外部调用复用。
    def topology(self) -> TopologyProvider:
        # 返回当前组合 provider 使用的拓扑查询实现。
        return self._topology

    # 定义 fetch_all 相关的处理逻辑，供流程或外部调用复用。
    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """并发获取所有数据"""
        # 并发查询指标、日志和拓扑，把三个数据源的结果合并成诊断节点可消费的现场数据。
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
