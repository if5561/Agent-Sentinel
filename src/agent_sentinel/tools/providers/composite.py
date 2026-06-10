from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_sentinel.tools.base import LogsProvider, MetricsProvider, TopologyProvider

logger = logging.getLogger(__name__)


class CompositeToolsProvider:
    """组合工具提供者，支持部分降级

    可以将不同的 MetricsProvider、LogsProvider、TopologyProvider 组合在一起。
    未提供的子 provider 会在工厂函数中使用 Mock 实现。
    """

    def __init__(
        self,
        metrics_provider: MetricsProvider,
        logs_provider: LogsProvider,
        topology_provider: TopologyProvider,
    ) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self._metrics = metrics_provider
        self._logs = logs_provider
        self._topology = topology_provider

    @property
    def metrics(self) -> MetricsProvider:
        # 方法说明：返回当前组合 provider 使用的指标查询实现。
        return self._metrics

    @property
    def logs(self) -> LogsProvider:
        # 方法说明：返回当前组合 provider 使用的日志查询实现。
        return self._logs

    @property
    def topology(self) -> TopologyProvider:
        # 方法说明：返回当前组合 provider 使用的拓扑查询实现。
        return self._topology

    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """并发获取所有数据"""
        # 方法说明：并发查询指标、日志和拓扑，把三个数据源的结果合并成诊断节点可消费的现场数据。
        metrics, logs, topology = await asyncio.gather(
            self._metrics.get_metrics(alert_summary, group_id),
            self._logs.query_logs(alert_summary, group_id),
            self._topology.get_topology(alert_summary, group_id),
        )
        return {"metrics": metrics, "logs": logs, "topology": topology}
