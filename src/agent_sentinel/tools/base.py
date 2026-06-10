from __future__ import annotations

from typing import Any, Protocol


class MetricsProvider(Protocol):
    """监控指标提供者接口"""

    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """获取监控指标"""
        # 定义“查询监控指标”的统一入口，不同实现可以去 Prometheus、ARMS 或 mock 数据源取数。
        ...


class LogsProvider(Protocol):
    """日志查询提供者接口"""

    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """查询日志"""
        # 定义“查询日志证据”的统一入口，不同实现可以去 SLS、MCP 或 mock 日志源取数。
        ...


class TopologyProvider(Protocol):
    """拓扑信息提供者接口"""

    async def get_topology(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """获取服务拓扑"""
        # 定义“查询服务依赖拓扑”的统一入口，用来辅助判断故障是否来自上下游依赖。
        ...


class ToolsProvider(Protocol):
    """完整的工具提供者接口"""

    @property
    def metrics(self) -> MetricsProvider:
        # 暴露指标 provider，让上层不用关心真实实现来自哪个平台。
        ...

    @property
    def logs(self) -> LogsProvider:
        # 暴露日志 provider，让诊断节点用统一方式查询错误日志。
        ...

    @property
    def topology(self) -> TopologyProvider:
        # 暴露拓扑 provider，让诊断节点用统一方式查询服务依赖关系。
        ...

    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """并发获取所有数据"""
        # 定义“一次性查询指标、日志和拓扑”的入口，方便诊断节点并发补齐现场数据。
        ...
