from __future__ import annotations

from typing import Any, Protocol


class MetricsProvider(Protocol):
    """监控指标提供者接口"""

    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """获取监控指标"""
        # 方法说明：读取并返回当前流程需要的数据。
        ...


class LogsProvider(Protocol):
    """日志查询提供者接口"""

    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """查询日志"""
        # 方法说明：从配置的后端或数据集中检索匹配内容。
        ...


class TopologyProvider(Protocol):
    """拓扑信息提供者接口"""

    async def get_topology(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """获取服务拓扑"""
        # 方法说明：读取并返回当前流程需要的数据。
        ...


class ToolsProvider(Protocol):
    """完整的工具提供者接口"""

    @property
    def metrics(self) -> MetricsProvider:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        ...

    @property
    def logs(self) -> LogsProvider:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        ...

    @property
    def topology(self) -> TopologyProvider:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        ...

    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """并发获取所有数据"""
        # 方法说明：读取并返回当前流程需要的数据。
        ...
