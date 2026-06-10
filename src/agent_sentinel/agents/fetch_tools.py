from __future__ import annotations

import logging

from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor

logger = logging.getLogger(__name__)

# 延迟初始化的 provider 单例
_provider = None


def _get_provider():
    """获取工具提供者（延迟初始化单例）"""
    # 方法说明：第一次需要实时数据时才创建工具 provider，后续复用同一个对象减少初始化成本。
    global _provider
    if _provider is None:
        from agent_sentinel.config import get_settings
        from agent_sentinel.tools.factory import build_tools_provider

        settings = get_settings()
        _provider = build_tools_provider(settings)
        logger.info("Tools provider initialized: %s", type(_provider).__name__)
    return _provider


async def fetch_live_data_node(state: DiagnosisState) -> DiagnosisState:
    # 方法说明：诊断中段补充实时现场数据，例如指标、日志和拓扑，帮助方案生成更贴近当前故障。
    logger.info("Node fetch_live_data started")
    group_id = state.get("chat_id")

    try:
        provider = _get_provider()
        live_data = await provider.fetch_all(state.get("alert_summary", ""), group_id=group_id)
    except Exception:
        monitor.record_error("tool_call_error")
        raise

    # 获取 provider 名称用于日志
    provider_name = type(provider).__name__

    logger.info("Node fetch_live_data completed keys=%s provider=%s", sorted(live_data.keys()), provider_name)
    return {
        "live_data": live_data,
        "messages": append_message(state, "assistant", f"实时指标、日志、拓扑数据已并发获取 (provider: {provider_name})。"),
        "evidence": append_evidence(state, f"Live metrics/logs/topology data fetched via {provider_name}."),
    }
