from __future__ import annotations

import logging

from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 延迟初始化的 provider 单例
_provider = None


# 定义 _get_provider 相关的处理逻辑，供流程或外部调用复用。
def _get_provider():
    """获取工具提供者（延迟初始化单例）"""
    # 第一次需要实时数据时才创建工具 provider，后续复用同一个对象减少初始化成本。
    global _provider
    # 根据 _provider is None 判断当前流程该进入哪个处理分支。
    if _provider is None:
        from agent_sentinel.config import get_settings
        from agent_sentinel.tools.factory import build_tools_provider

        # 将 settings 的值保存下来，供后续流程判断或组装响应时使用。
        settings = get_settings()
        # 将 _provider 的值保存下来，供后续流程判断或组装响应时使用。
        _provider = build_tools_provider(settings)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Tools provider initialized: %s", type(_provider).__name__)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return _provider


# 定义 fetch_live_data_node 相关的处理逻辑，供流程或外部调用复用。
async def fetch_live_data_node(state: DiagnosisState) -> DiagnosisState:
    # 诊断中段补充实时现场数据，例如指标、日志和拓扑，帮助方案生成更贴近当前故障。
    logger.info("Node fetch_live_data started")
    # 将 group_id 的值保存下来，供后续流程判断或组装响应时使用。
    group_id = state.get("chat_id")

    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 将 provider 的值保存下来，供后续流程判断或组装响应时使用。
        provider = _get_provider()
        # 将 live_data 的值保存下来，供后续流程判断或组装响应时使用。
        live_data = await provider.fetch_all(state.get("alert_summary", ""), group_id=group_id)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception:
        # 调用 monitor.record_error 完成当前步骤需要的业务处理。
        monitor.record_error("tool_call_error")
        # 执行当前业务步骤，推动流程继续向下游推进。
        raise

    # 获取 provider 名称用于日志
    provider_name = type(provider).__name__

    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Node fetch_live_data completed keys=%s provider=%s", sorted(live_data.keys()), provider_name)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "live_data": live_data,
        # 调用 append_message 完成当前步骤需要的业务处理。
        "messages": append_message(state, "assistant", f"实时指标、日志、拓扑数据已并发获取 (provider: {provider_name})。"),
        # 调用 append_evidence 完成当前步骤需要的业务处理。
        "evidence": append_evidence(state, f"Live metrics/logs/topology data fetched via {provider_name}."),
    }
