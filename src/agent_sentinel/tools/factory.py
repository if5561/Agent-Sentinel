from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_sentinel.config import Settings
    from agent_sentinel.tools.base import ToolsProvider

logger = logging.getLogger(__name__)


def build_tools_provider(settings: Settings) -> ToolsProvider:
    # 方法说明：构建并返回调用方需要的对象。
    from agent_sentinel.tools.mock_tools import MockToolsProvider

    provider = settings.tools_provider.strip().lower()

    if provider == "mcp":
        # MCP provider 依赖外部进程，配置不完整时降级到 mock，保证诊断流程仍可演示和测试。
        if not settings.mcp_enabled:
            logger.warning("MCP tools provider requested but MCP_ENABLED is false, falling back to mock")
            return MockToolsProvider()
        if not settings.mcp_server_command:
            logger.warning("MCP tools provider requested but MCP_SERVER_COMMAND is empty, falling back to mock")
            return MockToolsProvider()
        try:
            return _build_mcp_provider(settings)
        except Exception as exc:
            logger.error("Failed to initialize MCP provider: %s, falling back to mock", exc)
            return MockToolsProvider()

    if provider != "aliyun":
        # 未显式选择真实云厂商 provider 时默认使用 mock，避免本地开发依赖云账号。
        logger.info("Using mock tools provider provider=%s", provider)
        return MockToolsProvider()

    if not _validate_aliyun_config(settings):
        # 阿里云日志和指标任一侧配置完整即可启用，缺失时不让初始化失败扩散到主服务。
        logger.warning("Aliyun config incomplete, falling back to mock tools provider")
        return MockToolsProvider()

    try:
        return _build_aliyun_provider(settings)
    except ImportError as exc:
        logger.warning("Aliyun SDK not available: %s, falling back to mock", exc)
        return MockToolsProvider()
    except Exception as exc:
        logger.error("Failed to initialize aliyun provider: %s, falling back to mock", exc)
        return MockToolsProvider()


def _validate_aliyun_config(settings: Settings) -> bool:
    # SLS 和 ARMS 是两个独立能力，只要配置了其中一个就可以构建组合 provider。
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    sls_configured = bool(
        settings.aliyun_sls_access_key_id
        and settings.aliyun_sls_access_key_secret
        and settings.aliyun_sls_project
        and settings.aliyun_sls_logstore
    )
    arms_configured = bool(
        settings.aliyun_arms_access_key_id
        and settings.aliyun_arms_access_key_secret
        and settings.aliyun_arms_app_id
    )
    return sls_configured or arms_configured


def _build_mcp_provider(settings: Settings) -> ToolsProvider:
    # 方法说明：构建并返回调用方需要的对象。
    from agent_sentinel.tools.providers.mcp import MCPToolConfig, MCPToolsProvider

    # 将 settings 映射成 MCP provider 自己的配置对象，避免 provider 直接依赖全局 Settings。
    config = MCPToolConfig(
        command=settings.mcp_server_command,
        args=settings.mcp_server_args,
        timeout_seconds=settings.mcp_timeout_seconds,
        sls_tool=settings.mcp_sls_tool,
        prometheus_tool=settings.mcp_prometheus_tool,
        prometheus_base_url=settings.mcp_prometheus_base_url,
    )
    logger.info(
        "Using MCP tools provider command=%s args_count=%s sls_tool=%s prometheus_tool=%s",
        config.command,
        len(config.args),
        config.sls_tool,
        config.prometheus_tool,
    )
    return MCPToolsProvider(config)


def _build_aliyun_provider(settings: Settings) -> ToolsProvider:
    # 方法说明：构建并返回调用方需要的对象。
    from agent_sentinel.tools.mock_tools import (
        MockLogsProvider,
        MockMetricsProvider,
        MockTopologyProvider,
    )

    logs_provider = None
    if settings.aliyun_sls_access_key_id and settings.aliyun_sls_access_key_secret:
        # 日志查询由 SLS provider 负责；缺省时由后面的 mock provider 补位。
        from agent_sentinel.tools.providers.aliyun_sls import AliyunSLSClient, SLSLogsProvider

        sls_client = AliyunSLSClient(
            access_key_id=settings.aliyun_sls_access_key_id,
            access_key_secret=settings.aliyun_sls_access_key_secret,
            endpoint=settings.aliyun_sls_endpoint,
            project=settings.aliyun_sls_project,
            logstore=settings.aliyun_sls_logstore,
            timeout_seconds=settings.aliyun_sls_query_timeout_seconds,
            max_lines=settings.aliyun_sls_max_lines,
        )
        logs_provider = SLSLogsProvider(sls_client)

    metrics_provider = None
    if settings.aliyun_arms_access_key_id and settings.aliyun_arms_access_key_secret:
        # 指标查询由 ARMS provider 负责；拓扑当前仍使用 mock，保持组合接口完整。
        from agent_sentinel.tools.providers.aliyun_arms import AliyunARMSClient, ARMSMetricsProvider

        arms_client = AliyunARMSClient(
            access_key_id=settings.aliyun_arms_access_key_id,
            access_key_secret=settings.aliyun_arms_access_key_secret,
            region_id=settings.aliyun_arms_region_id,
            app_id=settings.aliyun_arms_app_id,
            timeout_seconds=settings.aliyun_arms_query_timeout_seconds,
        )
        metrics_provider = ARMSMetricsProvider(arms_client)

    from agent_sentinel.tools.providers.composite import CompositeToolsProvider

    # CompositeToolsProvider 屏蔽不同数据源差异，让上层 fetch_tools 节点按统一接口调用。
    return CompositeToolsProvider(
        metrics_provider=metrics_provider or MockMetricsProvider(),
        logs_provider=logs_provider or MockLogsProvider(),
        topology_provider=MockTopologyProvider(),
    )
