from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# 根据 TYPE_CHECKING 判断当前流程该进入哪个处理分支。
if TYPE_CHECKING:
    from agent_sentinel.config import Settings
    from agent_sentinel.tools.base import ToolsProvider

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 build_tools_provider 相关的处理逻辑，供流程或外部调用复用。
def build_tools_provider(settings: Settings) -> ToolsProvider:
    # 根据配置选择实时数据来源，优先使用真实 provider，配置不完整时降级到 mock。
    from agent_sentinel.tools.mock_tools import MockToolsProvider

    # 将 provider 的值保存下来，供后续流程判断或组装响应时使用。
    provider = settings.tools_provider.strip().lower()

    # 根据 provider == "mcp" 判断当前流程该进入哪个处理分支。
    if provider == "mcp":
        # MCP provider 依赖外部进程，配置不完整时降级到 mock，保证诊断流程仍可演示和测试。
        if not settings.mcp_enabled:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("MCP tools provider requested but MCP_ENABLED is false, falling back to mock")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return MockToolsProvider()
        # 根据 not settings.mcp_server_command 判断当前流程该进入哪个处理分支。
        if not settings.mcp_server_command:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("MCP tools provider requested but MCP_SERVER_COMMAND is empty, falling back to mock")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return MockToolsProvider()
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return _build_mcp_provider(settings)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.error("Failed to initialize MCP provider: %s, falling back to mock", exc)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return MockToolsProvider()

    # 根据 provider != "aliyun" 判断当前流程该进入哪个处理分支。
    if provider != "aliyun":
        # 未显式选择真实云厂商 provider 时默认使用 mock，避免本地开发依赖云账号。
        logger.info("Using mock tools provider provider=%s", provider)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return MockToolsProvider()

    # 根据 not _validate_aliyun_config(settings) 判断当前流程该进入哪个处理分支。
    if not _validate_aliyun_config(settings):
        # 阿里云日志和指标任一侧配置完整即可启用，缺失时不让初始化失败扩散到主服务。
        logger.warning("Aliyun config incomplete, falling back to mock tools provider")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return MockToolsProvider()

    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _build_aliyun_provider(settings)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except ImportError as exc:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.warning("Aliyun SDK not available: %s, falling back to mock", exc)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return MockToolsProvider()
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception as exc:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.error("Failed to initialize aliyun provider: %s, falling back to mock", exc)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return MockToolsProvider()


# 定义 _validate_aliyun_config 相关的处理逻辑，供流程或外部调用复用。
def _validate_aliyun_config(settings: Settings) -> bool:
    # SLS 和 ARMS 是两个独立能力，只要配置了其中一个就可以构建组合 provider。
    # 检查阿里云日志或指标配置是否至少有一组完整，决定能否启用真实云数据源。
    sls_configured = bool(
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.aliyun_sls_access_key_id
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_sls_access_key_secret
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_sls_project
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_sls_logstore
    )
    # 将 arms_configured 的值保存下来，供后续流程判断或组装响应时使用。
    arms_configured = bool(
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.aliyun_arms_access_key_id
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_arms_access_key_secret
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_arms_app_id
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return sls_configured or arms_configured


# 定义 _build_mcp_provider 相关的处理逻辑，供流程或外部调用复用。
def _build_mcp_provider(settings: Settings) -> ToolsProvider:
    # 把全局配置转换成 MCP 工具配置，创建通过 MCP Server 查询外部数据的 provider。
    from agent_sentinel.tools.providers.mcp import MCPToolConfig, MCPToolsProvider

    # 将 settings 映射成 MCP provider 自己的配置对象，避免 provider 直接依赖全局 Settings。
    config = MCPToolConfig(
        # 将 command 的值保存下来，供后续流程判断或组装响应时使用。
        command=settings.mcp_server_command,
        # 将 args 的值保存下来，供后续流程判断或组装响应时使用。
        args=settings.mcp_server_args,
        # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        timeout_seconds=settings.mcp_timeout_seconds,
        # 将 sls_tool 的值保存下来，供后续流程判断或组装响应时使用。
        sls_tool=settings.mcp_sls_tool,
        # 将 prometheus_tool 的值保存下来，供后续流程判断或组装响应时使用。
        prometheus_tool=settings.mcp_prometheus_tool,
        # 将 prometheus_base_url 的值保存下来，供后续流程判断或组装响应时使用。
        prometheus_base_url=settings.mcp_prometheus_base_url,
    )
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info(
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "Using MCP tools provider command=%s args_count=%s sls_tool=%s prometheus_tool=%s",
        # 执行当前业务步骤，推动流程继续向下游推进。
        config.command,
        # 调用 len 完成当前步骤需要的业务处理。
        len(config.args),
        # 执行当前业务步骤，推动流程继续向下游推进。
        config.sls_tool,
        # 执行当前业务步骤，推动流程继续向下游推进。
        config.prometheus_tool,
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return MCPToolsProvider(config)


# 定义 _build_aliyun_provider 相关的处理逻辑，供流程或外部调用复用。
def _build_aliyun_provider(settings: Settings) -> ToolsProvider:
    # 按已配置的阿里云能力组装组合 provider，日志走 SLS，指标走 ARMS，缺失部分用 mock 补位。
    from agent_sentinel.tools.mock_tools import (
        # 执行当前业务步骤，推动流程继续向下游推进。
        MockLogsProvider,
        # 执行当前业务步骤，推动流程继续向下游推进。
        MockMetricsProvider,
        # 执行当前业务步骤，推动流程继续向下游推进。
        MockTopologyProvider,
    )

    # 将 logs_provider 的值保存下来，供后续流程判断或组装响应时使用。
    logs_provider = None
    # 根据 settings.aliyun_sls_access_key_id and settings.aliyu... 判断当前流程该进入哪个处理分支。
    if settings.aliyun_sls_access_key_id and settings.aliyun_sls_access_key_secret:
        # 日志查询由 SLS provider 负责；缺省时由后面的 mock provider 补位。
        from agent_sentinel.tools.providers.aliyun_sls import AliyunSLSClient, SLSLogsProvider

        # 将 sls_client 的值保存下来，供后续流程判断或组装响应时使用。
        sls_client = AliyunSLSClient(
            # 将 access_key_id 的值保存下来，供后续流程判断或组装响应时使用。
            access_key_id=settings.aliyun_sls_access_key_id,
            # 将 access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
            access_key_secret=settings.aliyun_sls_access_key_secret,
            # 将 endpoint 的值保存下来，供后续流程判断或组装响应时使用。
            endpoint=settings.aliyun_sls_endpoint,
            # 将 project 的值保存下来，供后续流程判断或组装响应时使用。
            project=settings.aliyun_sls_project,
            # 将 logstore 的值保存下来，供后续流程判断或组装响应时使用。
            logstore=settings.aliyun_sls_logstore,
            # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            timeout_seconds=settings.aliyun_sls_query_timeout_seconds,
            # 将 max_lines 的值保存下来，供后续流程判断或组装响应时使用。
            max_lines=settings.aliyun_sls_max_lines,
        )
        # 将 logs_provider 的值保存下来，供后续流程判断或组装响应时使用。
        logs_provider = SLSLogsProvider(sls_client)

    # 将 metrics_provider 的值保存下来，供后续流程判断或组装响应时使用。
    metrics_provider = None
    # 根据 settings.aliyun_arms_access_key_id and settings.aliy... 判断当前流程该进入哪个处理分支。
    if settings.aliyun_arms_access_key_id and settings.aliyun_arms_access_key_secret:
        # 指标查询由 ARMS provider 负责；拓扑当前仍使用 mock，保持组合接口完整。
        from agent_sentinel.tools.providers.aliyun_arms import AliyunARMSClient, ARMSMetricsProvider

        # 将 arms_client 的值保存下来，供后续流程判断或组装响应时使用。
        arms_client = AliyunARMSClient(
            # 将 access_key_id 的值保存下来，供后续流程判断或组装响应时使用。
            access_key_id=settings.aliyun_arms_access_key_id,
            # 将 access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
            access_key_secret=settings.aliyun_arms_access_key_secret,
            # 将 region_id 的值保存下来，供后续流程判断或组装响应时使用。
            region_id=settings.aliyun_arms_region_id,
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=settings.aliyun_arms_app_id,
            # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            timeout_seconds=settings.aliyun_arms_query_timeout_seconds,
        )
        # 将 metrics_provider 的值保存下来，供后续流程判断或组装响应时使用。
        metrics_provider = ARMSMetricsProvider(arms_client)

    from agent_sentinel.tools.providers.composite import CompositeToolsProvider

    # CompositeToolsProvider 屏蔽不同数据源差异，让上层 fetch_tools 节点按统一接口调用。
    return CompositeToolsProvider(
        # 将 metrics_provider 的值保存下来，供后续流程判断或组装响应时使用。
        metrics_provider=metrics_provider or MockMetricsProvider(),
        # 将 logs_provider 的值保存下来，供后续流程判断或组装响应时使用。
        logs_provider=logs_provider or MockLogsProvider(),
        # 将 topology_provider 的值保存下来，供后续流程判断或组装响应时使用。
        topology_provider=MockTopologyProvider(),
    )
