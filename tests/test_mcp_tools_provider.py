from __future__ import annotations

import asyncio
from typing import Any

import agent_sentinel.config as config
from agent_sentinel.config import Settings
from agent_sentinel.tools.factory import build_tools_provider
from agent_sentinel.tools.mock_tools import MockToolsProvider
from agent_sentinel.tools.providers.mcp import MCPToolConfig, MCPToolsProvider


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if name == "aliyun_sls_query_logs":
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "matches": [{"message": "timeout", "level": "ERROR"}],
                            "total": 1,
                            "query": "level:ERROR",
                        },
                    }
                ]
            }
        if name == "prometheus_query_metrics":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": '{"series":[{"metric":"http_errors_total","value":3}],"summary":"errors increased"}',
                    }
                ]
            }
        raise AssertionError(f"unexpected tool: {name}")


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_name": "Agent Sentinel",
        "app_host": "0.0.0.0",
        "app_port": 8000,
        "app_env": "test",
        "log_level": "INFO",
        "openai_api_key": "",
        "openai_base_url": None,
        "openai_model": "qwen-plus",
        "openai_temperature": 0,
        "openai_http_trust_env": False,
        "feishu_api_base_url": "https://open.feishu.cn",
        "feishu_app_id": None,
        "feishu_app_secret": None,
        "feishu_event_verification_token": None,
        "feishu_event_encrypt_key": None,
        "feishu_bot_name": "Analysis Bot",
        "feishu_allowed_chat_ids": [],
        "feishu_analyze_mention_only": True,
        "feishu_long_connection_enabled": False,
        "feishu_message_polling_enabled": False,
        "feishu_message_polling_interval_seconds": 5,
        "feishu_message_polling_page_size": 20,
        "feishu_webhook_url": None,
        "feishu_secret": None,
        "feishu_alert_enabled": False,
        "feishu_alert_title_prefix": "Agent Sentinel",
        "alert_analysis_enabled": True,
        "alert_analysis_title_prefix": "Alert Analysis",
        "alert_api_token": None,
        "alert_dedup_window_seconds": 60,
        "alert_store_limit": 100,
    }
    values.update(overrides)
    return Settings(**values)


def test_mcp_tools_provider_fetches_sls_logs_and_prometheus_metrics() -> None:
    async def run() -> None:
        client = FakeMCPClient()
        provider = MCPToolsProvider(
            MCPToolConfig(
                command="fake",
                sls_tool="aliyun_sls_query_logs",
                prometheus_tool="prometheus_query_metrics",
            ),
            client=client,  # type: ignore[arg-type]
        )

        data = await provider.fetch_all("order-service timeout", group_id="oc-chat")

        assert data["logs"]["provider"] == "mcp_aliyun_sls"
        assert data["logs"]["matches"][0]["message"] == "timeout"
        assert data["metrics"]["provider"] == "mcp_prometheus"
        assert data["metrics"]["series"][0]["metric"] == "http_errors_total"
        assert data["topology"]["service"] == "order-sync"
        assert [name for name, _args in client.calls] == [
            "prometheus_query_metrics",
            "aliyun_sls_query_logs",
        ]
        assert client.calls[0][1]["alert_summary"] == "order-service timeout"

    asyncio.run(run())


def test_factory_builds_mcp_provider_when_enabled() -> None:
    provider = build_tools_provider(
        _settings(
            tools_provider="mcp",
            mcp_enabled=True,
            mcp_server_command="python",
            mcp_server_args=["-m", "agent_sentinel_mcp_server"],
        )
    )

    assert isinstance(provider, MCPToolsProvider)


def test_factory_falls_back_to_mock_when_mcp_disabled() -> None:
    provider = build_tools_provider(
        _settings(
            tools_provider="mcp",
            mcp_enabled=False,
            mcp_server_command="python",
        )
    )

    assert isinstance(provider, MockToolsProvider)


def test_mcp_server_args_support_shell_style_split(monkeypatch) -> None:
    monkeypatch.setenv("MCP_SERVER_ARGS", "-m agent_sentinel_mcp_server")

    assert config._to_args("-m agent_sentinel_mcp_server") == ["-m", "agent_sentinel_mcp_server"]  # noqa: SLF001
