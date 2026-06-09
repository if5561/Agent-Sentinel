from __future__ import annotations

import asyncio
from typing import Any

import agent_sentinel_mcp_server.__main__ as mcp_server


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {"job": "api"}, "value": [1, "3"]}],
            },
        }


class FakeAsyncClient:
    last_url = ""
    last_params: dict[str, Any] = {}

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(self, url: str, params: dict[str, Any]) -> FakeResponse:
        FakeAsyncClient.last_url = url
        FakeAsyncClient.last_params = params
        return FakeResponse()


def test_prometheus_mcp_tool_queries_http_api(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setattr(mcp_server.httpx, "AsyncClient", FakeAsyncClient)

        result = await mcp_server.query_prometheus(
            {
                "alert_summary": "api 500 error",
                "prometheus_base_url": "http://prometheus:9090",
            }
        )

        assert FakeAsyncClient.last_url == "http://prometheus:9090/api/v1/query"
        assert "http_requests_total" in FakeAsyncClient.last_params["query"]
        assert result["provider"] == "mcp_prometheus"
        assert result["summary"] == "Prometheus returned 1 series."

    asyncio.run(run())


def test_sls_mcp_tool_returns_error_when_config_incomplete(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.delenv("ALIYUN_SLS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("ALIYUN_SLS_ACCESS_KEY_SECRET", raising=False)
        monkeypatch.delenv("ALIYUN_SLS_PROJECT", raising=False)
        monkeypatch.delenv("ALIYUN_SLS_LOGSTORE", raising=False)

        result = await mcp_server.query_aliyun_sls({"alert_summary": "error"})

        assert result["provider"] == "mcp_aliyun_sls"
        assert result["total"] == 0
        assert "incomplete" in result["error"]

    asyncio.run(run())
