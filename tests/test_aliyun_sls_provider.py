from __future__ import annotations

import asyncio
from typing import Any

from agent_sentinel.tools.providers.aliyun_sls import AliyunSLSClient, SLSLogsProvider


class FakeLog:
    def get_time(self) -> int:
        return 123

    def get_contents(self) -> dict[str, Any]:
        return {"level": "ERROR", "message": "App build started", "service": "agent-sentinel"}


class FakeResult:
    def get_logs(self) -> list[FakeLog]:
        return [FakeLog()]


class FakeSDKClient:
    def __init__(self) -> None:
        self.request = None

    def get_logs(self, request):
        self.request = request
        return FakeResult()


class FakeDockerStdoutLog:
    def get_time(self) -> int:
        return 456

    def get_contents(self) -> dict[str, Any]:
        return {
            "content": 'INFO:     127.0.0.1:38658 - "GET /metrics HTTP/1.1" 200 OK',
            "_container_name_": "agent-sentinel",
        }


class FakeDockerStdoutResult:
    def get_logs(self) -> list[FakeDockerStdoutLog]:
        return [FakeDockerStdoutLog()]


class FakeDockerStdoutSDKClient:
    def __init__(self) -> None:
        self.request = None

    def get_logs(self, request):
        self.request = request
        return FakeDockerStdoutResult()


def test_aliyun_sls_client_uses_get_logs_request() -> None:
    async def run() -> None:
        sdk = FakeSDKClient()
        client = AliyunSLSClient(
            access_key_id="ak",
            access_key_secret="sk",
            endpoint="cn-hangzhou.log.aliyuncs.com",
            project="agentsentinel",
            logstore="agent-sentinel-app",
        )
        client._client = sdk  # type: ignore[attr-defined]  # noqa: SLF001

        logs = await client.query_logs("App build started", max_lines=3)

        assert sdk.request.project == "agentsentinel"
        assert sdk.request.logstore == "agent-sentinel-app"
        assert sdk.request.query == "App build started"
        assert sdk.request.line == 3
        assert logs[0]["message"] == "App build started"

    asyncio.run(run())


def test_aliyun_sls_client_uses_docker_stdout_content_as_message() -> None:
    async def run() -> None:
        sdk = FakeDockerStdoutSDKClient()
        client = AliyunSLSClient(
            access_key_id="ak",
            access_key_secret="sk",
            endpoint="cn-hangzhou.log.aliyuncs.com",
            project="agentsentinel",
            logstore="agent-sentinel-app",
        )
        client._client = sdk  # type: ignore[attr-defined]  # noqa: SLF001

        logs = await client.query_logs('"GET /metrics"', max_lines=3)

        assert logs[0]["message"] == 'INFO:     127.0.0.1:38658 - "GET /metrics HTTP/1.1" 200 OK'

    asyncio.run(run())


class RecordingClient:
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = responses
        self.queries: list[str] = []

    async def query_logs(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.responses.pop(0) if self.responses else []


def test_sls_logs_provider_prefers_exact_query_for_identifiers() -> None:
    async def run() -> None:
        client = RecordingClient(
            [
                [
                    {
                        "timestamp": 123,
                        "level": "",
                        "service": "agent-sentinel",
                        "message": "codex-sls-mcp-test-1780547514",
                        "trace_id": "",
                    }
                ]
            ]
        )
        provider = SLSLogsProvider(client)  # type: ignore[arg-type]

        result = await provider.query_logs("codex-sls-mcp-test-1780547514")

        assert client.queries == ['"codex-sls-mcp-test-1780547514"']
        assert result["total"] == 1
        assert result["query"] == '"codex-sls-mcp-test-1780547514"'

    asyncio.run(run())


def test_sls_logs_provider_falls_back_to_keyword_query() -> None:
    async def run() -> None:
        client = RecordingClient(
            [
                [],
                [
                    {
                        "timestamp": 123,
                        "level": "ERROR",
                        "service": "agent-sentinel",
                        "message": "App build started",
                        "trace_id": "",
                    }
                ],
            ]
        )
        provider = SLSLogsProvider(client)  # type: ignore[arg-type]

        result = await provider.query_logs("App build started")

        assert client.queries == ['"App build started"', '("App" OR "build" OR "started")']
        assert result["total"] == 1
        assert result["query"] == '("App" OR "build" OR "started")'

    asyncio.run(run())
