from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.tools.mock_tools import MockTopologyProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MCPToolConfig:
    command: str
    args: list[str] = field(default_factory=list)
    timeout_seconds: int = 20
    sls_tool: str = "aliyun_sls_query_logs"
    prometheus_tool: str = "prometheus_query_metrics"
    prometheus_base_url: str | None = None


class MCPClientError(RuntimeError):
    pass


class MCPStdioClient:
    """Minimal MCP JSON-RPC client for stdio servers."""

    def __init__(self, config: MCPToolConfig) -> None:
        self._config = config
        self._request_id = 0
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self._lock:
            await self._ensure_started()
            return await self._request(
                "tools/call",
                {
                    "name": name,
                    "arguments": arguments,
                },
            )

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _ensure_started(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        if not self._config.command:
            raise MCPClientError("MCP server command is empty")
        self._process = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-sentinel", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        process = self._active_process()
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._write_message(process, payload)

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        process = self._active_process()
        self._request_id += 1
        request_id = self._request_id
        await self._write_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        while True:
            response = await asyncio.wait_for(self._read_message(process), timeout=self._config.timeout_seconds)
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise MCPClientError(f"MCP {method} failed: {response['error']}")
            return response.get("result")

    def _active_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is None or process.returncode is not None:
            raise MCPClientError("MCP server is not running")
        if process.stdin is None or process.stdout is None:
            raise MCPClientError("MCP server stdio is unavailable")
        return process

    async def _write_message(self, process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise MCPClientError("MCP server stdin is unavailable")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        process.stdin.write(header + body)
        await process.stdin.drain()

    async def _read_message(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        if process.stdout is None:
            raise MCPClientError("MCP server stdout is unavailable")
        headers: dict[str, str] = {}
        while True:
            line = await process.stdout.readline()
            if not line:
                raise MCPClientError("MCP server closed stdout")
            stripped = line.decode("ascii", errors="ignore").strip()
            if not stripped:
                break
            key, _, value = stripped.partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            raise MCPClientError("MCP response missing Content-Length")
        raw = await process.stdout.readexactly(length)
        return json.loads(raw.decode("utf-8"))


class MCPLogsProvider:
    def __init__(self, client: MCPStdioClient, config: MCPToolConfig) -> None:
        self._client = client
        self._config = config

    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        logger.info("Querying MCP SLS logs summary_chars=%s group_id=%s", len(alert_summary), group_id)
        result = await self._client.call_tool(
            self._config.sls_tool,
            {
                "alert_summary": alert_summary,
                "group_id": group_id,
                "max_lines": 100,
            },
        )
        monitor.record_tool_call("query_logs", group_id)
        payload = _extract_tool_payload(result)
        if isinstance(payload, dict):
            payload.setdefault("provider", "mcp_aliyun_sls")
            payload.setdefault("tool", self._config.sls_tool)
            return payload
        return {
            "provider": "mcp_aliyun_sls",
            "tool": self._config.sls_tool,
            "matches": [],
            "total": 0,
            "raw": payload,
        }


class MCPMetricsProvider:
    def __init__(self, client: MCPStdioClient, config: MCPToolConfig) -> None:
        self._client = client
        self._config = config

    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        logger.info("Querying MCP Prometheus metrics summary_chars=%s group_id=%s", len(alert_summary), group_id)
        result = await self._client.call_tool(
            self._config.prometheus_tool,
            {
                "alert_summary": alert_summary,
                "group_id": group_id,
                "window": "30m",
                "prometheus_base_url": self._config.prometheus_base_url,
            },
        )
        monitor.record_tool_call("get_metrics", group_id)
        payload = _extract_tool_payload(result)
        if isinstance(payload, dict):
            payload.setdefault("provider", "mcp_prometheus")
            payload.setdefault("tool", self._config.prometheus_tool)
            return payload
        return {
            "provider": "mcp_prometheus",
            "tool": self._config.prometheus_tool,
            "series": [],
            "raw": payload,
        }


class MCPToolsProvider:
    def __init__(self, config: MCPToolConfig, client: MCPStdioClient | None = None) -> None:
        self._config = config
        self._client = client or MCPStdioClient(config)
        self._metrics = MCPMetricsProvider(self._client, config)
        self._logs = MCPLogsProvider(self._client, config)
        self._topology = MockTopologyProvider()

    @property
    def metrics(self) -> MCPMetricsProvider:
        return self._metrics

    @property
    def logs(self) -> MCPLogsProvider:
        return self._logs

    @property
    def topology(self) -> MockTopologyProvider:
        return self._topology

    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        metrics, logs, topology = await asyncio.gather(
            self._metrics.get_metrics(alert_summary, group_id),
            self._logs.query_logs(alert_summary, group_id),
            self._topology.get_topology(alert_summary, group_id),
        )
        return {"metrics": metrics, "logs": logs, "topology": topology}


def _extract_tool_payload(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result
    first = content[0]
    if not isinstance(first, dict):
        return result
    if first.get("type") == "json":
        return first.get("json")
    if first.get("type") == "text":
        text = str(first.get("text", ""))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return result
