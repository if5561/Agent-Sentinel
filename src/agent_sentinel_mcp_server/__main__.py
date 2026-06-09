from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

from agent_sentinel.config import get_settings
from agent_sentinel.tools.providers.aliyun_sls import AliyunSLSClient, SLSLogsProvider

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stderr)
    server = AgentSentinelMCPServer()
    await server.serve()


class AgentSentinelMCPServer:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def serve(self) -> None:
        while True:
            request = await _read_message()
            if request is None:
                return
            response = await self._handle_request(request)
            if response is not None:
                await _write_message(response)

    async def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if request_id is None:
            return None
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "agent-sentinel-mcp-server", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": _tool_definitions()}
            elif method == "tools/call":
                result = await self._call_tool(params)
            else:
                return _error_response(request_id, -32601, f"Method not found: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            logger.exception("MCP request failed method=%s", method)
            return _error_response(request_id, -32000, str(exc))

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name == "aliyun_sls_query_logs":
            payload = await query_aliyun_sls(arguments)
        elif name == "prometheus_query_metrics":
            payload = await query_prometheus(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
        return {"content": [{"type": "json", "json": payload}]}


async def query_aliyun_sls(arguments: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    if not (
        settings.aliyun_sls_access_key_id
        and settings.aliyun_sls_access_key_secret
        and settings.aliyun_sls_project
        and settings.aliyun_sls_logstore
    ):
        return {
            "provider": "mcp_aliyun_sls",
            "matches": [],
            "total": 0,
            "error": "Aliyun SLS config is incomplete",
        }

    provider = SLSLogsProvider(
        AliyunSLSClient(
            access_key_id=settings.aliyun_sls_access_key_id,
            access_key_secret=settings.aliyun_sls_access_key_secret,
            endpoint=settings.aliyun_sls_endpoint,
            project=settings.aliyun_sls_project,
            logstore=settings.aliyun_sls_logstore,
            timeout_seconds=settings.aliyun_sls_query_timeout_seconds,
            max_lines=_int_arg(arguments, "max_lines", settings.aliyun_sls_max_lines),
        )
    )
    return await provider.query_logs(str(arguments.get("alert_summary") or ""), group_id=_str_or_none(arguments.get("group_id")))


async def query_prometheus(arguments: dict[str, Any]) -> dict[str, Any]:
    base_url = (
        _str_or_none(arguments.get("prometheus_base_url"))
        or os.getenv("PROMETHEUS_BASE_URL")
        or os.getenv("MCP_PROMETHEUS_BASE_URL")
        or "http://127.0.0.1:9090"
    ).rstrip("/")
    alert_summary = str(arguments.get("alert_summary") or "")
    query = _str_or_none(arguments.get("promql")) or _build_default_promql(alert_summary)
    timeout_seconds = _int_arg(arguments, "timeout_seconds", 15)
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(f"{base_url}/api/v1/query", params={"query": query})
        response.raise_for_status()
        data = response.json()
    result = data.get("data", {}).get("result", []) if isinstance(data, dict) else []
    return {
        "provider": "mcp_prometheus",
        "query": query,
        "result_type": data.get("data", {}).get("resultType") if isinstance(data, dict) else None,
        "series": result,
        "summary": _summarize_prometheus_result(result),
    }


def _build_default_promql(alert_summary: str) -> str:
    lowered = alert_summary.lower()
    if any(token in lowered for token in ("5xx", "500", "error", "错误", "异常")):
        return 'sum(rate(http_requests_total{status=~"5.."}[5m]))'
    if any(token in lowered for token in ("latency", "p99", "p95", "延迟", "耗时", "超时")):
        return 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))'
    if any(token in lowered for token in ("cpu", "CPU")):
        return '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
    if any(token in lowered for token in ("memory", "mem", "内存")):
        return "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100"
    return "up"


def _summarize_prometheus_result(result: Any) -> str:
    if not isinstance(result, list):
        return "Prometheus returned a non-list result."
    if not result:
        return "Prometheus returned no matching series."
    return f"Prometheus returned {len(result)} series."


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "aliyun_sls_query_logs",
            "description": "Query Aliyun SLS logs for an alert summary.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alert_summary": {"type": "string"},
                    "group_id": {"type": "string"},
                    "max_lines": {"type": "integer"},
                },
            },
        },
        {
            "name": "prometheus_query_metrics",
            "description": "Query Prometheus metrics for an alert summary or explicit PromQL.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alert_summary": {"type": "string"},
                    "group_id": {"type": "string"},
                    "window": {"type": "string"},
                    "promql": {"type": "string"},
                    "prometheus_base_url": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
            },
        },
    ]


async def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not line:
            return None
        stripped = line.decode("ascii", errors="ignore").strip()
        if not stripped:
            break
        key, _, value = stripped.partition(":")
        headers[key.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = await asyncio.to_thread(sys.stdin.buffer.read, length)
    return json.loads(body.decode("utf-8"))


async def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header + body)
    await asyncio.to_thread(sys.stdout.buffer.flush)


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_arg(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    asyncio.run(main())
