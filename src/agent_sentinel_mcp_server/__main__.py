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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 main 相关的处理逻辑，供流程或外部调用复用。
async def main() -> None:
    # 初始化日志并启动 MCP 服务主循环，让外部 Agent 可以通过标准输入输出调用工具。
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stderr)
    # 将 server 的值保存下来，供后续流程判断或组装响应时使用。
    server = AgentSentinelMCPServer()
    # 等待异步操作完成，再继续推进当前业务流程。
    await server.serve()


# 定义 AgentSentinelMCPServer 组件，集中管理这个模块的状态和行为。
class AgentSentinelMCPServer:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self) -> None:
        # 读取项目配置，后续工具调用会用这些配置连接日志和指标后端。
        self._settings = get_settings()

    # 定义 serve 相关的处理逻辑，供流程或外部调用复用。
    async def serve(self) -> None:
        # 持续读取 MCP 客户端请求，处理后按 JSON-RPC 协议写回响应。
        while True:
            # 将 request 的值保存下来，供后续流程判断或组装响应时使用。
            request = await _read_message()
            # 根据 request is None 判断当前流程该进入哪个处理分支。
            if request is None:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return
            # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
            response = await self._handle_request(request)
            # 根据 response is not None 判断当前流程该进入哪个处理分支。
            if response is not None:
                # 等待异步操作完成，再继续推进当前业务流程。
                await _write_message(response)

    # 定义 _handle_request 相关的处理逻辑，供流程或外部调用复用。
    async def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        # 按 MCP 方法名分发请求，支持初始化、工具列表和工具调用三类消息。
        request_id = request.get("id")
        # 将 method 的值保存下来，供后续流程判断或组装响应时使用。
        method = request.get("method")
        # 将 params 的值保存下来，供后续流程判断或组装响应时使用。
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        # 根据 request_id is None 判断当前流程该进入哪个处理分支。
        if request_id is None:
            return None
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 根据 method == "initialize" 判断当前流程该进入哪个处理分支。
            if method == "initialize":
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "protocolVersion": "2024-11-05",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "capabilities": {"tools": {}},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "serverInfo": {"name": "agent-sentinel-mcp-server", "version": "0.1.0"},
                }
            # 根据 method == "tools/list" 判断当前流程该进入哪个处理分支。
            elif method == "tools/list":
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = {"tools": _tool_definitions()}
            # 根据 method == "tools/call" 判断当前流程该进入哪个处理分支。
            elif method == "tools/call":
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = await self._call_tool(params)
            # 处理前面条件都不满足时的默认分支。
            else:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return _error_response(request_id, -32601, f"Method not found: {method}")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("MCP request failed method=%s", method)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return _error_response(request_id, -32000, str(exc))

    # 定义 _call_tool 相关的处理逻辑，供流程或外部调用复用。
    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        # 根据工具名称调用对应的数据查询逻辑，并包装成 MCP 要求的返回格式。
        name = str(params.get("name") or "")
        # 将 arguments 的值保存下来，供后续流程判断或组装响应时使用。
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        # 根据 name == "aliyun_sls_query_logs" 判断当前流程该进入哪个处理分支。
        if name == "aliyun_sls_query_logs":
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = await query_aliyun_sls(arguments)
        # 根据 name == "prometheus_query_metrics" 判断当前流程该进入哪个处理分支。
        elif name == "prometheus_query_metrics":
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = await query_prometheus(arguments)
        # 处理前面条件都不满足时的默认分支。
        else:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise ValueError(f"Unknown tool: {name}")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"content": [{"type": "json", "json": payload}]}


# 定义 query_aliyun_sls 相关的处理逻辑，供流程或外部调用复用。
async def query_aliyun_sls(arguments: dict[str, Any]) -> dict[str, Any]:
    # 通过 MCP 查询阿里云 SLS 日志，为告警诊断补充原始日志证据。
    settings = get_settings()
    # 根据 not ( 判断当前流程该进入哪个处理分支。
    if not (
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.aliyun_sls_access_key_id
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_sls_access_key_secret
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_sls_project
        # 执行当前业务步骤，推动流程继续向下游推进。
        and settings.aliyun_sls_logstore
    ):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "provider": "mcp_aliyun_sls",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "matches": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "total": 0,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "error": "Aliyun SLS config is incomplete",
        }

    # 将 provider 的值保存下来，供后续流程判断或组装响应时使用。
    provider = SLSLogsProvider(
        # 调用 AliyunSLSClient 完成当前步骤需要的业务处理。
        AliyunSLSClient(
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
            max_lines=_int_arg(arguments, "max_lines", settings.aliyun_sls_max_lines),
        )
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return await provider.query_logs(str(arguments.get("alert_summary") or ""), group_id=_str_or_none(arguments.get("group_id")))


# 定义 query_prometheus 相关的处理逻辑，供流程或外部调用复用。
async def query_prometheus(arguments: dict[str, Any]) -> dict[str, Any]:
    # 通过 MCP 查询 Prometheus 指标，支持显式 PromQL，也能按告警摘要生成默认查询。
    base_url = (
        # 调用 _str_or_none 完成当前步骤需要的业务处理。
        _str_or_none(arguments.get("prometheus_base_url"))
        # 调用 os.getenv 完成当前步骤需要的业务处理。
        or os.getenv("PROMETHEUS_BASE_URL")
        # 调用 os.getenv 完成当前步骤需要的业务处理。
        or os.getenv("MCP_PROMETHEUS_BASE_URL")
        # 执行当前业务步骤，推动流程继续向下游推进。
        or "http://127.0.0.1:9090"
    # 调用 rstrip 完成当前步骤需要的业务处理。
    ).rstrip("/")
    # 将 alert_summary 的值保存下来，供后续流程判断或组装响应时使用。
    alert_summary = str(arguments.get("alert_summary") or "")
    # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
    query = _str_or_none(arguments.get("promql")) or _build_default_promql(alert_summary)
    # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    timeout_seconds = _int_arg(arguments, "timeout_seconds", 15)
    # 进入异步上下文管理区域，确保异步资源按约定释放。
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = await client.get(f"{base_url}/api/v1/query", params={"query": query})
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = response.json()
    # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
    result = data.get("data", {}).get("result", []) if isinstance(data, dict) else []
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "provider": "mcp_prometheus",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "query": query,
        # 调用 data.get 完成当前步骤需要的业务处理。
        "result_type": data.get("data", {}).get("resultType") if isinstance(data, dict) else None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "series": result,
        # 调用 _summarize_prometheus_result 完成当前步骤需要的业务处理。
        "summary": _summarize_prometheus_result(result),
    }


# 定义 _build_default_promql 相关的处理逻辑，供流程或外部调用复用。
def _build_default_promql(alert_summary: str) -> str:
    # 根据告警摘要中的关键词选择一个基础 PromQL，给未传查询语句的调用兜底。
    lowered = alert_summary.lower()
    # 根据 any(token in lowered for token in ("5xx", "500", "er... 判断当前流程该进入哪个处理分支。
    if any(token in lowered for token in ("5xx", "500", "error", "错误", "异常")):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 'sum(rate(http_requests_total{status=~"5.."}[5m]))'
    # 根据 any(token in lowered for token in ("latency", "p99",... 判断当前流程该进入哪个处理分支。
    if any(token in lowered for token in ("latency", "p99", "p95", "延迟", "耗时", "超时")):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))'
    # 根据 any(token in lowered for token in ("cpu", "CPU")) 判断当前流程该进入哪个处理分支。
    if any(token in lowered for token in ("cpu", "CPU")):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
    # 根据 any(token in lowered for token in ("memory", "mem", ... 判断当前流程该进入哪个处理分支。
    if any(token in lowered for token in ("memory", "mem", "内存")):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "up"


# 定义 _summarize_prometheus_result 相关的处理逻辑，供流程或外部调用复用。
def _summarize_prometheus_result(result: Any) -> str:
    # 把 Prometheus 原始返回压缩成一句摘要，方便模型快速判断是否查到时间序列。
    if not isinstance(result, list):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "Prometheus returned a non-list result."
    # 根据 not result 判断当前流程该进入哪个处理分支。
    if not result:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "Prometheus returned no matching series."
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return f"Prometheus returned {len(result)} series."


# 定义 _tool_definitions 相关的处理逻辑，供流程或外部调用复用。
def _tool_definitions() -> list[dict[str, Any]]:
    # 声明 MCP 对外暴露的工具名称、用途和入参结构，客户端会据此展示可调用能力。
    return [
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "name": "aliyun_sls_query_logs",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "description": "Query Aliyun SLS logs for an alert summary.",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "inputSchema": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "type": "object",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "properties": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "alert_summary": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "group_id": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "max_lines": {"type": "integer"},
                },
            },
        },
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "name": "prometheus_query_metrics",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "description": "Query Prometheus metrics for an alert summary or explicit PromQL.",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "inputSchema": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "type": "object",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "properties": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "alert_summary": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "group_id": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "window": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "promql": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "prometheus_base_url": {"type": "string"},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "timeout_seconds": {"type": "integer"},
                },
            },
        },
    ]


# 定义 _read_message 相关的处理逻辑，供流程或外部调用复用。
async def _read_message() -> dict[str, Any] | None:
    # 按 MCP 的 Content-Length 协议从标准输入读取一条完整 JSON 请求。
    headers: dict[str, str] = {}
    # 在条件仍然成立时持续执行循环逻辑。
    while True:
        # 将 line 的值保存下来，供后续流程判断或组装响应时使用。
        line = await asyncio.to_thread(sys.stdin.buffer.readline)
        # 根据 not line 判断当前流程该进入哪个处理分支。
        if not line:
            return None
        # 将 stripped 的值保存下来，供后续流程判断或组装响应时使用。
        stripped = line.decode("ascii", errors="ignore").strip()
        # 根据 not stripped 判断当前流程该进入哪个处理分支。
        if not stripped:
            # 满足停止条件后退出循环，避免继续执行无效处理。
            break
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        key, _, value = stripped.partition(":")
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        headers[key.lower()] = value.strip()
    # 将 length 的值保存下来，供后续流程判断或组装响应时使用。
    length = int(headers.get("content-length", "0"))
    # 根据 length <= 0 判断当前流程该进入哪个处理分支。
    if length <= 0:
        return None
    # 将 body 的值保存下来，供后续流程判断或组装响应时使用。
    body = await asyncio.to_thread(sys.stdin.buffer.read, length)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return json.loads(body.decode("utf-8"))


# 定义 _write_message 相关的处理逻辑，供流程或外部调用复用。
async def _write_message(payload: dict[str, Any]) -> None:
    # 把 JSON-RPC 响应按 MCP 的 Content-Length 协议写回标准输出。
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    # 将 header 的值保存下来，供后续流程判断或组装响应时使用。
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    # 调用 buffer.write 完成当前步骤需要的业务处理。
    sys.stdout.buffer.write(header + body)
    # 等待异步操作完成，再继续推进当前业务流程。
    await asyncio.to_thread(sys.stdout.buffer.flush)


# 定义 _error_response 相关的处理逻辑，供流程或外部调用复用。
def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    # 生成标准 JSON-RPC 错误响应，让客户端能明确知道失败原因。
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# 定义 _str_or_none 相关的处理逻辑，供流程或外部调用复用。
def _str_or_none(value: Any) -> str | None:
    # 把可选参数统一转换成非空字符串；空值统一返回 None 方便后续兜底。
    if value is None:
        return None
    # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
    text = str(value).strip()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return text or None


# 定义 _int_arg 相关的处理逻辑，供流程或外部调用复用。
def _int_arg(arguments: dict[str, Any], key: str, default: int) -> int:
    # 从工具参数中读取整数配置，未传或为空时使用默认值。
    value = arguments.get(key)
    # 根据 value is None or value == "" 判断当前流程该进入哪个处理分支。
    if value is None or value == "":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return int(value)


# 根据 __name__ == "__main__" 判断当前流程该进入哪个处理分支。
if __name__ == "__main__":
    # 调用 asyncio.run 完成当前步骤需要的业务处理。
    asyncio.run(main())
