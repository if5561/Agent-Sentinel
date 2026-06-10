from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.tools.mock_tools import MockTopologyProvider

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 MCPToolConfig 组件，集中管理这个模块的状态和行为。
class MCPToolConfig:
    # 执行当前业务步骤，推动流程继续向下游推进。
    command: str
    # 将 args 的值保存下来，供后续流程判断或组装响应时使用。
    args: list[str] = field(default_factory=list)
    # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    timeout_seconds: int = 20
    # 将 sls_tool 的值保存下来，供后续流程判断或组装响应时使用。
    sls_tool: str = "aliyun_sls_query_logs"
    # 将 prometheus_tool 的值保存下来，供后续流程判断或组装响应时使用。
    prometheus_tool: str = "prometheus_query_metrics"
    # 将 prometheus_base_url 的值保存下来，供后续流程判断或组装响应时使用。
    prometheus_base_url: str | None = None


# 定义 MCPClientError 组件，集中管理这个模块的状态和行为。
class MCPClientError(RuntimeError):
    pass


# 定义 MCPStdioClient 组件，集中管理这个模块的状态和行为。
class MCPStdioClient:
    """Minimal MCP JSON-RPC client for stdio servers."""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, config: MCPToolConfig) -> None:
        # 保存 MCP 子进程启动配置，并准备请求编号、进程句柄和串行调用锁。
        self._config = config
        # 将 self._request_id 的值保存下来，供后续流程判断或组装响应时使用。
        self._request_id = 0
        # 将 self._process 的值保存下来，供后续流程判断或组装响应时使用。
        self._process: asyncio.subprocess.Process | None = None
        # 将 self._lock 的值保存下来，供后续流程判断或组装响应时使用。
        self._lock = asyncio.Lock()

    # 定义 call_tool 相关的处理逻辑，供流程或外部调用复用。
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        # 通过 MCP 协议调用指定工具，例如查询 Prometheus 指标或 SLS 日志。
        async with self._lock:
            # 等待异步操作完成，再继续推进当前业务流程。
            await self._ensure_started()
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return await self._request(
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tools/call",
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "name": name,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "arguments": arguments,
                },
            )

    # 定义 close 相关的处理逻辑，供流程或外部调用复用。
    async def close(self) -> None:
        # 关闭 MCP 子进程，服务退出或连接失效时释放底层进程资源。
        process = self._process
        # 将 self._process 的值保存下来，供后续流程判断或组装响应时使用。
        self._process = None
        # 根据 process is None or process.returncode is not None 判断当前流程该进入哪个处理分支。
        if process is None or process.returncode is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 调用 process.terminate 完成当前步骤需要的业务处理。
        process.terminate()
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.wait_for(process.wait(), timeout=2)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except asyncio.TimeoutError:
            # 调用 process.kill 完成当前步骤需要的业务处理。
            process.kill()
            # 等待异步操作完成，再继续推进当前业务流程。
            await process.wait()

    # 定义 _ensure_started 相关的处理逻辑，供流程或外部调用复用。
    async def _ensure_started(self) -> None:
        # 确保 MCP Server 子进程已启动，并完成 JSON-RPC 初始化握手。
        if self._process is not None and self._process.returncode is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 not self._config.command 判断当前流程该进入哪个处理分支。
        if not self._config.command:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise MCPClientError("MCP server command is empty")
        # 将 self._process 的值保存下来，供后续流程判断或组装响应时使用。
        self._process = await asyncio.create_subprocess_exec(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._config.command,
            # 执行当前业务步骤，推动流程继续向下游推进。
            *self._config.args,
            # 将 stdin 的值保存下来，供后续流程判断或组装响应时使用。
            stdin=asyncio.subprocess.PIPE,
            # 将 stdout 的值保存下来，供后续流程判断或组装响应时使用。
            stdout=asyncio.subprocess.PIPE,
            # 将 stderr 的值保存下来，供后续流程判断或组装响应时使用。
            stderr=asyncio.subprocess.PIPE,
        )
        # 等待异步操作完成，再继续推进当前业务流程。
        await self._request(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "initialize",
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "protocolVersion": "2024-11-05",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "capabilities": {},
                # 执行当前业务步骤，推动流程继续向下游推进。
                "clientInfo": {"name": "agent-sentinel", "version": "0.1.0"},
            },
        )
        # 等待异步操作完成，再继续推进当前业务流程。
        await self._notify("notifications/initialized", {})

    # 定义 _notify 相关的处理逻辑，供流程或外部调用复用。
    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        # 向 MCP Server 发送无需响应的通知，例如 initialized。
        process = self._active_process()
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        # 等待异步操作完成，再继续推进当前业务流程。
        await self._write_message(process, payload)

    # 定义 _request 相关的处理逻辑，供流程或外部调用复用。
    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        # 发送一条 JSON-RPC 请求，并等待匹配 request_id 的响应。
        process = self._active_process()
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        self._request_id += 1
        # 将 request_id 的值保存下来，供后续流程判断或组装响应时使用。
        request_id = self._request_id
        # 等待异步操作完成，再继续推进当前业务流程。
        await self._write_message(
            # 执行当前业务步骤，推动流程继续向下游推进。
            process,
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "jsonrpc": "2.0",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "id": request_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "method": method,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "params": params,
            },
        )
        # 在条件仍然成立时持续执行循环逻辑。
        while True:
            # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
            response = await asyncio.wait_for(self._read_message(process), timeout=self._config.timeout_seconds)
            # 根据 response.get("id") != request_id 判断当前流程该进入哪个处理分支。
            if response.get("id") != request_id:
                # 跳过当前项剩余逻辑，继续处理下一项数据。
                continue
            # 根据 "error" in response 判断当前流程该进入哪个处理分支。
            if "error" in response:
                # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
                raise MCPClientError(f"MCP {method} failed: {response['error']}")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return response.get("result")

    # 定义 _active_process 相关的处理逻辑，供流程或外部调用复用。
    def _active_process(self) -> asyncio.subprocess.Process:
        # 取得当前可用的 MCP 子进程；进程不存在或 stdio 不可用时直接报错。
        process = self._process
        # 根据 process is None or process.returncode is not None 判断当前流程该进入哪个处理分支。
        if process is None or process.returncode is not None:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise MCPClientError("MCP server is not running")
        # 根据 process.stdin is None or process.stdout is None 判断当前流程该进入哪个处理分支。
        if process.stdin is None or process.stdout is None:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise MCPClientError("MCP server stdio is unavailable")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return process

    # 定义 _write_message 相关的处理逻辑，供流程或外部调用复用。
    async def _write_message(self, process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        # 按 MCP stdio 协议写入带 Content-Length 头的 JSON 消息。
        if process.stdin is None:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise MCPClientError("MCP server stdin is unavailable")
        # 将 body 的值保存下来，供后续流程判断或组装响应时使用。
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # 将 header 的值保存下来，供后续流程判断或组装响应时使用。
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        # 调用 stdin.write 完成当前步骤需要的业务处理。
        process.stdin.write(header + body)
        # 等待异步操作完成，再继续推进当前业务流程。
        await process.stdin.drain()

    # 定义 _read_message 相关的处理逻辑，供流程或外部调用复用。
    async def _read_message(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        # 从 MCP stdout 读取一条完整 JSON 消息，先读头部再按长度读正文。
        if process.stdout is None:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise MCPClientError("MCP server stdout is unavailable")
        # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
        headers: dict[str, str] = {}
        # 在条件仍然成立时持续执行循环逻辑。
        while True:
            # 将 line 的值保存下来，供后续流程判断或组装响应时使用。
            line = await process.stdout.readline()
            # 根据 not line 判断当前流程该进入哪个处理分支。
            if not line:
                # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
                raise MCPClientError("MCP server closed stdout")
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
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise MCPClientError("MCP response missing Content-Length")
        # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
        raw = await process.stdout.readexactly(length)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return json.loads(raw.decode("utf-8"))


# 定义 MCPLogsProvider 组件，集中管理这个模块的状态和行为。
class MCPLogsProvider:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, client: MCPStdioClient, config: MCPToolConfig) -> None:
        # 绑定 MCP 客户端和日志工具名，后续日志查询会通过 MCP Server 转发。
        self._client = client
        # 将 self._config 的值保存下来，供后续流程判断或组装响应时使用。
        self._config = config

    # 定义 query_logs 相关的处理逻辑，供流程或外部调用复用。
    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 通过 MCP Server 调用日志工具，把告警摘要转换成 SLS 查询结果。
        logger.info("Querying MCP SLS logs summary_chars=%s group_id=%s", len(alert_summary), group_id)
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = await self._client.call_tool(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._config.sls_tool,
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "alert_summary": alert_summary,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "group_id": group_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "max_lines": 100,
            },
        )
        # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
        monitor.record_tool_call("query_logs", group_id)
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = _extract_tool_payload(result)
        # 根据 isinstance(payload, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(payload, dict):
            # 调用 payload.setdefault 完成当前步骤需要的业务处理。
            payload.setdefault("provider", "mcp_aliyun_sls")
            # 调用 payload.setdefault 完成当前步骤需要的业务处理。
            payload.setdefault("tool", self._config.sls_tool)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return payload
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "provider": "mcp_aliyun_sls",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tool": self._config.sls_tool,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "matches": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "total": 0,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "raw": payload,
        }


# 定义 MCPMetricsProvider 组件，集中管理这个模块的状态和行为。
class MCPMetricsProvider:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, client: MCPStdioClient, config: MCPToolConfig) -> None:
        # 绑定 MCP 客户端和 Prometheus 工具名，后续指标查询会通过 MCP Server 转发。
        self._client = client
        # 将 self._config 的值保存下来，供后续流程判断或组装响应时使用。
        self._config = config

    # 定义 get_metrics 相关的处理逻辑，供流程或外部调用复用。
    async def get_metrics(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 通过 MCP Server 调用 Prometheus 工具，获取和告警相关的指标数据。
        logger.info("Querying MCP Prometheus metrics summary_chars=%s group_id=%s", len(alert_summary), group_id)
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = await self._client.call_tool(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self._config.prometheus_tool,
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "alert_summary": alert_summary,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "group_id": group_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "window": "30m",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "prometheus_base_url": self._config.prometheus_base_url,
            },
        )
        # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
        monitor.record_tool_call("get_metrics", group_id)
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = _extract_tool_payload(result)
        # 根据 isinstance(payload, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(payload, dict):
            # 调用 payload.setdefault 完成当前步骤需要的业务处理。
            payload.setdefault("provider", "mcp_prometheus")
            # 调用 payload.setdefault 完成当前步骤需要的业务处理。
            payload.setdefault("tool", self._config.prometheus_tool)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return payload
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "provider": "mcp_prometheus",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tool": self._config.prometheus_tool,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "series": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "raw": payload,
        }


# 定义 MCPToolsProvider 组件，集中管理这个模块的状态和行为。
class MCPToolsProvider:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, config: MCPToolConfig, client: MCPStdioClient | None = None) -> None:
        # 组装 MCP 指标、MCP 日志和 mock 拓扑 provider，形成完整工具接口。
        self._config = config
        # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
        self._client = client or MCPStdioClient(config)
        # 将 self._metrics 的值保存下来，供后续流程判断或组装响应时使用。
        self._metrics = MCPMetricsProvider(self._client, config)
        # 将 self._logs 的值保存下来，供后续流程判断或组装响应时使用。
        self._logs = MCPLogsProvider(self._client, config)
        # 将 self._topology 的值保存下来，供后续流程判断或组装响应时使用。
        self._topology = MockTopologyProvider()

    @property
    # 定义 metrics 相关的处理逻辑，供流程或外部调用复用。
    def metrics(self) -> MCPMetricsProvider:
        # 返回 MCP 指标 provider，供上层统一调用 get_metrics。
        return self._metrics

    @property
    # 定义 logs 相关的处理逻辑，供流程或外部调用复用。
    def logs(self) -> MCPLogsProvider:
        # 返回 MCP 日志 provider，供上层统一调用 query_logs。
        return self._logs

    @property
    # 定义 topology 相关的处理逻辑，供流程或外部调用复用。
    def topology(self) -> MockTopologyProvider:
        # 当前 MCP 组合里拓扑仍用 mock provider 补位，保持接口完整。
        return self._topology

    # 定义 fetch_all 相关的处理逻辑，供流程或外部调用复用。
    async def fetch_all(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        # 并发通过 MCP 查询指标和日志，同时用 mock 拓扑补齐完整现场数据结构。
        metrics, logs, topology = await asyncio.gather(
            # 调用 _metrics.get_metrics 完成当前步骤需要的业务处理。
            self._metrics.get_metrics(alert_summary, group_id),
            # 调用 _logs.query_logs 完成当前步骤需要的业务处理。
            self._logs.query_logs(alert_summary, group_id),
            # 调用 _topology.get_topology 完成当前步骤需要的业务处理。
            self._topology.get_topology(alert_summary, group_id),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"metrics": metrics, "logs": logs, "topology": topology}


# 定义 _extract_tool_payload 相关的处理逻辑，供流程或外部调用复用。
def _extract_tool_payload(result: Any) -> Any:
    # 从 MCP 工具返回的 content 数组里提取真正的 JSON 或文本结果。
    if not isinstance(result, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return result
    # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
    content = result.get("content")
    # 根据 not isinstance(content, list) or not content 判断当前流程该进入哪个处理分支。
    if not isinstance(content, list) or not content:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return result
    # 将 first 的值保存下来，供后续流程判断或组装响应时使用。
    first = content[0]
    # 根据 not isinstance(first, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(first, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return result
    # 根据 first.get("type") == "json" 判断当前流程该进入哪个处理分支。
    if first.get("type") == "json":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return first.get("json")
    # 根据 first.get("type") == "text" 判断当前流程该进入哪个处理分支。
    if first.get("type") == "text":
        # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
        text = str(first.get("text", ""))
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return json.loads(text)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except json.JSONDecodeError:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return text
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return result
