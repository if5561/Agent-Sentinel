from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import quote

import httpx
import yaml
from cachetools import TTLCache

from agent_sentinel.config import LangfuseConfig
from agent_sentinel.utils.config_loader import PROJECT_ROOT

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 将 PromptFallback 的值保存下来，供后续流程判断或组装响应时使用。
PromptFallback = Callable[[str, str], "LangfusePrompt | str | dict[str, Any] | None"]

# 将 _TRACE_CONTEXT 的值保存下来，供后续流程判断或组装响应时使用。
_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("langfuse_trace_context", default={})
# 将 _TEMPLATE_PATTERN 的值保存下来，供后续流程判断或组装响应时使用。
_TEMPLATE_PATTERN = re.compile(r"(?<!\\)\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")


@dataclass(slots=True)
# 定义 LangfusePrompt 组件，集中管理这个模块的状态和行为。
class LangfusePrompt:
    # 执行当前业务步骤，推动流程继续向下游推进。
    name: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    prompt_type: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    content: str | list[dict[str, str]]
    # 将 version 的值保存下来，供后续流程判断或组装响应时使用。
    version: int | str | None = None
    # 将 prompt_id 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_id: str | None = None
    # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
    label: str | None = None


# 定义 SupportsAinvoke 组件，集中管理这个模块的状态和行为。
class SupportsAinvoke(Protocol):
    # 定义 ainvoke 相关的处理逻辑，供流程或外部调用复用。
    async def ainvoke(self, input: object, config: dict[str, Any] | None = None, **kwargs: Any) -> object:
        # 声明模型对象需要支持异步调用，方便 Langfuse 服务统一包装不同模型客户端。
        ...


# 定义 set_trace_context 相关的处理逻辑，供流程或外部调用复用。
def set_trace_context(**values: Any) -> Token[dict[str, Any]]:
    # 把当前诊断的 trace、任务和节点信息放进上下文，后续模型调用会自动带上这些信息。
    current = dict(_TRACE_CONTEXT.get())
    # contextvars 让同一次工作流中的多次 LLM 调用共享 trace 元数据，同时不污染并发请求。
    current.update({key: value for key, value in values.items() if value is not None})
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return _TRACE_CONTEXT.set(current)


# 定义 reset_trace_context 相关的处理逻辑，供流程或外部调用复用。
def reset_trace_context(token: Token[dict[str, Any]]) -> None:
    # 模型调用结束后恢复旧上下文，避免并发请求之间互相串 trace。
    _TRACE_CONTEXT.reset(token)


# 定义 current_trace_context 相关的处理逻辑，供流程或外部调用复用。
def current_trace_context() -> dict[str, Any]:
    # 读取当前异步任务里的追踪上下文，供工具路由、证据审查等节点复用。
    return dict(_TRACE_CONTEXT.get())


# 定义 current_trace_id 相关的处理逻辑，供流程或外部调用复用。
def current_trace_id() -> str | None:
    # 取出当前诊断的业务追踪编号；没有追踪信息时返回空值。
    trace_id = str(_TRACE_CONTEXT.get().get("trace_id") or "").strip()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return trace_id or None


# 定义 LangfuseHttpClient 组件，集中管理这个模块的状态和行为。
class LangfuseHttpClient:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, config: LangfuseConfig) -> None:
        # 保存 Langfuse 地址和认证信息，并创建复用的异步 HTTP 客户端。
        self.config = config
        # 将 self.host 的值保存下来，供后续流程判断或组装响应时使用。
        self.host = str(config.host or "").rstrip("/")
        # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
        self._client = httpx.AsyncClient(
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=httpx.Timeout(10.0),
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={"Authorization": self._auth_header()},
        )

    # 定义 close 相关的处理逻辑，供流程或外部调用复用。
    async def close(self) -> None:
        # 关闭 Langfuse HTTP 连接池，服务退出时用来释放网络资源。
        await self._client.aclose()

    # 定义 get_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def get_prompt(self, name: str, label: str) -> dict[str, Any] | None:
        # 从 Langfuse 远程读取指定名称和标签的提示词，找不到时返回空值走本地兜底。
        path = f"/api/public/v2/prompts/{quote(name)}"
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = await self._client.get(
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"{self.host}{path}",
            # 将 params 的值保存下来，供后续流程判断或组装响应时使用。
            params={"label": label},
        )
        # 根据 response.status_code == 404 判断当前流程该进入哪个处理分支。
        if response.status_code == 404:
            return None
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = response.json()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return data if isinstance(data, dict) else None

    # 定义 create_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def create_prompt(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt: str | list[dict[str, str]],
        # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_type: str = "text",
        # 将 labels 的值保存下来，供后续流程判断或组装响应时使用。
        labels: list[str] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, Any]:
        # 把本地准备好的提示词创建到 Langfuse，供后续统一版本管理。
        response = await self._client.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"{self.host}/api/public/v2/prompts",
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "name": name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "type": prompt_type,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "prompt": prompt,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "labels": labels or [self.config.label],
            },
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = response.json()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return data if isinstance(data, dict) else {}

    # 定义 list_generations 相关的处理逻辑，供流程或外部调用复用。
    async def list_generations(self, trace_id: str) -> list[dict[str, Any]]:
        # 查询某个 trace 下已经写入 Langfuse 的模型 generation，用于后续补充 prompt 关联。
        response = await self._client.get(
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"{self.host}/api/public/observations",
            # 将 params 的值保存下来，供后续流程判断或组装响应时使用。
            params={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "traceId": trace_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "type": "GENERATION",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "limit": "50",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "fields": "core,prompt",
            },
        )
        # 根据 response.status_code == 404 判断当前流程该进入哪个处理分支。
        if response.status_code == 404:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = response.json()
        # 根据 isinstance(data, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(data, dict):
            # 将 items 的值保存下来，供后续流程判断或组装响应时使用。
            items = data.get("data") or data.get("observations") or []
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return [item for item in items if isinstance(item, dict)]
        # 根据 isinstance(data, list) 判断当前流程该进入哪个处理分支。
        if isinstance(data, list):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return [item for item in data if isinstance(item, dict)]
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []

    # 定义 ingest 相关的处理逻辑，供流程或外部调用复用。
    async def ingest(self, events: list[dict[str, Any]]) -> None:
        # 向 Langfuse 批量写入事件，例如把已有 generation 补绑定到某个 prompt。
        if not events:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = await self._client.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"{self.host}/api/public/ingestion",
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "batch": events,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "metadata": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "sdk": "agent-sentinel-httpx",
                    # 调用 len 完成当前步骤需要的业务处理。
                    "batch_size": len(events),
                },
            },
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()

    # 定义 _auth_header 相关的处理逻辑，供流程或外部调用复用。
    def _auth_header(self) -> str:
        # 把 Langfuse 公钥和密钥编码成 HTTP Basic 认证头。
        raw = f"{self.config.public_key or ''}:{self.config.secret_key or ''}".encode("utf-8")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"Basic {base64.b64encode(raw).decode('ascii')}"


# 定义 NullLangfusePromptService 组件，集中管理这个模块的状态和行为。
class NullLangfusePromptService:
    # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
    enabled = False

    # 定义 call_llm_with_trace 相关的处理逻辑，供流程或外部调用复用。
    async def call_llm_with_trace(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        llm: SupportsAinvoke,
        # 执行当前业务步骤，推动流程继续向下游推进。
        fallback_prompt: str,
        # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_name: str | None = None,
        # 将 variables 的值保存下来，供后续流程判断或组装响应时使用。
        variables: dict[str, Any] | None = None,
        # 将 prompt_label 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_label: str | None = None,
        # 将 model_name 的值保存下来，供后续流程判断或组装响应时使用。
        model_name: str | None = None,
        # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
        config: dict[str, Any] | None = None,
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata: dict[str, Any] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> object:
        # Langfuse 未启用时直接调用模型，不做远程提示词和观测增强。
        return await llm.ainvoke(fallback_prompt, config=config)

    # 定义 callbacks 相关的处理逻辑，供流程或外部调用复用。
    def callbacks(self) -> list[Any]:
        # 空服务不提供 LangChain 回调，保持调用方接口一致。
        return []

    # 定义 flush 相关的处理逻辑，供流程或外部调用复用。
    async def flush(self) -> None:
        # 空服务没有待写入的观测任务，直接返回。
        return None


# 定义 LangfusePromptService 组件，集中管理这个模块的状态和行为。
class LangfusePromptService:
    # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
    enabled = True

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        config: LangfuseConfig,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 fallback_provider 的值保存下来，供后续流程判断或组装响应时使用。
        fallback_provider: PromptFallback | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 准备 Langfuse 提示词服务，包含远程客户端、兜底加载器、缓存和后台补链任务集合。
        self.config = config
        # 将 self.client 的值保存下来，供后续流程判断或组装响应时使用。
        self.client = LangfuseHttpClient(config)
        # 将 self.fallback_provider 的值保存下来，供后续流程判断或组装响应时使用。
        self.fallback_provider = fallback_provider
        # 将 self.cache 的值保存下来，供后续流程判断或组装响应时使用。
        self.cache: TTLCache[tuple[str, str], LangfusePrompt] = TTLCache(
            # 将 maxsize 的值保存下来，供后续流程判断或组装响应时使用。
            maxsize=256,
            # 将 ttl 的值保存下来，供后续流程判断或组装响应时使用。
            ttl=max(config.cache_ttl_seconds, 1),
        )
        # 将 self._cache_lock 的值保存下来，供后续流程判断或组装响应时使用。
        self._cache_lock = asyncio.Lock()
        # 将 self._bound_generation_ids 的值保存下来，供后续流程判断或组装响应时使用。
        self._bound_generation_ids: set[str] = set()
        # 将 self._pending_tasks 的值保存下来，供后续流程判断或组装响应时使用。
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    # 定义 get_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def get_prompt(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
        label: str | None = None,
        # 将 fallback_prompt 的值保存下来，供后续流程判断或组装响应时使用。
        fallback_prompt: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> LangfusePrompt:
        # 按优先级获取提示词：先查缓存和远程 Langfuse，再回退到自定义或本地模板。
        prompt_label = label or self.config.label
        # 将 cache_key 的值保存下来，供后续流程判断或组装响应时使用。
        cache_key = (name, prompt_label)
        # 进入异步上下文管理区域，确保异步资源按约定释放。
        async with self._cache_lock:
            # 将 cached 的值保存下来，供后续流程判断或组装响应时使用。
            cached = self.cache.get(cache_key)
        # 根据 cached is not None 判断当前流程该进入哪个处理分支。
        if cached is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return cached

        # Prompt 解析顺序：远程 Langfuse -> 自定义 fallback -> 本地文件 -> 调用方传入的 inline prompt。
        prompt = await self._load_remote_prompt(name, prompt_label)
        # 根据 prompt is None 判断当前流程该进入哪个处理分支。
        if prompt is None:
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt = await self._load_custom_fallback(name, prompt_label)
        # 根据 prompt is None 判断当前流程该进入哪个处理分支。
        if prompt is None:
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt = self._load_local_prompt(name, prompt_label)
        # 根据 prompt is None and fallback_prompt is not None 判断当前流程该进入哪个处理分支。
        if prompt is None and fallback_prompt is not None:
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt = LangfusePrompt(name=name, prompt_type="text", content=fallback_prompt, label=prompt_label)
        # 根据 prompt is None 判断当前流程该进入哪个处理分支。
        if prompt is None:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Prompt not found in Langfuse or local fallbacks: {name}")

        # 进入异步上下文管理区域，确保异步资源按约定释放。
        async with self._cache_lock:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            self.cache[cache_key] = prompt
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return prompt

    # 定义 create_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def create_prompt(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt: str | list[dict[str, str]],
        # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_type: str = "text",
        # 将 labels 的值保存下来，供后续流程判断或组装响应时使用。
        labels: list[str] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> LangfusePrompt:
        # 通过 Langfuse API 创建提示词，并把接口返回内容转成项目内部统一结构。
        try:
            # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
            data = await self.client.create_prompt(
                # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
                name=name,
                # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
                prompt=prompt,
                # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
                prompt_type=prompt_type,
                # 将 labels 的值保存下来，供后续流程判断或组装响应时使用。
                labels=labels,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return _parse_prompt_response(data, default_name=name, default_label=self.config.label)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Langfuse prompt creation failed name=%s error=%s", name, exc)
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise

    # 定义 call_llm_with_trace 相关的处理逻辑，供流程或外部调用复用。
    async def call_llm_with_trace(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        llm: SupportsAinvoke,
        # 执行当前业务步骤，推动流程继续向下游推进。
        fallback_prompt: str,
        # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_name: str | None = None,
        # 将 variables 的值保存下来，供后续流程判断或组装响应时使用。
        variables: dict[str, Any] | None = None,
        # 将 prompt_label 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_label: str | None = None,
        # 将 model_name 的值保存下来，供后续流程判断或组装响应时使用。
        model_name: str | None = None,
        # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
        config: dict[str, Any] | None = None,
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata: dict[str, Any] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> object:
        # 渲染提示词、合并追踪元数据，再调用模型并把本次调用接入 Langfuse 观测链路。
        variables = variables or {}
        # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
        prompt = await self._resolve_prompt(prompt_name, prompt_label, fallback_prompt)
        # 将 rendered_prompt 的值保存下来，供后续流程判断或组装响应时使用。
        rendered_prompt = render_prompt(prompt, variables)
        # 将 trace_context 的值保存下来，供后续流程判断或组装响应时使用。
        trace_context = current_trace_context()
        # metadata 会同时传给 LangChain config 和 LiteLLM extra_body，尽量兼容不同观测链路。
        langfuse_metadata = _build_langfuse_metadata(
            # 将 trace_context 的值保存下来，供后续流程判断或组装响应时使用。
            trace_context=trace_context,
            # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
            metadata=metadata,
            # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
            prompt=prompt,
        )
        # 将 trace_id 的值保存下来，供后续流程判断或组装响应时使用。
        trace_id = str(langfuse_metadata["trace_id"])
        # 将 generation_name 的值保存下来，供后续流程判断或组装响应时使用。
        generation_name = str(langfuse_metadata["generation_name"])
        # 将 runnable_config 的值保存下来，供后续流程判断或组装响应时使用。
        runnable_config = _merge_runnable_config(
            # 执行当前业务步骤，推动流程继续向下游推进。
            config,
            # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
            metadata=langfuse_metadata,
        )
        # 将 extra_body 的值保存下来，供后续流程判断或组装响应时使用。
        extra_body = {"metadata": langfuse_metadata}
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
            response = await llm.ainvoke(
                # 执行当前业务步骤，推动流程继续向下游推进。
                rendered_prompt,
                # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
                config=runnable_config,
                # 将 extra_body 的值保存下来，供后续流程判断或组装响应时使用。
                extra_body=extra_body,
            )
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except TypeError:
            # 某些 LangChain model 不接受 extra_body，退回只传 config 的调用方式。
            response = await llm.ainvoke(rendered_prompt, config=runnable_config)
        # 根据 self.config.link_generations and prompt.prompt_id 判断当前流程该进入哪个处理分支。
        if self.config.link_generations and prompt.prompt_id:
            # 绑定 generation 到 prompt 需要等待 Langfuse/LiteLLM 先写入 observation，因此异步轮询补链。
            self._schedule(
                # 调用 self.link_generation_to_prompt 完成当前步骤需要的业务处理。
                self.link_generation_to_prompt(
                    # 将 trace_id 的值保存下来，供后续流程判断或组装响应时使用。
                    trace_id=trace_id,
                    # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt=prompt,
                    # 将 generation_name 的值保存下来，供后续流程判断或组装响应时使用。
                    generation_name=generation_name,
                )
            )
        # 根据 prompt.prompt_id and not self.config.link_generation... 判断当前流程该进入哪个处理分支。
        elif prompt.prompt_id and not self.config.link_generations:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.debug(
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Langfuse generation prompt linking disabled; prompt metadata is sent with LiteLLM metadata "
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "prompt_name=%s prompt_version=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                prompt.name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                prompt.version,
            )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return response

    # 定义 link_generation_to_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def link_generation_to_prompt(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        trace_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt: LangfusePrompt,
        # 执行当前业务步骤，推动流程继续向下游推进。
        generation_name: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 异步轮询 Langfuse，找到刚写入的 generation 后补上它对应的 prompt 信息。
        await asyncio.sleep(max(self.config.link_initial_delay_ms, 0) / 1000)
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for _ in range(max(self.config.link_max_retries, 1)):
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 generations 的值保存下来，供后续流程判断或组装响应时使用。
                generations = await self.client.list_generations(trace_id)
                # 只绑定尚未关联 prompt 的最新 generation，避免重复写入同一个 observation。
                unbound = [
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    generation
                    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
                    for generation in generations
                    # 根据 _generation_id(generation) 判断当前流程该进入哪个处理分支。
                    if _generation_id(generation)
                    # 调用 _generation_id 完成当前步骤需要的业务处理。
                    and _generation_id(generation) not in self._bound_generation_ids
                    # 调用 _generation_matches 完成当前步骤需要的业务处理。
                    and _generation_matches(generation, generation_name)
                    # 调用 generation.get 完成当前步骤需要的业务处理。
                    and not generation.get("promptId")
                    # 调用 generation.get 完成当前步骤需要的业务处理。
                    and not generation.get("prompt")
                ]
                # 根据 unbound 判断当前流程该进入哪个处理分支。
                if unbound:
                    # 等待异步操作完成，再继续推进当前业务流程。
                    await self._bind_generation(_generation_id(unbound[-1]), trace_id, prompt)
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as exc:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Langfuse generation prompt link poll failed trace_id=%s error=%s", trace_id, exc)
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.sleep(max(self.config.link_poll_interval_ms, 1) / 1000)

    # 定义 callbacks 相关的处理逻辑，供流程或外部调用复用。
    def callbacks(self) -> list[Any]:
        # 当前实现通过 LiteLLM metadata 上报，不额外注册 LangChain callbacks。
        return []

    # 定义 flush 相关的处理逻辑，供流程或外部调用复用。
    async def flush(self) -> None:
        # 等待后台补链任务完成，并关闭 Langfuse HTTP 客户端。
        tasks = list(self._pending_tasks)
        # 根据 tasks 判断当前流程该进入哪个处理分支。
        if tasks:
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.gather(*tasks, return_exceptions=True)
        # 等待异步操作完成，再继续推进当前业务流程。
        await self.client.close()

    # 定义 _resolve_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def _resolve_prompt(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt_name: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt_label: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        fallback_prompt: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> LangfusePrompt:
        # 解析一次模型调用应该使用的提示词；指定名称失败时回退到调用方传入的文本。
        if prompt_name:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return await self.get_prompt(prompt_name, label=prompt_label, fallback_prompt=fallback_prompt)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as exc:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Langfuse prompt resolution failed name=%s error=%s", prompt_name, exc)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return LangfusePrompt(
            # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
            name=prompt_name or "inline",
            # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_type="text",
            # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
            content=fallback_prompt,
            # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
            label=prompt_label or self.config.label,
        )

    # 定义 _load_remote_prompt 相关的处理逻辑，供流程或外部调用复用。
    async def _load_remote_prompt(self, name: str, label: str) -> LangfusePrompt | None:
        # 尝试从远程 Langfuse 加载提示词，失败时返回空值让主流程继续走本地兜底。
        if not self.config.remote_prompts_enabled:
            return None
        # 根据 not self.config.host 判断当前流程该进入哪个处理分支。
        if not self.config.host:
            return None
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 远程 prompt 获取失败不能影响诊断主流程，失败后会继续走本地 fallback。
            data = await self.client.get_prompt(name, label)
            # 根据 data is None 判断当前流程该进入哪个处理分支。
            if data is None:
                return None
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return _parse_prompt_response(data, default_name=name, default_label=label)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Langfuse prompt fetch failed name=%s label=%s error=%s", name, label, exc)
            return None

    # 定义 _load_custom_fallback 相关的处理逻辑，供流程或外部调用复用。
    async def _load_custom_fallback(self, name: str, label: str) -> LangfusePrompt | None:
        # 调用项目自定义的提示词兜底函数，适配测试或特殊部署环境。
        if self.fallback_provider is None:
            return None
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
            raw = self.fallback_provider(name, label)
            # 根据 asyncio.iscoroutine(raw) 判断当前流程该进入哪个处理分支。
            if asyncio.iscoroutine(raw):
                # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
                raw = await raw
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return _coerce_prompt(raw, name=name, label=label)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Langfuse custom prompt fallback failed name=%s error=%s", name, exc)
            return None

    # 定义 _load_local_prompt 相关的处理逻辑，供流程或外部调用复用。
    def _load_local_prompt(self, name: str, label: str) -> LangfusePrompt | None:
        # 在本地 prompts 目录里查找提示词文件，支持 txt、json、yaml 等格式。
        search_dirs = [
            # 调用 _resolve_path 完成当前步骤需要的业务处理。
            _resolve_path(self.config.prompt_dir),
            # 执行当前业务步骤，推动流程继续向下游推进。
            PROJECT_ROOT / "prompts",
            # 执行当前业务步骤，推动流程继续向下游推进。
            PROJECT_ROOT / "config" / "prompts",
        ]
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for directory in search_dirs:
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for suffix in (".txt", ".json", ".yaml", ".yml"):
                # 将 path 的值保存下来，供后续流程判断或组装响应时使用。
                path = directory / f"{name}{suffix}"
                # 根据 not path.exists() 判断当前流程该进入哪个处理分支。
                if not path.exists():
                    # 跳过当前项剩余逻辑，继续处理下一项数据。
                    continue
                # 进入可能失败的处理块，便于后续统一捕获和恢复。
                try:
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return _load_prompt_file(path, name=name, label=label)
                # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
                except Exception as exc:
                    # 记录关键运行信息，方便排查流程进展和异常现场。
                    logger.warning("Langfuse local prompt load failed path=%s error=%s", path, exc)
        return None

    # 定义 _bind_generation 相关的处理逻辑，供流程或外部调用复用。
    async def _bind_generation(self, generation_id: str | None, trace_id: str, prompt: LangfusePrompt) -> None:
        # 把已经产生的模型 generation 和提示词版本关联起来，便于在 Langfuse 页面追溯。
        if not generation_id:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 等待异步操作完成，再继续推进当前业务流程。
            await self.client.ingest(
                [
                    # 调用 _event 完成当前步骤需要的业务处理。
                    _event(
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "generation-create",
                        {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "id": generation_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "traceId": trace_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "promptId": prompt.prompt_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "promptName": prompt.name,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "promptVersion": prompt.version,
                        },
                    )
                ]
            )
            # 调用 _bound_generation_ids.add 完成当前步骤需要的业务处理。
            self._bound_generation_ids.add(generation_id)
            # 根据 len(self._bound_generation_ids) > 10000 判断当前流程该进入哪个处理分支。
            if len(self._bound_generation_ids) > 10000:
                # 调用 _bound_generation_ids.clear 完成当前步骤需要的业务处理。
                self._bound_generation_ids.clear()
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Langfuse generation prompt bind failed generation_id=%s error=%s", generation_id, exc)

    # 定义 _schedule 相关的处理逻辑，供流程或外部调用复用。
    def _schedule(self, coroutine: Any) -> None:
        # 把补链任务放到后台执行，避免阻塞当前模型调用返回。
        try:
            # 将 task 的值保存下来，供后续流程判断或组装响应时使用。
            task = asyncio.create_task(coroutine)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except RuntimeError:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 调用 _pending_tasks.add 完成当前步骤需要的业务处理。
        self._pending_tasks.add(task)
        # 调用 task.add_done_callback 完成当前步骤需要的业务处理。
        task.add_done_callback(self._pending_tasks.discard)


# 定义 build_langfuse_prompt_service 相关的处理逻辑，供流程或外部调用复用。
def build_langfuse_prompt_service(config: LangfuseConfig) -> LangfusePromptService | NullLangfusePromptService:
    # 根据配置决定使用真实 Langfuse 服务还是空实现，让业务代码不用到处判断开关。
    if not config.enabled:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return NullLangfusePromptService()
    # 根据 not config.host or not config.public_key or not conf... 判断当前流程该进入哪个处理分支。
    if not config.host or not config.public_key or not config.secret_key:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.warning("Langfuse is enabled but host/public_key/secret_key is incomplete; using null service.")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return NullLangfusePromptService()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return LangfusePromptService(config)


# 定义 render_prompt 相关的处理逻辑，供流程或外部调用复用。
def render_prompt(prompt: LangfusePrompt, variables: dict[str, Any]) -> str | list[dict[str, str]]:
    # 把提示词模板里的变量替换成真实值，chat 类型会保留 role/content 消息结构。
    if prompt.prompt_type == "chat" and isinstance(prompt.content, list):
        # chat prompt 保留 role/content 结构，便于传给支持消息数组的模型客户端。
        return [
            {
                # 调用 str 完成当前步骤需要的业务处理。
                "role": str(message.get("role", "")),
                # 调用 _render_template 完成当前步骤需要的业务处理。
                "content": _render_template(str(message.get("content", "")), variables),
            }
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for message in prompt.content
        ]
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return _render_template(str(prompt.content), variables)


# 定义 _render_template 相关的处理逻辑，供流程或外部调用复用。
def _render_template(template: str, variables: dict[str, Any]) -> str:
    # 渲染模板文本中的 {{变量}} 和兼容旧格式的 {变量} 占位符。
    escaped: list[str] = []

    # 定义 protect_escaped 相关的处理逻辑，供流程或外部调用复用。
    def protect_escaped(match: re.Match[str]) -> str:
        # 支持 \{{var}} 形式的转义变量，避免模板渲染误替换示例文本。
        # 先保护转义变量，等正常变量替换完再把原文放回去。
        escaped.append(match.group(0)[1:])
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"__LANGFUSE_ESCAPED_VAR_{len(escaped) - 1}__"

    # 将 protected 的值保存下来，供后续流程判断或组装响应时使用。
    protected = re.sub(r"\\\{\{\s*[A-Za-z_][A-Za-z0-9_.-]*\s*\}\}", protect_escaped, template)

    # 定义 replace 相关的处理逻辑，供流程或外部调用复用。
    def replace(match: re.Match[str]) -> str:
        # 把单个 {{变量}} 替换成 variables 里对应的值，找不到就替换成空字符串。
        key = match.group(1)
        # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
        value = _lookup_variable(variables, key)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return str(value if value is not None else "")

    # 将 rendered 的值保存下来，供后续流程判断或组装响应时使用。
    rendered = _TEMPLATE_PATTERN.sub(replace, protected)
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for key, value in variables.items():
        # 根据 re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)) 判断当前流程该进入哪个处理分支。
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
            # 兼容旧模板里的 {var} 占位符，同时保留 Langfuse 推荐的 {{var}} 写法。
            rendered = rendered.replace(f"{{{key}}}", str(value))
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, value in enumerate(escaped):
        # 将 rendered 的值保存下来，供后续流程判断或组装响应时使用。
        rendered = rendered.replace(f"__LANGFUSE_ESCAPED_VAR_{index}__", value)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return rendered


# 定义 _lookup_variable 相关的处理逻辑，供流程或外部调用复用。
def _lookup_variable(variables: dict[str, Any], key: str) -> Any:
    # 支持 a.b.c 这种嵌套变量读取，让提示词可以引用复杂上下文里的字段。
    value: Any = variables
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for part in key.split("."):
        # 根据 isinstance(value, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(value, dict):
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = value.get(part)
        # 处理前面条件都不满足时的默认分支。
        else:
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = getattr(value, part, None)
        # 根据 value is None 判断当前流程该进入哪个处理分支。
        if value is None:
            return None
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return value


# 定义 _parse_prompt_response 相关的处理逻辑，供流程或外部调用复用。
def _parse_prompt_response(data: dict[str, Any], *, default_name: str, default_label: str) -> LangfusePrompt:
    # 把 Langfuse API 返回的提示词 JSON 转成项目内部的 LangfusePrompt 对象。
    content = data.get("prompt")
    # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_type = str(data.get("type") or ("chat" if isinstance(content, list) else "text"))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return LangfusePrompt(
        # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
        name=str(data.get("name") or default_name),
        # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_type=prompt_type,
        # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
        content=content if isinstance(content, (str, list)) else str(content or ""),
        # 将 version 的值保存下来，供后续流程判断或组装响应时使用。
        version=data.get("version"),
        # 将 prompt_id 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_id=str(data.get("id") or data.get("promptId") or "") or None,
        # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
        label=default_label,
    )


# 定义 _coerce_prompt 相关的处理逻辑，供流程或外部调用复用。
def _coerce_prompt(raw: LangfusePrompt | str | dict[str, Any] | None, *, name: str, label: str) -> LangfusePrompt | None:
    # 把字符串、字典或已有对象统一转换成 LangfusePrompt，方便上层统一处理。
    if raw is None:
        return None
    # 根据 isinstance(raw, LangfusePrompt) 判断当前流程该进入哪个处理分支。
    if isinstance(raw, LangfusePrompt):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return raw
    # 根据 isinstance(raw, str) 判断当前流程该进入哪个处理分支。
    if isinstance(raw, str):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return LangfusePrompt(name=name, prompt_type="text", content=raw, label=label)
    # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
    content = raw.get("prompt") or raw.get("content") or ""
    # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_type = str(raw.get("type") or ("chat" if isinstance(content, list) else "text"))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return LangfusePrompt(
        # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
        name=str(raw.get("name") or name),
        # 将 prompt_type 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_type=prompt_type,
        # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
        content=content if isinstance(content, (str, list)) else str(content),
        # 将 version 的值保存下来，供后续流程判断或组装响应时使用。
        version=raw.get("version"),
        # 将 prompt_id 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_id=str(raw.get("id") or raw.get("promptId") or "") or None,
        # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
        label=label,
    )


# 定义 _load_prompt_file 相关的处理逻辑，供流程或外部调用复用。
def _load_prompt_file(path: Path, *, name: str, label: str) -> LangfusePrompt:
    # 读取本地提示词文件，并按文件类型解析成统一提示词对象。
    if path.suffix == ".txt":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return LangfusePrompt(name=name, prompt_type="text", content=path.read_text(encoding="utf-8"), label=label)
    # 根据 path.suffix == ".json" 判断当前流程该进入哪个处理分支。
    if path.suffix == ".json":
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = json.loads(path.read_text(encoding="utf-8"))
        # 根据 not isinstance(data, dict) 判断当前流程该进入哪个处理分支。
        if not isinstance(data, dict):
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise ValueError("Prompt JSON root must be an object.")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _coerce_prompt(data, name=name, label=label) or LangfusePrompt(name=name, prompt_type="text", content="", label=label)
    # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # 根据 not isinstance(data, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(data, dict):
        # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
        raise ValueError("Prompt YAML root must be an object.")
    # 根据 "prompt" in data or "content" in data 判断当前流程该进入哪个处理分支。
    if "prompt" in data or "content" in data:
        # 将 coerced 的值保存下来，供后续流程判断或组装响应时使用。
        coerced = _coerce_prompt(data, name=name, label=label)
        # 根据 coerced is not None 判断当前流程该进入哪个处理分支。
        if coerced is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return coerced
    # 将 system 的值保存下来，供后续流程判断或组装响应时使用。
    system = str(data.get("system", ""))
    # 将 human 的值保存下来，供后续流程判断或组装响应时使用。
    human = str(data.get("human", ""))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return LangfusePrompt(name=name, prompt_type="text", content=f"{system}\n\n{human}".strip(), label=label)


# 定义 _merge_runnable_config 相关的处理逻辑，供流程或外部调用复用。
def _merge_runnable_config(
    # 执行当前业务步骤，推动流程继续向下游推进。
    config: dict[str, Any] | None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    metadata: dict[str, Any],
# 执行当前业务步骤，推动流程继续向下游推进。
) -> dict[str, Any]:
    # 把 Langfuse 元数据合并进 LangChain runnable config，随模型调用一起传下去。
    merged = dict(config or {})
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    merged["metadata"] = {**dict(merged.get("metadata") or {}), **metadata}
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return merged


# 定义 _event 相关的处理逻辑，供流程或外部调用复用。
def _event(event_type: str, body: dict[str, Any]) -> dict[str, Any]:
    # 构造 Langfuse ingestion API 需要的事件结构，并自动去掉空字段。
    return {
        # 调用 uuid.uuid4 完成当前步骤需要的业务处理。
        "id": f"evt-{uuid.uuid4()}",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "type": event_type,
        # 调用 body.items 完成当前步骤需要的业务处理。
        "body": {key: value for key, value in body.items() if value is not None},
    }


# 定义 _generation_id 相关的处理逻辑，供流程或外部调用复用。
def _generation_id(generation: dict[str, Any]) -> str | None:
    # 从不同版本的 Langfuse observation 字段里提取 generation ID。
    value = generation.get("id") or generation.get("observationId")
    # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
    text = str(value or "").strip()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return text or None


# 定义 _generation_matches 相关的处理逻辑，供流程或外部调用复用。
def _generation_matches(generation: dict[str, Any], generation_name: str) -> bool:
    # 判断某个 generation 是否属于当前节点，避免把 prompt 绑定到错误调用上。
    name = str(generation.get("name") or generation.get("generationName") or "").strip()
    # 根据 name and name != generation_name 判断当前流程该进入哪个处理分支。
    if name and name != generation_name:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return False
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return True


# 定义 _tags_for 相关的处理逻辑，供流程或外部调用复用。
def _tags_for(
    # 执行当前业务步骤，推动流程继续向下游推进。
    trace_context: dict[str, Any],
    # 执行当前业务步骤，推动流程继续向下游推进。
    metadata: dict[str, Any] | None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    prompt: LangfusePrompt,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> list[str]:
    # 合并工作流、节点和提示词标签，去重后写入 Langfuse 便于筛选。
    raw_tags = [*(trace_context.get("tags") or []), *((metadata or {}).get("tags") or [])]
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags = [str(tag) for tag in raw_tags if str(tag or "").strip()]
    # 把一组结果合并到集合中，扩展后续可使用的数据范围。
    tags.extend(["agent-sentinel", str(trace_context.get("workflow_type") or "workflow"), f"prompt:{prompt.name}"])
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return list(dict.fromkeys(tags))


# 定义 _build_langfuse_metadata 相关的处理逻辑，供流程或外部调用复用。
def _build_langfuse_metadata(
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    trace_context: dict[str, Any],
    # 执行当前业务步骤，推动流程继续向下游推进。
    metadata: dict[str, Any] | None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    prompt: LangfusePrompt,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> dict[str, Any]:
    # 构造一次模型调用的完整观测元数据，串联业务 trace、节点名、prompt 版本和标签。
    metadata = metadata or {}
    # 将 business_trace_id 的值保存下来，供后续流程判断或组装响应时使用。
    business_trace_id = str(
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        metadata.get("business_trace_id")
        # 调用 trace_context.get 完成当前步骤需要的业务处理。
        or trace_context.get("business_trace_id")
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        or metadata.get("trace_id")
        # 调用 trace_context.get 完成当前步骤需要的业务处理。
        or trace_context.get("trace_id")
        # 调用 uuid.uuid4 完成当前步骤需要的业务处理。
        or uuid.uuid4()
    )
    # business_trace_id 用于业务日志关联；Langfuse/OTEL 自身的 trace id 不一定完全等同于它。
    workflow_type = str(metadata.get("workflow_type") or trace_context.get("workflow_type") or "workflow")
    # 将 node_name 的值保存下来，供后续流程判断或组装响应时使用。
    node_name = str(metadata.get("node_name") or prompt.name)
    # 将 generation_name 的值保存下来，供后续流程判断或组装响应时使用。
    generation_name = str(metadata.get("generation_name") or node_name or prompt.name)
    # 将 trace_name 的值保存下来，供后续流程判断或组装响应时使用。
    trace_name = str(metadata.get("trace_name") or trace_context.get("trace_name") or workflow_type)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        **trace_context,
        # 执行当前业务步骤，推动流程继续向下游推进。
        **metadata,
        # Compatibility field for LiteLLM integrations that still look for metadata.trace_id.
        # 执行当前业务步骤，推动流程继续向下游推进。
        "trace_id": business_trace_id,
        # Explicit business correlation id. We do not rely on OTEL/LiteLLM using this as the Langfuse trace id.
        # 执行当前业务步骤，推动流程继续向下游推进。
        "business_trace_id": business_trace_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "trace_name": trace_name,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "generation_name": generation_name,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "node_name": node_name,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "workflow_type": workflow_type,
        # 调用 _tags_for 完成当前步骤需要的业务处理。
        "tags": _tags_for(trace_context, metadata, prompt),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "prompt_name": prompt.name,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "prompt_version": prompt.version,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "prompt_id": prompt.prompt_id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "prompt_label": prompt.label,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "otel_trace_source": "litellm",
    }


# 定义 _resolve_path 相关的处理逻辑，供流程或外部调用复用。
def _resolve_path(path: str | Path) -> Path:
    # 把提示词目录路径解析成绝对路径，相对路径默认基于项目根目录。
    resolved = Path(path)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
