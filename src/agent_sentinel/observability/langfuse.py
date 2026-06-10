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

logger = logging.getLogger(__name__)

PromptFallback = Callable[[str, str], "LangfusePrompt | str | dict[str, Any] | None"]

_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("langfuse_trace_context", default={})
_TEMPLATE_PATTERN = re.compile(r"(?<!\\)\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")


@dataclass(slots=True)
class LangfusePrompt:
    name: str
    prompt_type: str
    content: str | list[dict[str, str]]
    version: int | str | None = None
    prompt_id: str | None = None
    label: str | None = None


class SupportsAinvoke(Protocol):
    async def ainvoke(self, input: object, config: dict[str, Any] | None = None, **kwargs: Any) -> object:
        # 方法说明：声明模型对象需要支持异步调用，方便 Langfuse 服务统一包装不同模型客户端。
        ...


def set_trace_context(**values: Any) -> Token[dict[str, Any]]:
    # 方法说明：把当前诊断的 trace、任务和节点信息放进上下文，后续模型调用会自动带上这些信息。
    current = dict(_TRACE_CONTEXT.get())
    # contextvars 让同一次工作流中的多次 LLM 调用共享 trace 元数据，同时不污染并发请求。
    current.update({key: value for key, value in values.items() if value is not None})
    return _TRACE_CONTEXT.set(current)


def reset_trace_context(token: Token[dict[str, Any]]) -> None:
    # 方法说明：模型调用结束后恢复旧上下文，避免并发请求之间互相串 trace。
    _TRACE_CONTEXT.reset(token)


def current_trace_context() -> dict[str, Any]:
    # 方法说明：读取当前异步任务里的追踪上下文，供工具路由、证据审查等节点复用。
    return dict(_TRACE_CONTEXT.get())


def current_trace_id() -> str | None:
    # 方法说明：取出当前诊断的业务追踪编号；没有追踪信息时返回空值。
    trace_id = str(_TRACE_CONTEXT.get().get("trace_id") or "").strip()
    return trace_id or None


class LangfuseHttpClient:
    def __init__(self, config: LangfuseConfig) -> None:
        # 方法说明：保存 Langfuse 地址和认证信息，并创建复用的异步 HTTP 客户端。
        self.config = config
        self.host = str(config.host or "").rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            headers={"Authorization": self._auth_header()},
        )

    async def close(self) -> None:
        # 方法说明：关闭 Langfuse HTTP 连接池，服务退出时用来释放网络资源。
        await self._client.aclose()

    async def get_prompt(self, name: str, label: str) -> dict[str, Any] | None:
        # 方法说明：从 Langfuse 远程读取指定名称和标签的提示词，找不到时返回空值走本地兜底。
        path = f"/api/public/v2/prompts/{quote(name)}"
        response = await self._client.get(
            f"{self.host}{path}",
            params={"label": label},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None

    async def create_prompt(
        self,
        *,
        name: str,
        prompt: str | list[dict[str, str]],
        prompt_type: str = "text",
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        # 方法说明：把本地准备好的提示词创建到 Langfuse，供后续统一版本管理。
        response = await self._client.post(
            f"{self.host}/api/public/v2/prompts",
            json={
                "name": name,
                "type": prompt_type,
                "prompt": prompt,
                "labels": labels or [self.config.label],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def list_generations(self, trace_id: str) -> list[dict[str, Any]]:
        # 方法说明：查询某个 trace 下已经写入 Langfuse 的模型 generation，用于后续补充 prompt 关联。
        response = await self._client.get(
            f"{self.host}/api/public/observations",
            params={
                "traceId": trace_id,
                "type": "GENERATION",
                "limit": "50",
                "fields": "core,prompt",
            },
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            items = data.get("data") or data.get("observations") or []
            return [item for item in items if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    async def ingest(self, events: list[dict[str, Any]]) -> None:
        # 方法说明：向 Langfuse 批量写入事件，例如把已有 generation 补绑定到某个 prompt。
        if not events:
            return
        response = await self._client.post(
            f"{self.host}/api/public/ingestion",
            json={
                "batch": events,
                "metadata": {
                    "sdk": "agent-sentinel-httpx",
                    "batch_size": len(events),
                },
            },
        )
        response.raise_for_status()

    def _auth_header(self) -> str:
        # 方法说明：把 Langfuse 公钥和密钥编码成 HTTP Basic 认证头。
        raw = f"{self.config.public_key or ''}:{self.config.secret_key or ''}".encode("utf-8")
        return f"Basic {base64.b64encode(raw).decode('ascii')}"


class NullLangfusePromptService:
    enabled = False

    async def call_llm_with_trace(
        self,
        *,
        llm: SupportsAinvoke,
        fallback_prompt: str,
        prompt_name: str | None = None,
        variables: dict[str, Any] | None = None,
        prompt_label: str | None = None,
        model_name: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> object:
        # 方法说明：Langfuse 未启用时直接调用模型，不做远程提示词和观测增强。
        return await llm.ainvoke(fallback_prompt, config=config)

    def callbacks(self) -> list[Any]:
        # 方法说明：空服务不提供 LangChain 回调，保持调用方接口一致。
        return []

    async def flush(self) -> None:
        # 方法说明：空服务没有待写入的观测任务，直接返回。
        return None


class LangfusePromptService:
    enabled = True

    def __init__(
        self,
        config: LangfuseConfig,
        *,
        fallback_provider: PromptFallback | None = None,
    ) -> None:
        # 方法说明：准备 Langfuse 提示词服务，包含远程客户端、兜底加载器、缓存和后台补链任务集合。
        self.config = config
        self.client = LangfuseHttpClient(config)
        self.fallback_provider = fallback_provider
        self.cache: TTLCache[tuple[str, str], LangfusePrompt] = TTLCache(
            maxsize=256,
            ttl=max(config.cache_ttl_seconds, 1),
        )
        self._cache_lock = asyncio.Lock()
        self._bound_generation_ids: set[str] = set()
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    async def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        fallback_prompt: str | None = None,
    ) -> LangfusePrompt:
        # 方法说明：按优先级获取提示词：先查缓存和远程 Langfuse，再回退到自定义或本地模板。
        prompt_label = label or self.config.label
        cache_key = (name, prompt_label)
        async with self._cache_lock:
            cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Prompt 解析顺序：远程 Langfuse -> 自定义 fallback -> 本地文件 -> 调用方传入的 inline prompt。
        prompt = await self._load_remote_prompt(name, prompt_label)
        if prompt is None:
            prompt = await self._load_custom_fallback(name, prompt_label)
        if prompt is None:
            prompt = self._load_local_prompt(name, prompt_label)
        if prompt is None and fallback_prompt is not None:
            prompt = LangfusePrompt(name=name, prompt_type="text", content=fallback_prompt, label=prompt_label)
        if prompt is None:
            raise RuntimeError(f"Prompt not found in Langfuse or local fallbacks: {name}")

        async with self._cache_lock:
            self.cache[cache_key] = prompt
        return prompt

    async def create_prompt(
        self,
        *,
        name: str,
        prompt: str | list[dict[str, str]],
        prompt_type: str = "text",
        labels: list[str] | None = None,
    ) -> LangfusePrompt:
        # 方法说明：通过 Langfuse API 创建提示词，并把接口返回内容转成项目内部统一结构。
        try:
            data = await self.client.create_prompt(
                name=name,
                prompt=prompt,
                prompt_type=prompt_type,
                labels=labels,
            )
            return _parse_prompt_response(data, default_name=name, default_label=self.config.label)
        except Exception as exc:
            logger.warning("Langfuse prompt creation failed name=%s error=%s", name, exc)
            raise

    async def call_llm_with_trace(
        self,
        *,
        llm: SupportsAinvoke,
        fallback_prompt: str,
        prompt_name: str | None = None,
        variables: dict[str, Any] | None = None,
        prompt_label: str | None = None,
        model_name: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> object:
        # 方法说明：渲染提示词、合并追踪元数据，再调用模型并把本次调用接入 Langfuse 观测链路。
        variables = variables or {}
        prompt = await self._resolve_prompt(prompt_name, prompt_label, fallback_prompt)
        rendered_prompt = render_prompt(prompt, variables)
        trace_context = current_trace_context()
        # metadata 会同时传给 LangChain config 和 LiteLLM extra_body，尽量兼容不同观测链路。
        langfuse_metadata = _build_langfuse_metadata(
            trace_context=trace_context,
            metadata=metadata,
            prompt=prompt,
        )
        trace_id = str(langfuse_metadata["trace_id"])
        generation_name = str(langfuse_metadata["generation_name"])
        runnable_config = _merge_runnable_config(
            config,
            metadata=langfuse_metadata,
        )
        extra_body = {"metadata": langfuse_metadata}
        try:
            response = await llm.ainvoke(
                rendered_prompt,
                config=runnable_config,
                extra_body=extra_body,
            )
        except TypeError:
            # 某些 LangChain model 不接受 extra_body，退回只传 config 的调用方式。
            response = await llm.ainvoke(rendered_prompt, config=runnable_config)
        if self.config.link_generations and prompt.prompt_id:
            # 绑定 generation 到 prompt 需要等待 Langfuse/LiteLLM 先写入 observation，因此异步轮询补链。
            self._schedule(
                self.link_generation_to_prompt(
                    trace_id=trace_id,
                    prompt=prompt,
                    generation_name=generation_name,
                )
            )
        elif prompt.prompt_id and not self.config.link_generations:
            logger.debug(
                "Langfuse generation prompt linking disabled; prompt metadata is sent with LiteLLM metadata "
                "prompt_name=%s prompt_version=%s",
                prompt.name,
                prompt.version,
            )
        return response

    async def link_generation_to_prompt(
        self,
        *,
        trace_id: str,
        prompt: LangfusePrompt,
        generation_name: str,
    ) -> None:
        # 方法说明：异步轮询 Langfuse，找到刚写入的 generation 后补上它对应的 prompt 信息。
        await asyncio.sleep(max(self.config.link_initial_delay_ms, 0) / 1000)
        for _ in range(max(self.config.link_max_retries, 1)):
            try:
                generations = await self.client.list_generations(trace_id)
                # 只绑定尚未关联 prompt 的最新 generation，避免重复写入同一个 observation。
                unbound = [
                    generation
                    for generation in generations
                    if _generation_id(generation)
                    and _generation_id(generation) not in self._bound_generation_ids
                    and _generation_matches(generation, generation_name)
                    and not generation.get("promptId")
                    and not generation.get("prompt")
                ]
                if unbound:
                    await self._bind_generation(_generation_id(unbound[-1]), trace_id, prompt)
                    return
            except Exception as exc:
                logger.warning("Langfuse generation prompt link poll failed trace_id=%s error=%s", trace_id, exc)
            await asyncio.sleep(max(self.config.link_poll_interval_ms, 1) / 1000)

    def callbacks(self) -> list[Any]:
        # 方法说明：当前实现通过 LiteLLM metadata 上报，不额外注册 LangChain callbacks。
        return []

    async def flush(self) -> None:
        # 方法说明：等待后台补链任务完成，并关闭 Langfuse HTTP 客户端。
        tasks = list(self._pending_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.close()

    async def _resolve_prompt(
        self,
        prompt_name: str | None,
        prompt_label: str | None,
        fallback_prompt: str,
    ) -> LangfusePrompt:
        # 方法说明：解析一次模型调用应该使用的提示词；指定名称失败时回退到调用方传入的文本。
        if prompt_name:
            try:
                return await self.get_prompt(prompt_name, label=prompt_label, fallback_prompt=fallback_prompt)
            except Exception as exc:
                logger.warning("Langfuse prompt resolution failed name=%s error=%s", prompt_name, exc)
        return LangfusePrompt(
            name=prompt_name or "inline",
            prompt_type="text",
            content=fallback_prompt,
            label=prompt_label or self.config.label,
        )

    async def _load_remote_prompt(self, name: str, label: str) -> LangfusePrompt | None:
        # 方法说明：尝试从远程 Langfuse 加载提示词，失败时返回空值让主流程继续走本地兜底。
        if not self.config.remote_prompts_enabled:
            return None
        if not self.config.host:
            return None
        try:
            # 远程 prompt 获取失败不能影响诊断主流程，失败后会继续走本地 fallback。
            data = await self.client.get_prompt(name, label)
            if data is None:
                return None
            return _parse_prompt_response(data, default_name=name, default_label=label)
        except Exception as exc:
            logger.warning("Langfuse prompt fetch failed name=%s label=%s error=%s", name, label, exc)
            return None

    async def _load_custom_fallback(self, name: str, label: str) -> LangfusePrompt | None:
        # 方法说明：调用项目自定义的提示词兜底函数，适配测试或特殊部署环境。
        if self.fallback_provider is None:
            return None
        try:
            raw = self.fallback_provider(name, label)
            if asyncio.iscoroutine(raw):
                raw = await raw
            return _coerce_prompt(raw, name=name, label=label)
        except Exception as exc:
            logger.warning("Langfuse custom prompt fallback failed name=%s error=%s", name, exc)
            return None

    def _load_local_prompt(self, name: str, label: str) -> LangfusePrompt | None:
        # 方法说明：在本地 prompts 目录里查找提示词文件，支持 txt、json、yaml 等格式。
        search_dirs = [
            _resolve_path(self.config.prompt_dir),
            PROJECT_ROOT / "prompts",
            PROJECT_ROOT / "config" / "prompts",
        ]
        for directory in search_dirs:
            for suffix in (".txt", ".json", ".yaml", ".yml"):
                path = directory / f"{name}{suffix}"
                if not path.exists():
                    continue
                try:
                    return _load_prompt_file(path, name=name, label=label)
                except Exception as exc:
                    logger.warning("Langfuse local prompt load failed path=%s error=%s", path, exc)
        return None

    async def _bind_generation(self, generation_id: str | None, trace_id: str, prompt: LangfusePrompt) -> None:
        # 方法说明：把已经产生的模型 generation 和提示词版本关联起来，便于在 Langfuse 页面追溯。
        if not generation_id:
            return
        try:
            await self.client.ingest(
                [
                    _event(
                        "generation-create",
                        {
                            "id": generation_id,
                            "traceId": trace_id,
                            "promptId": prompt.prompt_id,
                            "promptName": prompt.name,
                            "promptVersion": prompt.version,
                        },
                    )
                ]
            )
            self._bound_generation_ids.add(generation_id)
            if len(self._bound_generation_ids) > 10000:
                self._bound_generation_ids.clear()
        except Exception as exc:
            logger.warning("Langfuse generation prompt bind failed generation_id=%s error=%s", generation_id, exc)

    def _schedule(self, coroutine: Any) -> None:
        # 方法说明：把补链任务放到后台执行，避免阻塞当前模型调用返回。
        try:
            task = asyncio.create_task(coroutine)
        except RuntimeError:
            return
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)


def build_langfuse_prompt_service(config: LangfuseConfig) -> LangfusePromptService | NullLangfusePromptService:
    # 方法说明：根据配置决定使用真实 Langfuse 服务还是空实现，让业务代码不用到处判断开关。
    if not config.enabled:
        return NullLangfusePromptService()
    if not config.host or not config.public_key or not config.secret_key:
        logger.warning("Langfuse is enabled but host/public_key/secret_key is incomplete; using null service.")
        return NullLangfusePromptService()
    return LangfusePromptService(config)


def render_prompt(prompt: LangfusePrompt, variables: dict[str, Any]) -> str | list[dict[str, str]]:
    # 方法说明：把提示词模板里的变量替换成真实值，chat 类型会保留 role/content 消息结构。
    if prompt.prompt_type == "chat" and isinstance(prompt.content, list):
        # chat prompt 保留 role/content 结构，便于传给支持消息数组的模型客户端。
        return [
            {
                "role": str(message.get("role", "")),
                "content": _render_template(str(message.get("content", "")), variables),
            }
            for message in prompt.content
        ]
    return _render_template(str(prompt.content), variables)


def _render_template(template: str, variables: dict[str, Any]) -> str:
    # 方法说明：渲染模板文本中的 {{变量}} 和兼容旧格式的 {变量} 占位符。
    escaped: list[str] = []

    def protect_escaped(match: re.Match[str]) -> str:
        # 支持 \{{var}} 形式的转义变量，避免模板渲染误替换示例文本。
        # 方法说明：先保护转义变量，等正常变量替换完再把原文放回去。
        escaped.append(match.group(0)[1:])
        return f"__LANGFUSE_ESCAPED_VAR_{len(escaped) - 1}__"

    protected = re.sub(r"\\\{\{\s*[A-Za-z_][A-Za-z0-9_.-]*\s*\}\}", protect_escaped, template)

    def replace(match: re.Match[str]) -> str:
        # 方法说明：把单个 {{变量}} 替换成 variables 里对应的值，找不到就替换成空字符串。
        key = match.group(1)
        value = _lookup_variable(variables, key)
        return str(value if value is not None else "")

    rendered = _TEMPLATE_PATTERN.sub(replace, protected)
    for key, value in variables.items():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
            # 兼容旧模板里的 {var} 占位符，同时保留 Langfuse 推荐的 {{var}} 写法。
            rendered = rendered.replace(f"{{{key}}}", str(value))
    for index, value in enumerate(escaped):
        rendered = rendered.replace(f"__LANGFUSE_ESCAPED_VAR_{index}__", value)
    return rendered


def _lookup_variable(variables: dict[str, Any], key: str) -> Any:
    # 方法说明：支持 a.b.c 这种嵌套变量读取，让提示词可以引用复杂上下文里的字段。
    value: Any = variables
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _parse_prompt_response(data: dict[str, Any], *, default_name: str, default_label: str) -> LangfusePrompt:
    # 方法说明：把 Langfuse API 返回的提示词 JSON 转成项目内部的 LangfusePrompt 对象。
    content = data.get("prompt")
    prompt_type = str(data.get("type") or ("chat" if isinstance(content, list) else "text"))
    return LangfusePrompt(
        name=str(data.get("name") or default_name),
        prompt_type=prompt_type,
        content=content if isinstance(content, (str, list)) else str(content or ""),
        version=data.get("version"),
        prompt_id=str(data.get("id") or data.get("promptId") or "") or None,
        label=default_label,
    )


def _coerce_prompt(raw: LangfusePrompt | str | dict[str, Any] | None, *, name: str, label: str) -> LangfusePrompt | None:
    # 方法说明：把字符串、字典或已有对象统一转换成 LangfusePrompt，方便上层统一处理。
    if raw is None:
        return None
    if isinstance(raw, LangfusePrompt):
        return raw
    if isinstance(raw, str):
        return LangfusePrompt(name=name, prompt_type="text", content=raw, label=label)
    content = raw.get("prompt") or raw.get("content") or ""
    prompt_type = str(raw.get("type") or ("chat" if isinstance(content, list) else "text"))
    return LangfusePrompt(
        name=str(raw.get("name") or name),
        prompt_type=prompt_type,
        content=content if isinstance(content, (str, list)) else str(content),
        version=raw.get("version"),
        prompt_id=str(raw.get("id") or raw.get("promptId") or "") or None,
        label=label,
    )


def _load_prompt_file(path: Path, *, name: str, label: str) -> LangfusePrompt:
    # 方法说明：读取本地提示词文件，并按文件类型解析成统一提示词对象。
    if path.suffix == ".txt":
        return LangfusePrompt(name=name, prompt_type="text", content=path.read_text(encoding="utf-8"), label=label)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Prompt JSON root must be an object.")
        return _coerce_prompt(data, name=name, label=label) or LangfusePrompt(name=name, prompt_type="text", content="", label=label)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Prompt YAML root must be an object.")
    if "prompt" in data or "content" in data:
        coerced = _coerce_prompt(data, name=name, label=label)
        if coerced is not None:
            return coerced
    system = str(data.get("system", ""))
    human = str(data.get("human", ""))
    return LangfusePrompt(name=name, prompt_type="text", content=f"{system}\n\n{human}".strip(), label=label)


def _merge_runnable_config(
    config: dict[str, Any] | None,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    # 方法说明：把 Langfuse 元数据合并进 LangChain runnable config，随模型调用一起传下去。
    merged = dict(config or {})
    merged["metadata"] = {**dict(merged.get("metadata") or {}), **metadata}
    return merged


def _event(event_type: str, body: dict[str, Any]) -> dict[str, Any]:
    # 方法说明：构造 Langfuse ingestion API 需要的事件结构，并自动去掉空字段。
    return {
        "id": f"evt-{uuid.uuid4()}",
        "type": event_type,
        "body": {key: value for key, value in body.items() if value is not None},
    }


def _generation_id(generation: dict[str, Any]) -> str | None:
    # 方法说明：从不同版本的 Langfuse observation 字段里提取 generation ID。
    value = generation.get("id") or generation.get("observationId")
    text = str(value or "").strip()
    return text or None


def _generation_matches(generation: dict[str, Any], generation_name: str) -> bool:
    # 方法说明：判断某个 generation 是否属于当前节点，避免把 prompt 绑定到错误调用上。
    name = str(generation.get("name") or generation.get("generationName") or "").strip()
    if name and name != generation_name:
        return False
    return True


def _tags_for(
    trace_context: dict[str, Any],
    metadata: dict[str, Any] | None,
    prompt: LangfusePrompt,
) -> list[str]:
    # 方法说明：合并工作流、节点和提示词标签，去重后写入 Langfuse 便于筛选。
    raw_tags = [*(trace_context.get("tags") or []), *((metadata or {}).get("tags") or [])]
    tags = [str(tag) for tag in raw_tags if str(tag or "").strip()]
    tags.extend(["agent-sentinel", str(trace_context.get("workflow_type") or "workflow"), f"prompt:{prompt.name}"])
    return list(dict.fromkeys(tags))


def _build_langfuse_metadata(
    *,
    trace_context: dict[str, Any],
    metadata: dict[str, Any] | None,
    prompt: LangfusePrompt,
) -> dict[str, Any]:
    # 方法说明：构造一次模型调用的完整观测元数据，串联业务 trace、节点名、prompt 版本和标签。
    metadata = metadata or {}
    business_trace_id = str(
        metadata.get("business_trace_id")
        or trace_context.get("business_trace_id")
        or metadata.get("trace_id")
        or trace_context.get("trace_id")
        or uuid.uuid4()
    )
    # business_trace_id 用于业务日志关联；Langfuse/OTEL 自身的 trace id 不一定完全等同于它。
    workflow_type = str(metadata.get("workflow_type") or trace_context.get("workflow_type") or "workflow")
    node_name = str(metadata.get("node_name") or prompt.name)
    generation_name = str(metadata.get("generation_name") or node_name or prompt.name)
    trace_name = str(metadata.get("trace_name") or trace_context.get("trace_name") or workflow_type)
    return {
        **trace_context,
        **metadata,
        # Compatibility field for LiteLLM integrations that still look for metadata.trace_id.
        "trace_id": business_trace_id,
        # Explicit business correlation id. We do not rely on OTEL/LiteLLM using this as the Langfuse trace id.
        "business_trace_id": business_trace_id,
        "trace_name": trace_name,
        "generation_name": generation_name,
        "node_name": node_name,
        "workflow_type": workflow_type,
        "tags": _tags_for(trace_context, metadata, prompt),
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_id": prompt.prompt_id,
        "prompt_label": prompt.label,
        "otel_trace_source": "litellm",
    }


def _resolve_path(path: str | Path) -> Path:
    # 方法说明：把提示词目录路径解析成绝对路径，相对路径默认基于项目根目录。
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
