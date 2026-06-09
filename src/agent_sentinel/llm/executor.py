from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_sentinel.llm.client import build_chat_model
from agent_sentinel.monitoring import monitor
from agent_sentinel.observability.langfuse import (
    NullLangfusePromptService,
    reset_trace_context,
    set_trace_context,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMExecutor:
    models: list[str]
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.0
    trust_env: bool = False
    timeout_seconds: float = 15.0
    max_retries: int = 2
    mock_enabled: bool = False
    langfuse: object = field(default_factory=NullLangfusePromptService)

    async def call(
        self,
        prompt: str,
        *,
        prompt_name: str | None = None,
        prompt_variables: dict[str, object] | None = None,
        prompt_label: str | None = None,
        metadata: dict[str, object] | None = None,
        **kwargs: object,
    ) -> str:
        if self.mock_enabled or not self.api_key:
            logger.info("LLM mock call started prompt_chars=%s", len(prompt))
            started = time.perf_counter()
            response = self._mock_response(prompt, **kwargs)
            duration_seconds = time.perf_counter() - started
            prompt_tokens = _estimate_tokens(prompt)
            completion_tokens = _estimate_tokens(response)
            monitor.record_llm_call("mock", duration_seconds)
            monitor.record_tokens("mock", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            logger.info(
                "LLM mock call completed model=mock elapsed_ms=%s prompt_chars=%s response_chars=%s "
                "prompt_tokens=%s completion_tokens=%s token_usage_source=estimated",
                int(duration_seconds * 1000),
                len(prompt),
                len(response),
                prompt_tokens,
                completion_tokens,
            )
            return response

        last_exc: Exception | None = None
        for model in self.models:
            try:
                return await self._call_with_retries(
                    model,
                    prompt,
                    prompt_name=prompt_name,
                    prompt_variables=prompt_variables,
                    prompt_label=prompt_label,
                    metadata=metadata,
                )
            except Exception as exc:  # pragma: no cover - depends on remote LLM
                last_exc = exc
                monitor.record_error("llm_timeout" if isinstance(exc, TimeoutError) else "llm_error")
                logger.exception("LLM model failed model=%s", model)

        raise RuntimeError("All LLM models failed.") from last_exc

    async def _call_with_retries(
        self,
        model_name: str,
        prompt: str,
        *,
        prompt_name: str | None = None,
        prompt_variables: dict[str, object] | None = None,
        prompt_label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                started = time.perf_counter()
                async with asyncio.timeout(self.timeout_seconds):
                    model = build_chat_model(
                        api_key=self.api_key,
                        model=model_name,
                        base_url=self.base_url,
                        temperature=self.temperature,
                        trust_env=self.trust_env,
                    )
                    response = await self.langfuse.call_llm_with_trace(  # type: ignore[attr-defined]
                        llm=model,
                        fallback_prompt=prompt,
                        prompt_name=prompt_name,
                        variables=prompt_variables,
                        prompt_label=prompt_label,
                        model_name=model_name,
                        metadata=metadata,
                    )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                content = str(getattr(response, "content", response))
                prompt_tokens, completion_tokens = _extract_token_usage(response)
                observed_prompt_tokens = prompt_tokens or _estimate_tokens(prompt)
                observed_completion_tokens = completion_tokens or _estimate_tokens(content)
                token_usage_source = "response_usage" if prompt_tokens or completion_tokens else "estimated"
                monitor.record_llm_call(model_name, time.perf_counter() - started)
                monitor.record_tokens(
                    model_name,
                    prompt_tokens=observed_prompt_tokens,
                    completion_tokens=observed_completion_tokens,
                )
                logger.info(
                    "LLM call completed model=%s elapsed_ms=%s prompt_chars=%s response_chars=%s "
                    "prompt_tokens=%s completion_tokens=%s token_usage_source=%s",
                    model_name,
                    elapsed_ms,
                    len(prompt),
                    len(content),
                    observed_prompt_tokens,
                    observed_completion_tokens,
                    token_usage_source,
                )
                logger.debug("LLM response model=%s content=%s", model_name, content)
                return content

        raise RuntimeError("LLM retry loop exited unexpectedly.")

    def set_trace_context(self, **values: object) -> object:
        return set_trace_context(**values)

    def reset_trace_context(self, token: object) -> None:
        reset_trace_context(token)  # type: ignore[arg-type]

    def callbacks(self) -> list[object]:
        return self.langfuse.callbacks()  # type: ignore[attr-defined]

    async def flush_observability(self) -> None:
        await self.langfuse.flush()  # type: ignore[attr-defined]

    def _mock_response(self, prompt: str, **_: object) -> str:
        lower = prompt.lower()
        if "feishu interactive alert workflow" in lower or "不要输出 json" in lower:
            return (
                "## 故障判断\n"
                "服务延迟或超时可能与下游依赖异常有关。\n\n"
                "## 关键证据\n"
                "- 告警内容包含 timeout 或失败特征。\n"
                "- RAG 与实时工具结果提供了上下文线索。\n\n"
                "## 处置建议\n"
                "1. 优先检查下游依赖的超时和错误率。\n"
                "2. 查看告警时间窗口内的 ERROR 日志。\n"
                "3. 确认近期拓扑或配置变更。\n\n"
                "## 风险与观察项\n"
                "- 避免直接执行高风险删除或重启操作。\n"
                "- 持续观察错误率、延迟和恢复趋势。"
            )
        if "recommended_plan" in lower or "evidence" in lower:
            return json.dumps(
                {
                    "recommended_plan": {
                        "summary": "Mock diagnosis: service latency is likely caused by downstream timeout.",
                        "actions": [
                            "Check recent timeout spikes on the order-sync dependency.",
                            "Inspect ERROR logs around the alert timestamp.",
                            "Verify upstream topology changes before rollback.",
                        ],
                        "risk_level": "medium",
                    },
                    "evidence": [
                        "Alert summary mentions timeout or failure.",
                        "Mock metrics/logs/topology were attached to the diagnosis state.",
                    ],
                    "need_human": True,
                },
                ensure_ascii=False,
            )
        if "校验" in prompt or "validate" in lower:
            return "PASS"
        return "Detected ERROR alert for order synchronization timeout. Need live metrics, logs, and topology context."


def _extract_token_usage(response: object) -> tuple[int, int]:
    """Extract token usage from common LangChain/OpenAI response shapes."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0), int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        if isinstance(token_usage, dict):
            return int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0), int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )
    return 0, 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
