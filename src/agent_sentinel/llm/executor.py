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
    # 执行当前业务步骤，推动流程继续向下游推进。
    NullLangfusePromptService,
    # 执行当前业务步骤，推动流程继续向下游推进。
    reset_trace_context,
    # 执行当前业务步骤，推动流程继续向下游推进。
    set_trace_context,
)

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 LLMExecutor 组件，集中管理这个模块的状态和行为。
class LLMExecutor:
    # 执行当前业务步骤，推动流程继续向下游推进。
    models: list[str]
    # 将 api_key 的值保存下来，供后续流程判断或组装响应时使用。
    api_key: str = ""
    # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
    base_url: str | None = None
    # 将 temperature 的值保存下来，供后续流程判断或组装响应时使用。
    temperature: float = 0.0
    # 将 trust_env 的值保存下来，供后续流程判断或组装响应时使用。
    trust_env: bool = False
    # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    timeout_seconds: float = 15.0
    # 将 max_retries 的值保存下来，供后续流程判断或组装响应时使用。
    max_retries: int = 2
    # 将 mock_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    mock_enabled: bool = False
    # 将 langfuse 的值保存下来，供后续流程判断或组装响应时使用。
    langfuse: object = field(default_factory=NullLangfusePromptService)

    # 定义 call 相关的处理逻辑，供流程或外部调用复用。
    async def call(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_name: str | None = None,
        # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_variables: dict[str, object] | None = None,
        # 将 prompt_label 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_label: str | None = None,
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata: dict[str, object] | None = None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        **kwargs: object,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> str:
        # 统一执行一次模型调用，负责 mock、本地监控、多模型降级和最终错误汇总。
        if self.mock_enabled or not self.api_key:
            # mock 分支用于本地开发和测试，仍然记录耗时/token 指标以保持监控面板可用。
            logger.info("LLM mock call started prompt_chars=%s", len(prompt))
            # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
            started = time.perf_counter()
            # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
            response = self._mock_response(prompt, **kwargs)
            # 将 duration_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            duration_seconds = time.perf_counter() - started
            # 将 prompt_tokens 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_tokens = _estimate_tokens(prompt)
            # 将 completion_tokens 的值保存下来，供后续流程判断或组装响应时使用。
            completion_tokens = _estimate_tokens(response)
            # 调用 monitor.record_llm_call 完成当前步骤需要的业务处理。
            monitor.record_llm_call("mock", duration_seconds)
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            monitor.record_tokens("mock", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "LLM mock call completed model=mock elapsed_ms=%s prompt_chars=%s response_chars=%s "
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "prompt_tokens=%s completion_tokens=%s token_usage_source=estimated",
                # 调用 int 完成当前步骤需要的业务处理。
                int(duration_seconds * 1000),
                # 调用 len 完成当前步骤需要的业务处理。
                len(prompt),
                # 调用 len 完成当前步骤需要的业务处理。
                len(response),
                # 执行当前业务步骤，推动流程继续向下游推进。
                prompt_tokens,
                # 执行当前业务步骤，推动流程继续向下游推进。
                completion_tokens,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return response

        # 将 last_exc 的值保存下来，供后续流程判断或组装响应时使用。
        last_exc: Exception | None = None
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for model in self.models:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 多模型按顺序尝试，前一个模型失败后自动降级到下一个模型。
                return await self._call_with_retries(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    model,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    prompt,
                    # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_name=prompt_name,
                    # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_variables=prompt_variables,
                    # 将 prompt_label 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_label=prompt_label,
                    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                    metadata=metadata,
                )
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception as exc:  # pragma: no cover - depends on remote LLM
                # 将 last_exc 的值保存下来，供后续流程判断或组装响应时使用。
                last_exc = exc
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("llm_timeout" if isinstance(exc, TimeoutError) else "llm_error")
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.exception("LLM model failed model=%s", model)

        # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
        raise RuntimeError("All LLM models failed.") from last_exc

    # 定义 _call_with_retries 相关的处理逻辑，供流程或外部调用复用。
    async def _call_with_retries(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        model_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_name: str | None = None,
        # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_variables: dict[str, object] | None = None,
        # 将 prompt_label 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_label: str | None = None,
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata: dict[str, object] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> str:
        # 单个模型内部用指数退避重试，模型列表层面再做跨模型 fallback。
        # 对单个模型执行带超时和重试的真实调用，并记录耗时与 token 使用量。
        async for attempt in AsyncRetrying(
            # 将 stop 的值保存下来，供后续流程判断或组装响应时使用。
            stop=stop_after_attempt(self.max_retries),
            # 将 wait 的值保存下来，供后续流程判断或组装响应时使用。
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            # 将 retry 的值保存下来，供后续流程判断或组装响应时使用。
            retry=retry_if_exception_type(Exception),
            # 将 reraise 的值保存下来，供后续流程判断或组装响应时使用。
            reraise=True,
        ):
            # 进入上下文管理器保护的区域，自动处理资源生命周期。
            with attempt:
                # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
                started = time.perf_counter()
                # 进入异步上下文管理区域，确保异步资源按约定释放。
                async with asyncio.timeout(self.timeout_seconds):
                    # 每次尝试重新构建 ChatModel，避免失败连接或客户端状态污染后续重试。
                    model = build_chat_model(
                        # 将 api_key 的值保存下来，供后续流程判断或组装响应时使用。
                        api_key=self.api_key,
                        # 将 model 的值保存下来，供后续流程判断或组装响应时使用。
                        model=model_name,
                        # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
                        base_url=self.base_url,
                        # 将 temperature 的值保存下来，供后续流程判断或组装响应时使用。
                        temperature=self.temperature,
                        # 将 trust_env 的值保存下来，供后续流程判断或组装响应时使用。
                        trust_env=self.trust_env,
                    )
                    # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
                    response = await self.langfuse.call_llm_with_trace(  # type: ignore[attr-defined]
                        # 将 llm 的值保存下来，供后续流程判断或组装响应时使用。
                        llm=model,
                        # 将 fallback_prompt 的值保存下来，供后续流程判断或组装响应时使用。
                        fallback_prompt=prompt,
                        # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
                        prompt_name=prompt_name,
                        # 将 variables 的值保存下来，供后续流程判断或组装响应时使用。
                        variables=prompt_variables,
                        # 将 prompt_label 的值保存下来，供后续流程判断或组装响应时使用。
                        prompt_label=prompt_label,
                        # 将 model_name 的值保存下来，供后续流程判断或组装响应时使用。
                        model_name=model_name,
                        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                        metadata=metadata,
                    )
                # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
                content = str(getattr(response, "content", response))
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                prompt_tokens, completion_tokens = _extract_token_usage(response)
                # 优先使用模型响应里的真实 token usage；没有返回时退化为字符长度估算。
                observed_prompt_tokens = prompt_tokens or _estimate_tokens(prompt)
                # 将 observed_completion_tokens 的值保存下来，供后续流程判断或组装响应时使用。
                observed_completion_tokens = completion_tokens or _estimate_tokens(content)
                # 将 token_usage_source 的值保存下来，供后续流程判断或组装响应时使用。
                token_usage_source = "response_usage" if prompt_tokens or completion_tokens else "estimated"
                # 调用 monitor.record_llm_call 完成当前步骤需要的业务处理。
                monitor.record_llm_call(model_name, time.perf_counter() - started)
                # 调用 monitor.record_tokens 完成当前步骤需要的业务处理。
                monitor.record_tokens(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    model_name,
                    # 将 prompt_tokens 的值保存下来，供后续流程判断或组装响应时使用。
                    prompt_tokens=observed_prompt_tokens,
                    # 将 completion_tokens 的值保存下来，供后续流程判断或组装响应时使用。
                    completion_tokens=observed_completion_tokens,
                )
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "LLM call completed model=%s elapsed_ms=%s prompt_chars=%s response_chars=%s "
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "prompt_tokens=%s completion_tokens=%s token_usage_source=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    model_name,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    elapsed_ms,
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(prompt),
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(content),
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    observed_prompt_tokens,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    observed_completion_tokens,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    token_usage_source,
                )
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.debug("LLM response model=%s content=%s", model_name, content)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return content

        # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
        raise RuntimeError("LLM retry loop exited unexpectedly.")

    # 定义 set_trace_context 相关的处理逻辑，供流程或外部调用复用。
    def set_trace_context(self, **values: object) -> object:
        # trace context 由工作流设置，LLM 调用层只负责透传给观测实现。
        # 把当前告警、任务等上下文写入观测追踪，后续模型调用会自动带上这些信息。
        return set_trace_context(**values)

    # 定义 reset_trace_context 相关的处理逻辑，供流程或外部调用复用。
    def reset_trace_context(self, token: object) -> None:
        # 模型调用结束后恢复旧的追踪上下文，避免不同诊断任务之间串数据。
        reset_trace_context(token)  # type: ignore[arg-type]

    # 定义 callbacks 相关的处理逻辑，供流程或外部调用复用。
    def callbacks(self) -> list[object]:
        # 暴露观测系统需要的回调列表，让 LangChain 执行过程能被采集。
        return self.langfuse.callbacks()  # type: ignore[attr-defined]

    # 定义 flush_observability 相关的处理逻辑，供流程或外部调用复用。
    async def flush_observability(self) -> None:
        # 刷新观测缓冲区，确保本次诊断产生的追踪数据尽快写到外部系统。
        await self.langfuse.flush()  # type: ignore[attr-defined]

    # 定义 _mock_response 相关的处理逻辑，供流程或外部调用复用。
    def _mock_response(self, prompt: str, **_: object) -> str:
        # 在没有真实模型时按提示词类型返回模拟结果，让完整诊断流程仍可本地演示。
        lower = prompt.lower()
        # 根据提示词特征返回结构化 mock，保证不同节点的解析逻辑在无真实模型时也能跑通。
        if "feishu interactive alert workflow" in lower or "不要输出 json" in lower:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return (
                # 执行当前业务步骤，推动流程继续向下游推进。
                "## 故障判断\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "服务延迟或超时可能与下游依赖异常有关。\n\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "## 关键证据\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "- 告警内容包含 timeout 或失败特征。\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "- RAG 与实时工具结果提供了上下文线索。\n\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "## 处置建议\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "1. 优先检查下游依赖的超时和错误率。\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "2. 查看告警时间窗口内的 ERROR 日志。\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "3. 确认近期拓扑或配置变更。\n\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "## 风险与观察项\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "- 避免直接执行高风险删除或重启操作。\n"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "- 持续观察错误率、延迟和恢复趋势。"
            )
        # 根据 "recommended_plan" in lower or "evidence" in lower 判断当前流程该进入哪个处理分支。
        if "recommended_plan" in lower or "evidence" in lower:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return json.dumps(
                {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "recommended_plan": {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "summary": "Mock diagnosis: service latency is likely caused by downstream timeout.",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "actions": [
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "Check recent timeout spikes on the order-sync dependency.",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "Inspect ERROR logs around the alert timestamp.",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "Verify upstream topology changes before rollback.",
                        ],
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "risk_level": "medium",
                    },
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "evidence": [
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "Alert summary mentions timeout or failure.",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "Mock metrics/logs/topology were attached to the diagnosis state.",
                    ],
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "need_human": True,
                },
                # 将 ensure_ascii 的值保存下来，供后续流程判断或组装响应时使用。
                ensure_ascii=False,
            )
        # 根据 "校验" in prompt or "validate" in lower 判断当前流程该进入哪个处理分支。
        if "校验" in prompt or "validate" in lower:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "PASS"
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "Detected ERROR alert for order synchronization timeout. Need live metrics, logs, and topology context."


# 定义 _extract_token_usage 相关的处理逻辑，供流程或外部调用复用。
def _extract_token_usage(response: object) -> tuple[int, int]:
    """Extract token usage from common LangChain/OpenAI response shapes."""
    # 从不同模型响应格式里提取输入和输出 token 数，提取不到时返回 0 交给估算逻辑。
    usage = getattr(response, "usage_metadata", None)
    # LangChain 不同版本/不同 provider 的 token 字段位置不完全一致，这里做兼容提取。
    if isinstance(usage, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0), int(
            # 调用 usage.get 完成当前步骤需要的业务处理。
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
    metadata = getattr(response, "response_metadata", None)
    # 根据 isinstance(metadata, dict) 判断当前流程该进入哪个处理分支。
    if isinstance(metadata, dict):
        # 将 token_usage 的值保存下来，供后续流程判断或组装响应时使用。
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        # 根据 isinstance(token_usage, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(token_usage, dict):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0), int(
                # 调用 token_usage.get 完成当前步骤需要的业务处理。
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return 0, 0


# 定义 _estimate_tokens 相关的处理逻辑，供流程或外部调用复用。
def _estimate_tokens(text: str) -> int:
    # 估算值只用于监控趋势，不用于计费或精确配额控制。
    # 用字符长度粗略估算 token 数，保证没有真实用量时监控图也能看到趋势。
    return max(1, len(text) // 4) if text else 0
