from __future__ import annotations

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_sentinel.config import Settings
from agent_sentinel.llm import build_chat_model
from agent_sentinel.llm.executor import _estimate_tokens
from agent_sentinel.monitoring import monitor

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 SingleTurnChatService 组件，集中管理这个模块的状态和行为。
class SingleTurnChatService:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, settings: Settings) -> None:
        # 创建普通聊天链路，主要用于简单问答或验证模型配置。
        self.settings = settings
        # 将 self.model 的值保存下来，供后续流程判断或组装响应时使用。
        self.model = build_chat_model(settings)
        # 将 self.prompt 的值保存下来，供后续流程判断或组装响应时使用。
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "system",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "You are Agent Sentinel, a concise and helpful assistant. "
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Answer the user's latest message clearly and directly.",
                ),
                # 执行当前业务步骤，推动流程继续向下游推进。
                ("human", "{user_input}"),
            ]
        )
        # 将 self.chain 的值保存下来，供后续流程判断或组装响应时使用。
        self.chain = self.prompt | self.model | StrOutputParser()

    # 定义 reply_once 相关的处理逻辑，供流程或外部调用复用。
    def reply_once(self, user_input: str) -> str:
        # 把用户输入交给普通聊天链路，并记录模型调用耗时和 token 估算。
        payload = {"user_input": user_input}
        # 将 prompt_text 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_text = self.prompt.invoke(payload).to_string()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _invoke_chain_with_monitoring(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.chain,
            # 执行当前业务步骤，推动流程继续向下游推进。
            payload,
            # 将 model 的值保存下来，供后续流程判断或组装响应时使用。
            model=self.settings.openai_model,
            # 将 prompt_text 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_text=prompt_text,
            # 将 operation 的值保存下来，供后续流程判断或组装响应时使用。
            operation="single_turn_chat",
        )


# 定义 AlertAnalysisService 组件，集中管理这个模块的状态和行为。
class AlertAnalysisService:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, settings: Settings) -> None:
        # 创建旧版单步告警分析链路，用固定提示词直接生成简短分析。
        self.settings = settings
        # 将 self.model 的值保存下来，供后续流程判断或组装响应时使用。
        self.model = build_chat_model(settings)
        # 将 self.prompt 的值保存下来，供后续流程判断或组装响应时使用。
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "system",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "You are an incident analysis assistant. "
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Reply in concise Chinese, but keep the section labels exactly as below. "
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Use one short line per item and avoid filler.\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "[Alert Analysis]\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Severity: ...\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Possible Cause: ...\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Impact: ...\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Suggested Action: ...\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Needs Human Follow-up: Yes/No + reason",
                ),
                (
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "human",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Source: {source}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Level: {level}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Summary: {summary}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Details: {details}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Raw Text: {raw_text}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Trigger Type: {trigger_type}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Tags: {tags}",
                ),
            ]
        )
        # 将 self.chain 的值保存下来，供后续流程判断或组装响应时使用。
        self.chain = self.prompt | self.model | StrOutputParser()

    # 定义 analyze_alert 相关的处理逻辑，供流程或外部调用复用。
    def analyze_alert(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        level: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        summary: str,
        # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
        details: str | None = None,
        # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
        raw_text: str | None = None,
        # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
        trigger_type: str = "unknown",
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags: list[str] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> str:
        # 把告警字段填入提示词，调用模型生成一段结构化中文分析。
        payload = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source": source,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "level": level,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "summary": summary,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "details": details or "",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "raw_text": raw_text or "",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "trigger_type": trigger_type,
            # 调用 join 完成当前步骤需要的业务处理。
            "tags": ", ".join(tags or []),
        }
        # 将 prompt_text 的值保存下来，供后续流程判断或组装响应时使用。
        prompt_text = self.prompt.invoke(payload).to_string()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _invoke_chain_with_monitoring(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.chain,
            # 执行当前业务步骤，推动流程继续向下游推进。
            payload,
            # 将 model 的值保存下来，供后续流程判断或组装响应时使用。
            model=self.settings.openai_model,
            # 将 prompt_text 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_text=prompt_text,
            # 将 operation 的值保存下来，供后续流程判断或组装响应时使用。
            operation="alert_analysis",
        )


# 定义 _invoke_chain_with_monitoring 相关的处理逻辑，供流程或外部调用复用。
def _invoke_chain_with_monitoring(chain: object, payload: dict[str, str], *, model: str, prompt_text: str, operation: str) -> str:
    # 统一执行 LangChain 调用，并把耗时、错误和 token 估算写入监控。
    started = time.perf_counter()
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = chain.invoke(payload)  # type: ignore[attr-defined]
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except TimeoutError:
        # 调用 monitor.record_error 完成当前步骤需要的业务处理。
        monitor.record_error("llm_timeout")
        # 执行当前业务步骤，推动流程继续向下游推进。
        raise
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception:
        # 调用 monitor.record_error 完成当前步骤需要的业务处理。
        monitor.record_error("llm_error")
        # 执行当前业务步骤，推动流程继续向下游推进。
        raise

    # 将 duration_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    duration_seconds = time.perf_counter() - started
    # 将 content 的值保存下来，供后续流程判断或组装响应时使用。
    content = str(result)
    # 将 prompt_tokens 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_tokens = _estimate_tokens(prompt_text)
    # 将 completion_tokens 的值保存下来，供后续流程判断或组装响应时使用。
    completion_tokens = _estimate_tokens(content)
    # 调用 monitor.record_llm_call 完成当前步骤需要的业务处理。
    monitor.record_llm_call(model, duration_seconds)
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    monitor.record_tokens(model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info(
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "LLM chain call completed operation=%s model=%s elapsed_ms=%s prompt_chars=%s response_chars=%s "
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "prompt_tokens=%s completion_tokens=%s token_usage_source=estimated",
        # 执行当前业务步骤，推动流程继续向下游推进。
        operation,
        # 执行当前业务步骤，推动流程继续向下游推进。
        model,
        # 调用 int 完成当前步骤需要的业务处理。
        int(duration_seconds * 1000),
        # 调用 len 完成当前步骤需要的业务处理。
        len(prompt_text),
        # 调用 len 完成当前步骤需要的业务处理。
        len(content),
        # 执行当前业务步骤，推动流程继续向下游推进。
        prompt_tokens,
        # 执行当前业务步骤，推动流程继续向下游推进。
        completion_tokens,
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return content
