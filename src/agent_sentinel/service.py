from __future__ import annotations

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_sentinel.config import Settings
from agent_sentinel.llm import build_chat_model
from agent_sentinel.llm.executor import _estimate_tokens
from agent_sentinel.monitoring import monitor

logger = logging.getLogger(__name__)


class SingleTurnChatService:
    def __init__(self, settings: Settings) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.settings = settings
        self.model = build_chat_model(settings)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Agent Sentinel, a concise and helpful assistant. "
                    "Answer the user's latest message clearly and directly.",
                ),
                ("human", "{user_input}"),
            ]
        )
        self.chain = self.prompt | self.model | StrOutputParser()

    def reply_once(self, user_input: str) -> str:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        payload = {"user_input": user_input}
        prompt_text = self.prompt.invoke(payload).to_string()
        return _invoke_chain_with_monitoring(
            self.chain,
            payload,
            model=self.settings.openai_model,
            prompt_text=prompt_text,
            operation="single_turn_chat",
        )


class AlertAnalysisService:
    def __init__(self, settings: Settings) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.settings = settings
        self.model = build_chat_model(settings)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an incident analysis assistant. "
                    "Reply in concise Chinese, but keep the section labels exactly as below. "
                    "Use one short line per item and avoid filler.\n"
                    "[Alert Analysis]\n"
                    "Severity: ...\n"
                    "Possible Cause: ...\n"
                    "Impact: ...\n"
                    "Suggested Action: ...\n"
                    "Needs Human Follow-up: Yes/No + reason",
                ),
                (
                    "human",
                    "Source: {source}\n"
                    "Level: {level}\n"
                    "Summary: {summary}\n"
                    "Details: {details}\n"
                    "Raw Text: {raw_text}\n"
                    "Trigger Type: {trigger_type}\n"
                    "Tags: {tags}",
                ),
            ]
        )
        self.chain = self.prompt | self.model | StrOutputParser()

    def analyze_alert(
        self,
        source: str,
        level: str,
        summary: str,
        details: str | None = None,
        raw_text: str | None = None,
        trigger_type: str = "unknown",
        tags: list[str] | None = None,
    ) -> str:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        payload = {
            "source": source,
            "level": level,
            "summary": summary,
            "details": details or "",
            "raw_text": raw_text or "",
            "trigger_type": trigger_type,
            "tags": ", ".join(tags or []),
        }
        prompt_text = self.prompt.invoke(payload).to_string()
        return _invoke_chain_with_monitoring(
            self.chain,
            payload,
            model=self.settings.openai_model,
            prompt_text=prompt_text,
            operation="alert_analysis",
        )


def _invoke_chain_with_monitoring(chain: object, payload: dict[str, str], *, model: str, prompt_text: str, operation: str) -> str:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    started = time.perf_counter()
    try:
        result = chain.invoke(payload)  # type: ignore[attr-defined]
    except TimeoutError:
        monitor.record_error("llm_timeout")
        raise
    except Exception:
        monitor.record_error("llm_error")
        raise

    duration_seconds = time.perf_counter() - started
    content = str(result)
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(content)
    monitor.record_llm_call(model, duration_seconds)
    monitor.record_tokens(model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    logger.info(
        "LLM chain call completed operation=%s model=%s elapsed_ms=%s prompt_chars=%s response_chars=%s "
        "prompt_tokens=%s completion_tokens=%s token_usage_source=estimated",
        operation,
        model,
        int(duration_seconds * 1000),
        len(prompt_text),
        len(content),
        prompt_tokens,
        completion_tokens,
    )
    return content
