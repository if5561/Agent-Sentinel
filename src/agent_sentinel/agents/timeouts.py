from __future__ import annotations

from agent_sentinel.llm.executor import LLMExecutor


def llm_timeout_seconds(llm: LLMExecutor) -> float:
    # 从 LLM 执行器读取超时时间，并保证至少 1 秒，避免等待模型响应时出现无效的超时配置。
    return max(float(getattr(llm, "timeout_seconds", 10.0)), 1.0)
