from __future__ import annotations

from agent_sentinel.llm.executor import LLMExecutor


# 定义 llm_timeout_seconds 相关的处理逻辑，供流程或外部调用复用。
def llm_timeout_seconds(llm: LLMExecutor) -> float:
    # 从 LLM 执行器读取超时时间，并保证至少 1 秒，避免等待模型响应时出现无效的超时配置。
    return max(float(getattr(llm, "timeout_seconds", 10.0)), 1.0)
