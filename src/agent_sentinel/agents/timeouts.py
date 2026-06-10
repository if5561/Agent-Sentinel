from __future__ import annotations

from agent_sentinel.llm.executor import LLMExecutor


def llm_timeout_seconds(llm: LLMExecutor) -> float:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    return max(float(getattr(llm, "timeout_seconds", 10.0)), 1.0)
