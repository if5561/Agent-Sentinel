from __future__ import annotations

from agent_sentinel.llm.executor import LLMExecutor


def llm_timeout_seconds(llm: LLMExecutor) -> float:
    return max(float(getattr(llm, "timeout_seconds", 10.0)), 1.0)
