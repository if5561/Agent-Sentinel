from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.agents.timeouts import llm_timeout_seconds
from agent_sentinel.llm.executor import LLMExecutor
from agent_sentinel.utils.config_loader import format_prompt
from agent_sentinel.utils.retry_utils import async_retry

logger = logging.getLogger(__name__)


class PlanOutput(BaseModel):
    recommended_plan: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    need_human: bool = True

    @field_validator("recommended_plan", mode="before")
    @classmethod
    def normalize_recommended_plan(cls, value: Any) -> dict[str, Any]:
        # 模型输出可能是 dict/list/string，统一归一化成 dict 方便后续节点消费。
        # 把模型返回的方案统一转成字典，避免后续节点因为字符串或列表格式不同而处理失败。
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"actions": value}
        if value is None:
            return {}
        return {"summary": str(value)}


@async_retry(max_attempts=2)
async def generate_plan_node(state: DiagnosisState, llm: LLMExecutor) -> DiagnosisState:
    # 把告警摘要、检索资料、实时数据和证据交给模型，生成可执行的诊断/处理方案。
    logger.info("Node generate_plan started")
    parser = PydanticOutputParser(pydantic_object=PlanOutput)
    # 方案生成只读取前面节点沉淀的摘要、检索文档、实时数据和证据链。
    prompt_variables = {
        "alert_summary": state.get("alert_summary", ""),
        "retrieved_docs": state.get("retrieved_docs", []),
        "live_data": state.get("live_data", {}),
        "evidence": state.get("evidence", []),
    }
    prompt = format_prompt("generate_plan", prompt_variables)
    # 把 Pydantic 的格式说明追加进 prompt，降低模型返回非 JSON/非结构化文本的概率。
    prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
    async with asyncio.timeout(llm_timeout_seconds(llm)):
        raw = await llm.call(
            prompt,
            prompt_name="generate_plan",
            prompt_variables={
                **prompt_variables,
                "format_instructions": parser.get_format_instructions(),
            },
            metadata={"node_name": "generate_plan"},
        )
    parsed = parser.parse(raw)
    logger.info("Node generate_plan completed evidence=%s", len(parsed.evidence))
    # 返回 state 增量，由 LangGraph 合并到全局诊断状态。
    return {
        "recommended_plan": parsed.recommended_plan,
        "evidence": append_evidence(state, *parsed.evidence),
        "need_human": parsed.need_human,
        "messages": append_message(state, "assistant", "诊断方案生成完成，已附带证据链。"),
    }
