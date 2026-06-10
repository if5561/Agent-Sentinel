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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 PlanOutput 组件，集中管理这个模块的状态和行为。
class PlanOutput(BaseModel):
    # 将 recommended_plan 的值保存下来，供后续流程判断或组装响应时使用。
    recommended_plan: dict[str, Any] = Field(default_factory=dict)
    # 将 evidence 的值保存下来，供后续流程判断或组装响应时使用。
    evidence: list[str] = Field(default_factory=list)
    # 将 need_human 的值保存下来，供后续流程判断或组装响应时使用。
    need_human: bool = True

    @field_validator("recommended_plan", mode="before")
    @classmethod
    # 定义 normalize_recommended_plan 相关的处理逻辑，供流程或外部调用复用。
    def normalize_recommended_plan(cls, value: Any) -> dict[str, Any]:
        # 模型输出可能是 dict/list/string，统一归一化成 dict 方便后续节点消费。
        # 把模型返回的方案统一转成字典，避免后续节点因为字符串或列表格式不同而处理失败。
        if isinstance(value, dict):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return value
        # 根据 isinstance(value, list) 判断当前流程该进入哪个处理分支。
        if isinstance(value, list):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"actions": value}
        # 根据 value is None 判断当前流程该进入哪个处理分支。
        if value is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {}
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"summary": str(value)}


@async_retry(max_attempts=2)
# 定义 generate_plan_node 相关的处理逻辑，供流程或外部调用复用。
async def generate_plan_node(state: DiagnosisState, llm: LLMExecutor) -> DiagnosisState:
    # 把告警摘要、检索资料、实时数据和证据交给模型，生成可执行的诊断/处理方案。
    logger.info("Node generate_plan started")
    # 将 parser 的值保存下来，供后续流程判断或组装响应时使用。
    parser = PydanticOutputParser(pydantic_object=PlanOutput)
    # 方案生成只读取前面节点沉淀的摘要、检索文档、实时数据和证据链。
    prompt_variables = {
        # 调用 state.get 完成当前步骤需要的业务处理。
        "alert_summary": state.get("alert_summary", ""),
        # 调用 state.get 完成当前步骤需要的业务处理。
        "retrieved_docs": state.get("retrieved_docs", []),
        # 调用 state.get 完成当前步骤需要的业务处理。
        "live_data": state.get("live_data", {}),
        # 调用 state.get 完成当前步骤需要的业务处理。
        "evidence": state.get("evidence", []),
    }
    # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
    prompt = format_prompt("generate_plan", prompt_variables)
    # 把 Pydantic 的格式说明追加进 prompt，降低模型返回非 JSON/非结构化文本的概率。
    prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
    # 进入异步上下文管理区域，确保异步资源按约定释放。
    async with asyncio.timeout(llm_timeout_seconds(llm)):
        # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
        raw = await llm.call(
            # 执行当前业务步骤，推动流程继续向下游推进。
            prompt,
            # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_name="generate_plan",
            # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_variables={
                # 执行当前业务步骤，推动流程继续向下游推进。
                **prompt_variables,
                # 调用 parser.get_format_instructions 完成当前步骤需要的业务处理。
                "format_instructions": parser.get_format_instructions(),
            },
            # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
            metadata={"node_name": "generate_plan"},
        )
    # 将 parsed 的值保存下来，供后续流程判断或组装响应时使用。
    parsed = parser.parse(raw)
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Node generate_plan completed evidence=%s", len(parsed.evidence))
    # 返回 state 增量，由 LangGraph 合并到全局诊断状态。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "recommended_plan": parsed.recommended_plan,
        # 调用 append_evidence 完成当前步骤需要的业务处理。
        "evidence": append_evidence(state, *parsed.evidence),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "need_human": parsed.need_human,
        # 调用 append_message 完成当前步骤需要的业务处理。
        "messages": append_message(state, "assistant", "诊断方案生成完成，已附带证据链。"),
    }
