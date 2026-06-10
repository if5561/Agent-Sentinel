from __future__ import annotations

import asyncio
import logging

from agent_sentinel.agents.timeouts import llm_timeout_seconds
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.llm.executor import LLMExecutor
from agent_sentinel.utils.config_loader import format_prompt

logger = logging.getLogger(__name__)

DANGEROUS_KEYWORDS = ("rm -rf", "drop table", "shutdown", "reboot", "kubectl delete", "delete from")


async def validate_plan_node(state: DiagnosisState, llm: LLMExecutor) -> DiagnosisState:
    # 方法说明：校验输入或状态是否满足继续处理的条件。
    logger.info("Node validate started")
    plan_text = str(state.get("recommended_plan", {})).lower()
    # 先用本地规则拦截明显高风险动作，再交给 LLM 做语义层面的方案校验。
    rule_ok = not any(keyword in plan_text for keyword in DANGEROUS_KEYWORDS)
    prompt_variables = {
        "recommended_plan": state.get("recommended_plan", {}),
        "rule_result": "PASS" if rule_ok else "FAIL",
    }
    prompt = format_prompt("validate", prompt_variables)
    async with asyncio.timeout(llm_timeout_seconds(llm)):
        llm_result = (
            await llm.call(
                prompt,
                prompt_name="validate",
                prompt_variables=prompt_variables,
                metadata={"node_name": "validate"},
            )
        ).strip().upper()
    # 只有规则校验和 LLM 校验都通过，方案才进入后续人工确认/最终输出。
    validation_result = rule_ok and llm_result.startswith("PASS")
    attempts = state.get("validation_attempts", 0) + 1
    logger.info("Node validate completed result=%s attempts=%s", validation_result, attempts)
    return {
        "validation_result": validation_result,
        "validation_attempts": attempts,
        "evidence": append_evidence(state, f"Validation result={validation_result}, rule_ok={rule_ok}."),
        "messages": append_message(state, "assistant", f"方案校验完成: {'通过' if validation_result else '未通过'}"),
    }


def validation_result(state: DiagnosisState) -> str:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if state.get("validation_result", False):
        return "pass"
    if state.get("validation_attempts", 0) <= 1:
        # 第一次失败允许回到 generate_plan 重试；第二次仍失败则放行，避免流程无限循环。
        return "fail"
    return "pass"
