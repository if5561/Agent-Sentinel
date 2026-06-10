from __future__ import annotations

import asyncio
import logging

from agent_sentinel.agents.timeouts import llm_timeout_seconds
from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.llm.executor import LLMExecutor
from agent_sentinel.utils.config_loader import format_prompt

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 将 DANGEROUS_KEYWORDS 的值保存下来，供后续流程判断或组装响应时使用。
DANGEROUS_KEYWORDS = ("rm -rf", "drop table", "shutdown", "reboot", "kubectl delete", "delete from")


# 定义 validate_plan_node 相关的处理逻辑，供流程或外部调用复用。
async def validate_plan_node(state: DiagnosisState, llm: LLMExecutor) -> DiagnosisState:
    # 在方案交给人或执行前做安全检查，拦截明显危险的操作建议。
    logger.info("Node validate started")
    # 将 plan_text 的值保存下来，供后续流程判断或组装响应时使用。
    plan_text = str(state.get("recommended_plan", {})).lower()
    # 先用本地规则拦截明显高风险动作，再交给 LLM 做语义层面的方案校验。
    rule_ok = not any(keyword in plan_text for keyword in DANGEROUS_KEYWORDS)
    # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_variables = {
        # 调用 state.get 完成当前步骤需要的业务处理。
        "recommended_plan": state.get("recommended_plan", {}),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "rule_result": "PASS" if rule_ok else "FAIL",
    }
    # 将 prompt 的值保存下来，供后续流程判断或组装响应时使用。
    prompt = format_prompt("validate", prompt_variables)
    # 进入异步上下文管理区域，确保异步资源按约定释放。
    async with asyncio.timeout(llm_timeout_seconds(llm)):
        # 将 llm_result 的值保存下来，供后续流程判断或组装响应时使用。
        llm_result = (
            # 等待异步操作完成，再继续推进当前业务流程。
            await llm.call(
                # 执行当前业务步骤，推动流程继续向下游推进。
                prompt,
                # 将 prompt_name 的值保存下来，供后续流程判断或组装响应时使用。
                prompt_name="validate",
                # 将 prompt_variables 的值保存下来，供后续流程判断或组装响应时使用。
                prompt_variables=prompt_variables,
                # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                metadata={"node_name": "validate"},
            )
        # 调用 strip 完成当前步骤需要的业务处理。
        ).strip().upper()
    # 只有规则校验和 LLM 校验都通过，方案才进入后续人工确认/最终输出。
    validation_result = rule_ok and llm_result.startswith("PASS")
    # 将 attempts 的值保存下来，供后续流程判断或组装响应时使用。
    attempts = state.get("validation_attempts", 0) + 1
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Node validate completed result=%s attempts=%s", validation_result, attempts)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "validation_result": validation_result,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "validation_attempts": attempts,
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "evidence": append_evidence(state, f"Validation result={validation_result}, rule_ok={rule_ok}."),
        # 调用 append_message 完成当前步骤需要的业务处理。
        "messages": append_message(state, "assistant", f"方案校验完成: {'通过' if validation_result else '未通过'}"),
    }


# 定义 validation_result 相关的处理逻辑，供流程或外部调用复用。
def validation_result(state: DiagnosisState) -> str:
    # 根据校验结果决定下一步：通过则继续，首次失败则回到方案生成节点重试。
    if state.get("validation_result", False):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "pass"
    # 根据 state.get("validation_attempts", 0) <= 1 判断当前流程该进入哪个处理分支。
    if state.get("validation_attempts", 0) <= 1:
        # 第一次失败允许回到 generate_plan 重试；第二次仍失败则放行，避免流程无限循环。
        return "fail"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "pass"
