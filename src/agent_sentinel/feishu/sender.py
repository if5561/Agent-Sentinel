from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_sentinel.feishu_app import FeishuBotClient

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 FeishuSender 组件，集中管理这个模块的状态和行为。
class FeishuSender:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, client: FeishuBotClient | None = None) -> None:
        # 保存飞书客户端，发送文本或卡片时统一通过这个出口访问飞书。
        self.client = client

    # 定义 send_message 相关的处理逻辑，供流程或外部调用复用。
    async def send_message(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        text: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
        # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
        mention_open_id: str | None = None,
        # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
        mention_name: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 向飞书会话发送普通文本；如果没有会话或客户端未配置，则记录日志后跳过。
        if not chat_id or not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu text skipped chat_id=%s text=%s", chat_id, text)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.to_thread(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.client.send_text_to_chat,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            text,
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=thread_root_message_id,
            # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
            mention_open_id=mention_open_id,
            # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
            mention_name=mention_name,
        )

    # 定义 send_card 相关的处理逻辑，供流程或外部调用复用。
    async def send_card(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        decision_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_thread_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_run_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        plan: dict[str, Any],
        # 执行当前业务步骤，推动流程继续向下游推进。
        evidence: list[str],
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 把诊断方案做成确认卡片发给人，让人可以在飞书里批准或拒绝。
        if not chat_id or not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu card skipped chat_id=%s decision_id=%s", chat_id, decision_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False
        # 将 card 的值保存下来，供后续流程判断或组装响应时使用。
        card = build_confirmation_card(decision_id, workflow_thread_id, workflow_run_id, plan, evidence)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.to_thread(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.client.send_interactive_card_to_chat,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            card,
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=thread_root_message_id,
        )

    # 定义 send_case_cache_card 相关的处理逻辑，供流程或外部调用复用。
    async def send_case_cache_card(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        decision_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_thread_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_run_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        case: dict[str, Any],
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        candidate_index: int,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 把命中的历史案例做成选择卡片，供用户判断是否直接采用这条经验。
        if not chat_id or not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu cache card skipped chat_id=%s decision_id=%s", chat_id, decision_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False
        # 将 card 的值保存下来，供后续流程判断或组装响应时使用。
        card = build_case_cache_card(
            # 执行当前业务步骤，推动流程继续向下游推进。
            decision_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            workflow_thread_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            workflow_run_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            case,
            # 将 candidate_index 的值保存下来，供后续流程判断或组装响应时使用。
            candidate_index=candidate_index,
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.to_thread(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.client.send_interactive_card_to_chat,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            card,
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=thread_root_message_id,
        )

    # 定义 send_feedback_card 相关的处理逻辑，供流程或外部调用复用。
    async def send_feedback_card(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        decision_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_thread_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        workflow_run_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        plan: dict[str, Any],
        # 执行当前业务步骤，推动流程继续向下游推进。
        evidence: list[str],
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 把本次诊断结果做成反馈卡片，收集“有效/无效”来决定是否沉淀为案例。
        if not chat_id or not self.client or not self.client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu feedback card skipped chat_id=%s decision_id=%s", chat_id, decision_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False
        # 将 card 的值保存下来，供后续流程判断或组装响应时使用。
        card = build_feedback_card(decision_id, workflow_thread_id, workflow_run_id, plan, evidence)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.to_thread(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.client.send_interactive_card_to_chat,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            card,
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=thread_root_message_id,
        )


# 定义 build_confirmation_card 相关的处理逻辑，供流程或外部调用复用。
def build_confirmation_card(
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_thread_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_run_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    plan: dict[str, Any],
    # 执行当前业务步骤，推动流程继续向下游推进。
    evidence: list[str],
# 执行当前业务步骤，推动流程继续向下游推进。
) -> dict[str, Any]:
    # 生成“诊断确认”飞书卡片，卡片里包含方案摘要、证据和批准/拒绝按钮。
    plan_summary = str(plan.get("summary") or plan)[:900]
    # 将 evidence_text 的值保存下来，供后续流程判断或组装响应时使用。
    evidence_text = "\n".join(f"- {item}" for item in evidence[:8]) or "- no evidence"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "config": {"wide_screen_mode": True},
        # 执行当前业务步骤，推动流程继续向下游推进。
        "header": {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "template": "orange",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title": {"tag": "plain_text", "content": "AIOps 告警诊断确认"},
        },
        # 执行当前业务步骤，推动流程继续向下游推进。
        "elements": [
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"tag": "markdown", "content": f"**推荐方案**\n{plan_summary}"},
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"tag": "markdown", "content": f"**证据链**\n{evidence_text}"},
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tag": "action",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "actions": [
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tag": "button",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "text": {"tag": "plain_text", "content": "同意"},
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "type": "primary",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "value": {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "action": "diagnosis_confirm",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision": "approved",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision_id": decision_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_thread_id": workflow_thread_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tag": "button",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "text": {"tag": "plain_text", "content": "拒绝"},
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "type": "danger",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "value": {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "action": "diagnosis_confirm",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision": "rejected",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision_id": decision_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_thread_id": workflow_thread_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                ],
            },
        ],
    }


# 定义 build_case_cache_card 相关的处理逻辑，供流程或外部调用复用。
def build_case_cache_card(
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_thread_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_run_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    case: dict[str, Any],
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    candidate_index: int,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> dict[str, Any]:
    # 生成“历史案例命中”飞书卡片，让用户比较相似案例并选择是否采用。
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    # 将 plan 的值保存下来，供后续流程判断或组装响应时使用。
    plan = case.get("final_plan") or metadata.get("recommended_plan") or metadata.get("final_text") or {}
    # 将 plan_summary 的值保存下来，供后续流程判断或组装响应时使用。
    plan_summary = _format_plan(plan)
    # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
    title = str(case.get("title") or metadata.get("alert_summary") or "历史相似告警")[:512]
    # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
    score = case.get("score", 0)
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 将 score_text 的值保存下来，供后续流程判断或组装响应时使用。
        score_text = f"{float(score):.3f}"
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except (TypeError, ValueError):
        # 将 score_text 的值保存下来，供后续流程判断或组装响应时使用。
        score_text = str(score)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "config": {"wide_screen_mode": True},
        # 执行当前业务步骤，推动流程继续向下游推进。
        "header": {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "template": "blue",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title": {"tag": "plain_text", "content": "命中历史相似案例"},
        },
        # 执行当前业务步骤，推动流程继续向下游推进。
        "elements": [
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tag": "markdown",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "content": (
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    f"**候选案例**：Top {candidate_index + 1}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    f"**相似度**：{score_text}\n"
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    f"**历史告警摘要**：{title}"
                ),
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"tag": "markdown", "content": f"**历史诊断结论**\n{plan_summary}"},
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tag": "action",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "actions": [
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tag": "button",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "text": {"tag": "plain_text", "content": "✅ 采用此方案"},
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "type": "primary",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "value": {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "action": "case_cache_decision",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision": "adopt",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision_id": decision_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "candidate_index": candidate_index,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_thread_id": workflow_thread_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tag": "button",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "text": {"tag": "plain_text", "content": "❌ 不采用"},
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "type": "danger",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "value": {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "action": "case_cache_decision",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision": "reject",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision_id": decision_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "candidate_index": candidate_index,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_thread_id": workflow_thread_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                ],
            },
        ],
    }


# 定义 build_feedback_card 相关的处理逻辑，供流程或外部调用复用。
def build_feedback_card(
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_thread_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_run_id: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    plan: dict[str, Any],
    # 执行当前业务步骤，推动流程继续向下游推进。
    evidence: list[str],
# 执行当前业务步骤，推动流程继续向下游推进。
) -> dict[str, Any]:
    # 生成“诊断反馈学习”飞书卡片，用一次点击收集结果是否值得存入知识库。
    plan_summary = _format_plan(plan)
    # 将 evidence_text 的值保存下来，供后续流程判断或组装响应时使用。
    evidence_text = "\n".join(f"- {item}" for item in evidence[:5]) or "- no evidence"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "config": {"wide_screen_mode": True},
        # 执行当前业务步骤，推动流程继续向下游推进。
        "header": {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "template": "green",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title": {"tag": "plain_text", "content": "诊断反馈学习"},
        },
        # 执行当前业务步骤，推动流程继续向下游推进。
        "elements": [
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"tag": "markdown", "content": "**当前回复是否有效？**"},
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"tag": "markdown", "content": f"**本次方案**\n{plan_summary}"},
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"tag": "markdown", "content": f"**证据摘要**\n{evidence_text}"},
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tag": "action",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "actions": [
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tag": "button",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "text": {"tag": "plain_text", "content": "✅ 有效，存入知识库"},
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "type": "primary",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "value": {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "action": "case_feedback",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision": "valid",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision_id": decision_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_thread_id": workflow_thread_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                    {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "tag": "button",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "text": {"tag": "plain_text", "content": "❌ 无效，不存储"},
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "type": "danger",
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "value": {
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "action": "case_feedback",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision": "invalid",
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "decision_id": decision_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_thread_id": workflow_thread_id,
                            # 执行当前业务步骤，推动流程继续向下游推进。
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                ],
            },
        ],
    }


# 定义 _format_plan 相关的处理逻辑，供流程或外部调用复用。
def _format_plan(plan: Any) -> str:
    # 把方案对象压缩成适合卡片展示的短文本，避免飞书卡片内容过长。
    if isinstance(plan, dict):
        # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
        text = str(plan.get("summary") or plan)
    # 处理前面条件都不满足时的默认分支。
    else:
        # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
        text = str(plan)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return text[:900] if text else "-"
