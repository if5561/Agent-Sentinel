from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_sentinel.feishu_app import FeishuBotClient

logger = logging.getLogger(__name__)


class FeishuSender:
    def __init__(self, client: FeishuBotClient | None = None) -> None:
        # 保存飞书客户端，发送文本或卡片时统一通过这个出口访问飞书。
        self.client = client

    async def send_message(
        self,
        chat_id: str | None,
        text: str,
        *,
        thread_root_message_id: str | None = None,
        mention_open_id: str | None = None,
        mention_name: str | None = None,
    ) -> bool:
        # 向飞书会话发送普通文本；如果没有会话或客户端未配置，则记录日志后跳过。
        if not chat_id or not self.client or not self.client.is_configured():
            logger.info("Feishu text skipped chat_id=%s text=%s", chat_id, text)
            return False
        return await asyncio.to_thread(
            self.client.send_text_to_chat,
            chat_id,
            text,
            thread_root_message_id=thread_root_message_id,
            mention_open_id=mention_open_id,
            mention_name=mention_name,
        )

    async def send_card(
        self,
        chat_id: str | None,
        decision_id: str,
        workflow_thread_id: str,
        workflow_run_id: str,
        plan: dict[str, Any],
        evidence: list[str],
        *,
        thread_root_message_id: str | None = None,
    ) -> bool:
        # 把诊断方案做成确认卡片发给人，让人可以在飞书里批准或拒绝。
        if not chat_id or not self.client or not self.client.is_configured():
            logger.info("Feishu card skipped chat_id=%s decision_id=%s", chat_id, decision_id)
            return False
        card = build_confirmation_card(decision_id, workflow_thread_id, workflow_run_id, plan, evidence)
        return await asyncio.to_thread(
            self.client.send_interactive_card_to_chat,
            chat_id,
            card,
            thread_root_message_id=thread_root_message_id,
        )

    async def send_case_cache_card(
        self,
        chat_id: str | None,
        decision_id: str,
        workflow_thread_id: str,
        workflow_run_id: str,
        case: dict[str, Any],
        *,
        candidate_index: int,
        thread_root_message_id: str | None = None,
    ) -> bool:
        # 把命中的历史案例做成选择卡片，供用户判断是否直接采用这条经验。
        if not chat_id or not self.client or not self.client.is_configured():
            logger.info("Feishu cache card skipped chat_id=%s decision_id=%s", chat_id, decision_id)
            return False
        card = build_case_cache_card(
            decision_id,
            workflow_thread_id,
            workflow_run_id,
            case,
            candidate_index=candidate_index,
        )
        return await asyncio.to_thread(
            self.client.send_interactive_card_to_chat,
            chat_id,
            card,
            thread_root_message_id=thread_root_message_id,
        )

    async def send_feedback_card(
        self,
        chat_id: str | None,
        decision_id: str,
        workflow_thread_id: str,
        workflow_run_id: str,
        plan: dict[str, Any],
        evidence: list[str],
        *,
        thread_root_message_id: str | None = None,
    ) -> bool:
        # 把本次诊断结果做成反馈卡片，收集“有效/无效”来决定是否沉淀为案例。
        if not chat_id or not self.client or not self.client.is_configured():
            logger.info("Feishu feedback card skipped chat_id=%s decision_id=%s", chat_id, decision_id)
            return False
        card = build_feedback_card(decision_id, workflow_thread_id, workflow_run_id, plan, evidence)
        return await asyncio.to_thread(
            self.client.send_interactive_card_to_chat,
            chat_id,
            card,
            thread_root_message_id=thread_root_message_id,
        )


def build_confirmation_card(
    decision_id: str,
    workflow_thread_id: str,
    workflow_run_id: str,
    plan: dict[str, Any],
    evidence: list[str],
) -> dict[str, Any]:
    # 生成“诊断确认”飞书卡片，卡片里包含方案摘要、证据和批准/拒绝按钮。
    plan_summary = str(plan.get("summary") or plan)[:900]
    evidence_text = "\n".join(f"- {item}" for item in evidence[:8]) or "- no evidence"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange",
            "title": {"tag": "plain_text", "content": "AIOps 告警诊断确认"},
        },
        "elements": [
            {"tag": "markdown", "content": f"**推荐方案**\n{plan_summary}"},
            {"tag": "markdown", "content": f"**证据链**\n{evidence_text}"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "同意"},
                        "type": "primary",
                        "value": {
                            "action": "diagnosis_confirm",
                            "decision": "approved",
                            "decision_id": decision_id,
                            "workflow_thread_id": workflow_thread_id,
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "拒绝"},
                        "type": "danger",
                        "value": {
                            "action": "diagnosis_confirm",
                            "decision": "rejected",
                            "decision_id": decision_id,
                            "workflow_thread_id": workflow_thread_id,
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                ],
            },
        ],
    }


def build_case_cache_card(
    decision_id: str,
    workflow_thread_id: str,
    workflow_run_id: str,
    case: dict[str, Any],
    *,
    candidate_index: int,
) -> dict[str, Any]:
    # 生成“历史案例命中”飞书卡片，让用户比较相似案例并选择是否采用。
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    plan = case.get("final_plan") or metadata.get("recommended_plan") or metadata.get("final_text") or {}
    plan_summary = _format_plan(plan)
    title = str(case.get("title") or metadata.get("alert_summary") or "历史相似告警")[:512]
    score = case.get("score", 0)
    try:
        score_text = f"{float(score):.3f}"
    except (TypeError, ValueError):
        score_text = str(score)
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "命中历史相似案例"},
        },
        "elements": [
            {
                "tag": "markdown",
                "content": (
                    f"**候选案例**：Top {candidate_index + 1}\n"
                    f"**相似度**：{score_text}\n"
                    f"**历史告警摘要**：{title}"
                ),
            },
            {"tag": "markdown", "content": f"**历史诊断结论**\n{plan_summary}"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ 采用此方案"},
                        "type": "primary",
                        "value": {
                            "action": "case_cache_decision",
                            "decision": "adopt",
                            "decision_id": decision_id,
                            "candidate_index": candidate_index,
                            "workflow_thread_id": workflow_thread_id,
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "❌ 不采用"},
                        "type": "danger",
                        "value": {
                            "action": "case_cache_decision",
                            "decision": "reject",
                            "decision_id": decision_id,
                            "candidate_index": candidate_index,
                            "workflow_thread_id": workflow_thread_id,
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                ],
            },
        ],
    }


def build_feedback_card(
    decision_id: str,
    workflow_thread_id: str,
    workflow_run_id: str,
    plan: dict[str, Any],
    evidence: list[str],
) -> dict[str, Any]:
    # 生成“诊断反馈学习”飞书卡片，用一次点击收集结果是否值得存入知识库。
    plan_summary = _format_plan(plan)
    evidence_text = "\n".join(f"- {item}" for item in evidence[:5]) or "- no evidence"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green",
            "title": {"tag": "plain_text", "content": "诊断反馈学习"},
        },
        "elements": [
            {"tag": "markdown", "content": "**当前回复是否有效？**"},
            {"tag": "markdown", "content": f"**本次方案**\n{plan_summary}"},
            {"tag": "markdown", "content": f"**证据摘要**\n{evidence_text}"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ 有效，存入知识库"},
                        "type": "primary",
                        "value": {
                            "action": "case_feedback",
                            "decision": "valid",
                            "decision_id": decision_id,
                            "workflow_thread_id": workflow_thread_id,
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "❌ 无效，不存储"},
                        "type": "danger",
                        "value": {
                            "action": "case_feedback",
                            "decision": "invalid",
                            "decision_id": decision_id,
                            "workflow_thread_id": workflow_thread_id,
                            "workflow_run_id": workflow_run_id,
                        },
                    },
                ],
            },
        ],
    }


def _format_plan(plan: Any) -> str:
    # 把方案对象压缩成适合卡片展示的短文本，避免飞书卡片内容过长。
    if isinstance(plan, dict):
        text = str(plan.get("summary") or plan)
    else:
        text = str(plan)
    return text[:900] if text else "-"
