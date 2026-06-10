from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HumanDecision:
    decision: str
    feedback: str = ""


@dataclass(slots=True)
class DecisionContext:
    decision_id: str
    workflow_thread_id: str
    workflow_run_id: str
    chat_id: str | None = None
    thread_root_message_id: str | None = None
    status: str = "pending"
    feedback: str = ""


class HumanDecisionStore:
    def __init__(self, redis_url: str | None = None) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.redis_url = redis_url
        self._redis: redis.Redis | None = None
        self._contexts: dict[str, DecisionContext] = {}

    async def open(self) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        if self.redis_url and self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)

    async def register_pending_decision(self, context: DecisionContext) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        await self.open()
        self._contexts[context.decision_id] = context
        if self._redis:
            await self._redis.setex(
                f"decision:{context.decision_id}:context",
                600,
                json.dumps(asdict(context), ensure_ascii=False),
            )
        logger.info(
            "Registered pending human decision decision_id=%s workflow_thread_id=%s",
            context.decision_id,
            context.workflow_thread_id,
        )

    async def get_decision_context(self, decision_id: str) -> DecisionContext | None:
        # 方法说明：读取并返回当前流程需要的数据。
        await self.open()
        context = self._contexts.get(decision_id)
        if context is not None:
            return context
        if not self._redis:
            return None
        raw = await self._redis.get(f"decision:{decision_id}:context")
        if not raw:
            return None
        data = json.loads(raw)
        context = DecisionContext(**data)
        self._contexts[decision_id] = context
        return context

    async def mark_decision_received(self, decision_id: str, decision: str, feedback: str = "") -> DecisionContext | None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        context = await self.get_decision_context(decision_id)
        if context is None:
            return None
        context.status = decision
        context.feedback = feedback
        await self.register_pending_decision(context)
        logger.info("Human decision received decision_id=%s decision=%s", decision_id, decision)
        return context


class FeishuCardHandler:
    def __init__(self, decision_store: HumanDecisionStore) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.decision_store = decision_store

    async def parse_callback(self, payload: dict[str, Any]) -> DecisionContext | None:
        # 方法说明：解析输入内容，转换为业务逻辑使用的结构。
        value = self._extract_value(payload)
        action = str(value.get("action") or "")
        allowed_decisions = {
            "diagnosis_confirm": {"approved", "rejected"},
            "case_cache_decision": {"adopt", "reject"},
            "case_feedback": {"valid", "invalid"},
        }
        if action not in allowed_decisions:
            return None
        decision_id = str(value.get("decision_id") or "")
        decision = str(value.get("decision") or "")
        feedback = str(value.get("feedback") or payload.get("feedback") or "")
        if not decision_id or decision not in allowed_decisions[action]:
            return None
        return await self.decision_store.mark_decision_received(decision_id, decision, feedback)

    async def handle(self, payload: dict[str, Any]) -> dict[str, str]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        context = await self.parse_callback(payload)
        if context is None:
            return {"status": "ignored"}
        return {"status": "ok"}

    def _extract_value(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        action = payload.get("action")
        if isinstance(action, dict):
            value = action.get("value")
            if isinstance(value, dict):
                return value
        value = payload.get("value")
        if isinstance(value, dict):
            return value
        return payload
