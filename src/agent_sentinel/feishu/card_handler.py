from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import redis.asyncio as redis

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 HumanDecision 组件，集中管理这个模块的状态和行为。
class HumanDecision:
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision: str
    # 将 feedback 的值保存下来，供后续流程判断或组装响应时使用。
    feedback: str = ""


@dataclass(slots=True)
# 定义 DecisionContext 组件，集中管理这个模块的状态和行为。
class DecisionContext:
    # 执行当前业务步骤，推动流程继续向下游推进。
    decision_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_thread_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    workflow_run_id: str
    # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
    chat_id: str | None = None
    # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
    thread_root_message_id: str | None = None
    # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
    status: str = "pending"
    # 将 feedback 的值保存下来，供后续流程判断或组装响应时使用。
    feedback: str = ""


# 定义 HumanDecisionStore 组件，集中管理这个模块的状态和行为。
class HumanDecisionStore:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, redis_url: str | None = None) -> None:
        # 准备人工决策的临时存储，内存用于本进程快速读取，Redis 用于跨请求恢复。
        self.redis_url = redis_url
        # 将 self._redis 的值保存下来，供后续流程判断或组装响应时使用。
        self._redis: redis.Redis | None = None
        # 将 self._contexts 的值保存下来，供后续流程判断或组装响应时使用。
        self._contexts: dict[str, DecisionContext] = {}

    # 定义 open 相关的处理逻辑，供流程或外部调用复用。
    async def open(self) -> None:
        # 在首次需要持久化人工决策时连接 Redis，避免启动阶段就强依赖外部服务。
        if self.redis_url and self._redis is None:
            # 将 self._redis 的值保存下来，供后续流程判断或组装响应时使用。
            self._redis = redis.from_url(self.redis_url, decode_responses=True)

    # 定义 register_pending_decision 相关的处理逻辑，供流程或外部调用复用。
    async def register_pending_decision(self, context: DecisionContext) -> None:
        # 登记一条等待人工点击的决策记录，让后续飞书卡片回调能找回对应工作流。
        await self.open()
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        self._contexts[context.decision_id] = context
        # 根据 self._redis 判断当前流程该进入哪个处理分支。
        if self._redis:
            # 等待异步操作完成，再继续推进当前业务流程。
            await self._redis.setex(
                # 执行当前业务步骤，推动流程继续向下游推进。
                f"decision:{context.decision_id}:context",
                # 执行当前业务步骤，推动流程继续向下游推进。
                600,
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                json.dumps(asdict(context), ensure_ascii=False),
            )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Registered pending human decision decision_id=%s workflow_thread_id=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.decision_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.workflow_thread_id,
        )

    # 定义 get_decision_context 相关的处理逻辑，供流程或外部调用复用。
    async def get_decision_context(self, decision_id: str) -> DecisionContext | None:
        # 根据决策编号查找原始上下文，先查本地内存，找不到再从 Redis 恢复。
        await self.open()
        # 将 context 的值保存下来，供后续流程判断或组装响应时使用。
        context = self._contexts.get(decision_id)
        # 根据 context is not None 判断当前流程该进入哪个处理分支。
        if context is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return context
        # 根据 not self._redis 判断当前流程该进入哪个处理分支。
        if not self._redis:
            return None
        # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
        raw = await self._redis.get(f"decision:{decision_id}:context")
        # 根据 not raw 判断当前流程该进入哪个处理分支。
        if not raw:
            return None
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = json.loads(raw)
        # 将 context 的值保存下来，供后续流程判断或组装响应时使用。
        context = DecisionContext(**data)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        self._contexts[decision_id] = context
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return context

    # 定义 mark_decision_received 相关的处理逻辑，供流程或外部调用复用。
    async def mark_decision_received(self, decision_id: str, decision: str, feedback: str = "") -> DecisionContext | None:
        # 把用户在飞书卡片上的选择写回决策上下文，供等待中的诊断流程继续向下走。
        context = await self.get_decision_context(decision_id)
        # 根据 context is None 判断当前流程该进入哪个处理分支。
        if context is None:
            return None
        # 将 context.status 的值保存下来，供后续流程判断或组装响应时使用。
        context.status = decision
        # 将 context.feedback 的值保存下来，供后续流程判断或组装响应时使用。
        context.feedback = feedback
        # 等待异步操作完成，再继续推进当前业务流程。
        await self.register_pending_decision(context)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Human decision received decision_id=%s decision=%s", decision_id, decision)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return context


# 定义 FeishuCardHandler 组件，集中管理这个模块的状态和行为。
class FeishuCardHandler:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, decision_store: HumanDecisionStore) -> None:
        # 绑定人工决策存储，后续解析卡片点击时会把结果写入这里。
        self.decision_store = decision_store

    # 定义 parse_callback 相关的处理逻辑，供流程或外部调用复用。
    async def parse_callback(self, payload: dict[str, Any]) -> DecisionContext | None:
        # 解析飞书卡片回调，识别用户点击的是诊断确认、案例采用还是反馈学习。
        value = self._extract_value(payload)
        # 将 action 的值保存下来，供后续流程判断或组装响应时使用。
        action = str(value.get("action") or "")
        # 将 allowed_decisions 的值保存下来，供后续流程判断或组装响应时使用。
        allowed_decisions = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "diagnosis_confirm": {"approved", "rejected"},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "case_cache_decision": {"adopt", "reject"},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "case_feedback": {"valid", "invalid"},
        }
        # 根据 action not in allowed_decisions 判断当前流程该进入哪个处理分支。
        if action not in allowed_decisions:
            return None
        # 将 decision_id 的值保存下来，供后续流程判断或组装响应时使用。
        decision_id = str(value.get("decision_id") or "")
        # 将 decision 的值保存下来，供后续流程判断或组装响应时使用。
        decision = str(value.get("decision") or "")
        # 将 feedback 的值保存下来，供后续流程判断或组装响应时使用。
        feedback = str(value.get("feedback") or payload.get("feedback") or "")
        # 根据 not decision_id or decision not in allowed_decisions... 判断当前流程该进入哪个处理分支。
        if not decision_id or decision not in allowed_decisions[action]:
            return None
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await self.decision_store.mark_decision_received(decision_id, decision, feedback)

    # 定义 handle 相关的处理逻辑，供流程或外部调用复用。
    async def handle(self, payload: dict[str, Any]) -> dict[str, str]:
        # 处理飞书推来的卡片事件，并用简单状态告诉调用方本次事件是否被接受。
        context = await self.parse_callback(payload)
        # 根据 context is None 判断当前流程该进入哪个处理分支。
        if context is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"status": "ok"}

    # 定义 _extract_value 相关的处理逻辑，供流程或外部调用复用。
    def _extract_value(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 兼容飞书不同回调格式，把真正的按钮参数统一取出来交给业务判断。
        action = payload.get("action")
        # 根据 isinstance(action, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(action, dict):
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = action.get("value")
            # 根据 isinstance(value, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(value, dict):
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return value
        # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
        value = payload.get("value")
        # 根据 isinstance(value, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(value, dict):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return value
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return payload
