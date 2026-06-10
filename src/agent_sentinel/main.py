from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response

from agent_sentinel.alerts import FeishuWebhookNotifier, RealtimeAlertService
from agent_sentinel.config import Settings, get_settings
from agent_sentinel.feishu_app import FeishuBotClient, extract_text_from_message_content
from agent_sentinel.feishu.card_handler import FeishuCardHandler, HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.feishu_longconn import FeishuLongConnectionBot
from agent_sentinel.feishu_poller import FeishuMessagePoller
from agent_sentinel.graph.state import DiagnosisState
from agent_sentinel.graph.workflow import DiagnosisWorkflow, build_llm_executor
from agent_sentinel.interactive_topic import InteractiveTopicWorkflow
from agent_sentinel.interactive_topic.topic_sender import InteractiveTopicSender
from agent_sentinel.monitoring import CONTENT_TYPE_LATEST, configure_monitoring, monitor, trace_id_from_parts
from agent_sentinel.rag.factory import build_retriever
from agent_sentinel.rag.history_cases import build_history_case_store
from agent_sentinel.schemas import (
    # 执行当前业务步骤，推动流程继续向下游推进。
    AlertAnalyzeRequest,
    # 执行当前业务步骤，推动流程继续向下游推进。
    AlertAnalyzeResponse,
    # 执行当前业务步骤，推动流程继续向下游推进。
    AlertRecordResponse,
    # 执行当前业务步骤，推动流程继续向下游推进。
    AlertReportRequest,
    # 执行当前业务步骤，推动流程继续向下游推进。
    AlertReportResponse,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ChatRequest,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ChatResponse,
    # 执行当前业务步骤，推动流程继续向下游推进。
    FeishuEventEnvelope,
    # 执行当前业务步骤，推动流程继续向下游推进。
    HealthResponse,
)
from agent_sentinel.service import SingleTurnChatService
from agent_sentinel.utils.logger import configure_logger


# 定义 configure_logging 相关的处理逻辑，供流程或外部调用复用。
def configure_logging(log_level: str) -> None:
    # 设置整套服务的日志级别，让后续每个模块都按同一规则输出运行记录。
    configure_logger(log_level)


# 定义 build_app 相关的处理逻辑，供流程或外部调用复用。
def build_app(settings: Settings | None = None) -> FastAPI:
    # 创建 Web 应用，并把配置、监控、飞书、诊断工作流等核心组件接到一起。
    settings = settings or get_settings()
    # 调用 configure_logging 完成当前步骤需要的业务处理。
    configure_logging(settings.log_level)
    # Prometheus 指标开关在应用构建时统一生效，避免各业务节点重复读取环境变量。
    configure_monitoring(enabled=settings.metrics_enabled)
    # 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
    logger = logging.getLogger(__name__)
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info(
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "App build started app=%s env=%s interactive_topic=%s rag_provider=%s feishu_longconn=%s feishu_polling=%s metrics_enabled=%s",
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.app_name,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.app_env,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.interactive_topic_enabled,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.rag_provider,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.feishu_long_connection_enabled,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.feishu_message_polling_enabled,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings.metrics_enabled,
    )

    # 将 notifier 的值保存下来，供后续流程判断或组装响应时使用。
    notifier = FeishuWebhookNotifier(
        # 将 webhook_url 的值保存下来，供后续流程判断或组装响应时使用。
        webhook_url=settings.feishu_webhook_url,
        # 将 secret 的值保存下来，供后续流程判断或组装响应时使用。
        secret=settings.feishu_secret,
        # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
        enabled=settings.feishu_alert_enabled,
    )
    # 将 alert_service 的值保存下来，供后续流程判断或组装响应时使用。
    alert_service = RealtimeAlertService(
        # 将 notifier 的值保存下来，供后续流程判断或组装响应时使用。
        notifier=notifier,
        # 将 app_env 的值保存下来，供后续流程判断或组装响应时使用。
        app_env=settings.app_env,
        # 将 title_prefix 的值保存下来，供后续流程判断或组装响应时使用。
        title_prefix=settings.feishu_alert_title_prefix,
        # 将 dedup_window_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        dedup_window_seconds=settings.alert_dedup_window_seconds,
        # 将 store_limit 的值保存下来，供后续流程判断或组装响应时使用。
        store_limit=settings.alert_store_limit,
    )

    # 将 chat_service 的值保存下来，供后续流程判断或组装响应时使用。
    chat_service: SingleTurnChatService | None = None
    # 将 feishu_bot_client 的值保存下来，供后续流程判断或组装响应时使用。
    feishu_bot_client = FeishuBotClient(settings)
    # 将 feishu_sender 的值保存下来，供后续流程判断或组装响应时使用。
    feishu_sender = FeishuSender(feishu_bot_client)
    # 将 decision_store 的值保存下来，供后续流程判断或组装响应时使用。
    decision_store = HumanDecisionStore(settings.redis_url)
    # 将 card_handler 的值保存下来，供后续流程判断或组装响应时使用。
    card_handler = FeishuCardHandler(decision_store)
    # 将 aiops_workflow 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_workflow: DiagnosisWorkflow | None = None
    # 将 interactive_topic_workflow 的值保存下来，供后续流程判断或组装响应时使用。
    interactive_topic_workflow: InteractiveTopicWorkflow | None = None
    # 将 longconn_bot 的值保存下来，供后续流程判断或组装响应时使用。
    longconn_bot: FeishuLongConnectionBot | None = None
    # 将 poller_bot 的值保存下来，供后续流程判断或组装响应时使用。
    poller_bot: FeishuMessagePoller | None = None

    # 将 app 的值保存下来，供后续流程判断或组装响应时使用。
    app = FastAPI(title=settings.app_name)

    # 定义 get_chat_service 相关的处理逻辑，供流程或外部调用复用。
    def get_chat_service() -> SingleTurnChatService:
        # 按需创建普通聊天服务；只有真正调用聊天接口时才初始化模型对象。
        nonlocal chat_service
        # 单轮聊天服务只在首次请求时初始化，避免启动阶段就创建不一定会用到的模型客户端。
        if chat_service is None:
            # 将 chat_service 的值保存下来，供后续流程判断或组装响应时使用。
            chat_service = SingleTurnChatService(settings)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return chat_service

    # 定义 get_aiops_workflow 相关的处理逻辑，供流程或外部调用复用。
    def get_aiops_workflow() -> DiagnosisWorkflow:
        # 按需创建 AIOps 诊断工作流；它负责把告警理解、检索、生成方案等步骤串起来。
        nonlocal aiops_workflow
        # 诊断工作流依赖 LLM、RAG、飞书发送器等对象，采用懒加载可以降低健康检查和简单接口的启动成本。
        if aiops_workflow is None:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Initializing diagnosis workflow rag_provider=%s mock_llm=%s", settings.rag_provider, settings.aiops_mock_llm_enabled)
            # 将 aiops_workflow 的值保存下来，供后续流程判断或组装响应时使用。
            aiops_workflow = DiagnosisWorkflow(
                # 将 settings 的值保存下来，供后续流程判断或组装响应时使用。
                settings=settings,
                # 将 llm 的值保存下来，供后续流程判断或组装响应时使用。
                llm=build_llm_executor(settings),
                # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
                sender=feishu_sender,
                # 将 decision_store 的值保存下来，供后续流程判断或组装响应时使用。
                decision_store=decision_store,
            )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return aiops_workflow

    # 定义 get_interactive_topic_workflow 相关的处理逻辑，供流程或外部调用复用。
    def get_interactive_topic_workflow() -> InteractiveTopicWorkflow:
        # 按需创建飞书单卡片交互流程；它负责在同一张卡片里持续更新诊断进度。
        nonlocal interactive_topic_workflow
        # 交互式话题工作流有独立的状态存储和卡片更新逻辑，仅在启用并首次触发时创建。
        if interactive_topic_workflow is None:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Initializing interactive topic workflow wait_seconds=%s rag_provider=%s static_top_k=%s history_top_k=%s final_top_k=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                settings.interactive_topic_wait_seconds,
                # 执行当前业务步骤，推动流程继续向下游推进。
                settings.rag_provider,
                # 执行当前业务步骤，推动流程继续向下游推进。
                settings.rag_static_top_k,
                # 执行当前业务步骤，推动流程继续向下游推进。
                settings.rag_message_top_k,
                # 执行当前业务步骤，推动流程继续向下游推进。
                settings.rag_final_top_k,
            )
            # 将 interactive_topic_workflow 的值保存下来，供后续流程判断或组装响应时使用。
            interactive_topic_workflow = InteractiveTopicWorkflow(
                # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
                sender=InteractiveTopicSender(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    feishu_bot_client,
                    # 将 wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
                    wait_seconds=settings.interactive_topic_wait_seconds,
                ),
                # 将 wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
                wait_seconds=settings.interactive_topic_wait_seconds,
                # 将 retriever 的值保存下来，供后续流程判断或组装响应时使用。
                retriever=build_retriever(settings),
                # 将 case_store 的值保存下来，供后续流程判断或组装响应时使用。
                case_store=build_history_case_store(settings),
                # 将 llm 的值保存下来，供后续流程判断或组装响应时使用。
                llm=build_llm_executor(settings),
            )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return interactive_topic_workflow

    # 定义 verify_alert_token 相关的处理逻辑，供流程或外部调用复用。
    def verify_alert_token(provided_token: str | None) -> None:
        # 检查外部告警接口的访问令牌，防止未授权系统随意触发诊断。
        expected_token = settings.alert_api_token
        # 未配置 token 时保持开发环境易用；一旦配置，则所有告警入口都必须携带正确凭证。
        if expected_token and provided_token != expected_token:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=401, detail="Invalid alert token.")

    # 定义 verify_feishu_event_token 相关的处理逻辑，供流程或外部调用复用。
    def verify_feishu_event_token(payload: FeishuEventEnvelope) -> None:
        # 检查飞书推送事件中的校验令牌，确认消息确实来自可信飞书应用。
        expected_token = settings.feishu_event_verification_token
        # 根据 not expected_token 判断当前流程该进入哪个处理分支。
        if not expected_token:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 将 provided_token 的值保存下来，供后续流程判断或组装响应时使用。
        provided_token = None
        # 根据 payload.header and payload.header.token 判断当前流程该进入哪个处理分支。
        if payload.header and payload.header.token:
            # 将 provided_token 的值保存下来，供后续流程判断或组装响应时使用。
            provided_token = payload.header.token
        # 根据 payload.token 判断当前流程该进入哪个处理分支。
        elif payload.token:
            # 将 provided_token 的值保存下来，供后续流程判断或组装响应时使用。
            provided_token = payload.token

        # 根据 provided_token != expected_token 判断当前流程该进入哪个处理分支。
        if provided_token != expected_token:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=401, detail="Invalid Feishu event token.")

    # 定义 report_exception 相关的处理逻辑，供流程或外部调用复用。
    def report_exception(scene: str, exc: Exception) -> None:
        # 把接口内部异常转成飞书告警，便于运维人员第一时间知道服务自身出错。
        try:
            # 调用 alert_service.report_exception 完成当前步骤需要的业务处理。
            alert_service.report_exception(scene, exc)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as notify_exc:  # pragma: no cover
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to send Feishu alert for %s: %s", scene, notify_exc)

    # 定义 build_diagnosis_state 相关的处理逻辑，供流程或外部调用复用。
    def build_diagnosis_state(
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        level: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        summary: str,
        # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
        details: str | None = None,
        # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
        raw_text: str | None = None,
        # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
        trigger_type: str = "unknown",
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags: list[str] | None = None,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
        # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
        mention_open_id: str | None = None,
        # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
        mention_name: str | None = None,
        # 将 workflow_thread_id 的值保存下来，供后续流程判断或组装响应时使用。
        workflow_thread_id: str | None = None,
        # 将 workflow_run_id 的值保存下来，供后续流程判断或组装响应时使用。
        workflow_run_id: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> DiagnosisState:
        # 把不同来源的告警统一整理成诊断状态，后面的每个节点都围绕这份状态继续补充信息。
        trace_id = trace_id_from_parts(chat_id, thread_root_message_id)
        # 所有诊断入口最终都归一化成同一份 LangGraph state，后续节点只依赖这个状态契约。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "raw_alert": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "source": source,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "level": level,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "summary": summary,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "details": details,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "raw_text": raw_text,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "trigger_type": trigger_type,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tags": tags,
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            "chat_id": chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "thread_root_message_id": thread_root_message_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "mention_open_id": mention_open_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "mention_name": mention_name,
            # 调用 str 完成当前步骤需要的业务处理。
            "workflow_thread_id": workflow_thread_id or str(uuid.uuid4()),
            # 调用 str 完成当前步骤需要的业务处理。
            "workflow_run_id": workflow_run_id or str(uuid.uuid4()),
            # trace_id 用于日志和可观测性关联，不作为 Prometheus label，避免指标高基数。
            "trace_id": trace_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "messages": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "evidence": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "retrieved_docs": [],
            # 执行当前业务步骤，推动流程继续向下游推进。
            "live_data": {},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "recommended_plan": {},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "validation_result": False,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "need_human": True,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "human_card_sent": False,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "human_feedback": None,
        }

    # 定义 run_workflow_and_send 相关的处理逻辑，供流程或外部调用复用。
    async def run_workflow_and_send(
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        level: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        summary: str,
        # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
        details: str | None = None,
        # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
        raw_text: str | None = None,
        # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
        trigger_type: str = "unknown",
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags: list[str] | None = None,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
        # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
        mention_open_id: str | None = None,
        # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
        mention_name: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> tuple[str, bool]:
        # 执行完整诊断链路，并在需要时把最终结果发送回飞书会话。
        if not settings.alert_analysis_enabled:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Diagnosis workflow skipped because alert analysis is disabled chat_id=%s source=%s", chat_id, source)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "Alert analysis is disabled.", False

        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 这里先构造初始状态再进入工作流，便于统一记录 trace、线程 ID 和告警原文。
        initial_state = build_diagnosis_state(
            # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
            chat_id=chat_id,
            # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
            source=source,
            # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
            level=level,
            # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
            summary=summary,
            # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
            details=details,
            # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
            raw_text=raw_text,
            # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
            trigger_type=trigger_type,
            # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
            tags=tags,
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id=thread_root_message_id,
            # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
            mention_open_id=mention_open_id,
            # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
            mention_name=mention_name,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Diagnosis workflow start trace_id=%s source=%s level=%s trigger_type=%s chat_id=%s thread_root_message_id=%s workflow_thread_id=%s workflow_run_id=%s summary_chars=%s raw_chars=%s tags=%s",
            # 调用 initial_state.get 完成当前步骤需要的业务处理。
            initial_state.get("trace_id", ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            source,
            # 执行当前业务步骤，推动流程继续向下游推进。
            level,
            # 执行当前业务步骤，推动流程继续向下游推进。
            trigger_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            thread_root_message_id,
            # 调用 initial_state.get 完成当前步骤需要的业务处理。
            initial_state.get("workflow_thread_id", ""),
            # 调用 initial_state.get 完成当前步骤需要的业务处理。
            initial_state.get("workflow_run_id", ""),
            # 调用 len 完成当前步骤需要的业务处理。
            len(summary or ""),
            # 调用 len 完成当前步骤需要的业务处理。
            len(raw_text or details or ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            tags or [],
        )
        # 将 final_state 的值保存下来，供后续流程判断或组装响应时使用。
        final_state = await get_aiops_workflow().run_streaming(initial_state)
        # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
        final_text = str(final_state.get("final_text", ""))
        # 是否真实发送飞书消息只取决于 chat_id 和机器人配置，工作流本身仍然可以在无 chat_id 场景运行。
        sent = bool(chat_id and feishu_bot_client.is_configured())
        # 将 elapsed_ms 的值保存下来，供后续流程判断或组装响应时使用。
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Diagnosis workflow completed trace_id=%s source=%s trigger_type=%s chat_id=%s workflow_thread_id=%s validation_result=%s human_decision=%s evidence=%s final_text_chars=%s sent=%s elapsed_ms=%s",
            # 调用 final_state.get 完成当前步骤需要的业务处理。
            final_state.get("trace_id", initial_state.get("trace_id", "")),
            # 执行当前业务步骤，推动流程继续向下游推进。
            source,
            # 执行当前业务步骤，推动流程继续向下游推进。
            trigger_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 调用 final_state.get 完成当前步骤需要的业务处理。
            final_state.get("workflow_thread_id", ""),
            # 调用 final_state.get 完成当前步骤需要的业务处理。
            final_state.get("validation_result", False),
            # 调用 final_state.get 完成当前步骤需要的业务处理。
            final_state.get("human_decision", "unknown"),
            # 调用 len 完成当前步骤需要的业务处理。
            len(final_state.get("evidence", [])),
            # 调用 len 完成当前步骤需要的业务处理。
            len(final_text),
            # 执行当前业务步骤，推动流程继续向下游推进。
            sent,
            # 执行当前业务步骤，推动流程继续向下游推进。
            elapsed_ms,
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return final_text, sent

    # 定义 resume_workflow_from_callback 相关的处理逻辑，供流程或外部调用复用。
    async def resume_workflow_from_callback(payload: dict[str, Any], transport: str) -> dict[str, str]:
        # 处理飞书卡片按钮回调，把“人工确认/拒绝”的结果送回暂停中的工作流。
        logger.info("Card callback received transport=%s payload_keys=%s", transport, sorted(payload.keys()))
        # 根据 settings.interactive_topic_enabled 判断当前流程该进入哪个处理分支。
        if settings.interactive_topic_enabled:
            # 新版单卡片流程优先消费回调；如果不是它的 payload，再回退到旧版人审卡片解析。
            interactive_result = await get_interactive_topic_workflow().handle_card_callback(payload, source=transport)
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Interactive topic callback result transport=%s status=%s", transport, interactive_result.get("status"))
            # 根据 interactive_result.get("status") != "ignored" 判断当前流程该进入哪个处理分支。
            if interactive_result.get("status") != "ignored":
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return interactive_result

        # 将 context 的值保存下来，供后续流程判断或组装响应时使用。
        context = await card_handler.parse_callback(payload)
        # 根据 context is None 判断当前流程该进入哪个处理分支。
        if context is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ignored"}
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Received Feishu card callback transport=%s decision_id=%s workflow_thread_id=%s status=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            transport,
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.decision_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.workflow_thread_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.status,
        )
        # 根据 context.feedback 判断当前流程该进入哪个处理分支。
        if context.feedback:
            # 等待异步操作完成，再继续推进当前业务流程。
            await get_aiops_workflow().update_state(
                # 执行当前业务步骤，推动流程继续向下游推进。
                context.workflow_thread_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                {"human_feedback": context.feedback},
            )
        # 等待异步操作完成，再继续推进当前业务流程。
        await get_aiops_workflow().resume(
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.workflow_thread_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            {"decision": context.status, "feedback": context.feedback},
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Resumed workflow from Feishu callback transport=%s decision_id=%s workflow_thread_id=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            transport,
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.decision_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            context.workflow_thread_id,
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"status": "ok"}

    # 定义 analyze_and_send_to_chat 相关的处理逻辑，供流程或外部调用复用。
    def analyze_and_send_to_chat(
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        level: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        summary: str,
        # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
        details: str | None = None,
        # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
        raw_text: str | None = None,
        # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
        trigger_type: str = "unknown",
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags: list[str] | None = None,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
        # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
        mention_open_id: str | None = None,
        # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
        mention_name: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> tuple[str, bool]:
        # 把同步入口包装成统一诊断调用；普通流程直接跑工作流，单卡片模式走交互式话题。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Analyze request routed chat_id=%s source=%s trigger_type=%s interactive_topic=%s root_message_id=%s query_chars=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            source,
            # 执行当前业务步骤，推动流程继续向下游推进。
            trigger_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            settings.interactive_topic_enabled,
            # 执行当前业务步骤，推动流程继续向下游推进。
            thread_root_message_id,
            # 调用 len 完成当前步骤需要的业务处理。
            len(raw_text or details or summary or ""),
        )
        # 根据 settings.interactive_topic_enabled 判断当前流程该进入哪个处理分支。
        if settings.interactive_topic_enabled:
            # 单卡片模式依赖原消息话题 ID 来持续更新同一张卡片，缺失时无法保证不刷屏。
            root_message_id = thread_root_message_id
            # 根据 not root_message_id 判断当前流程该进入哪个处理分支。
            if not root_message_id:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Interactive topic skipped because root message id is missing chat_id=%s", chat_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return "Interactive topic skipped: root message id is missing.", False
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return get_interactive_topic_workflow().start_sync(
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                raw_text or details or summary,
            )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return asyncio.run(
            # 调用 run_workflow_and_send 完成当前步骤需要的业务处理。
            run_workflow_and_send(
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=chat_id,
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source=source,
                # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
                level=level,
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary=summary,
                # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
                details=details,
                # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
                raw_text=raw_text,
                # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
                trigger_type=trigger_type,
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=tags,
                # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                thread_root_message_id=thread_root_message_id,
                # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
                mention_open_id=mention_open_id,
                # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
                mention_name=mention_name,
            )
        )

    @app.on_event("startup")
    # 定义 startup_event 相关的处理逻辑，供流程或外部调用复用。
    def startup_event() -> None:
        # 服务启动时准备飞书长连接和轮询组件，让机器人能主动接收群消息。
        nonlocal longconn_bot, poller_bot
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Application startup begin")
        # 长连接和轮询都封装为可配置组件，组件内部会根据开关决定是否真正启动。
        if longconn_bot is None:
            # 将 longconn_bot 的值保存下来，供后续流程判断或组装响应时使用。
            longconn_bot = FeishuLongConnectionBot(
                # 将 settings 的值保存下来，供后续流程判断或组装响应时使用。
                settings=settings,
                # 将 analyze_callback 的值保存下来，供后续流程判断或组装响应时使用。
                analyze_callback=analyze_and_send_to_chat,
                # 将 card_action_callback 的值保存下来，供后续流程判断或组装响应时使用。
                card_action_callback=lambda payload: resume_workflow_from_callback(payload, transport="longconn"),
            )
        # 调用 longconn_bot.start 完成当前步骤需要的业务处理。
        longconn_bot.start()

        # 根据 poller_bot is None 判断当前流程该进入哪个处理分支。
        if poller_bot is None:
            # 将 poller_bot 的值保存下来，供后续流程判断或组装响应时使用。
            poller_bot = FeishuMessagePoller(
                # 将 settings 的值保存下来，供后续流程判断或组装响应时使用。
                settings=settings,
                # 将 feishu_bot_client 的值保存下来，供后续流程判断或组装响应时使用。
                feishu_bot_client=feishu_bot_client,
                # 将 analyze_callback 的值保存下来，供后续流程判断或组装响应时使用。
                analyze_callback=analyze_and_send_to_chat,
            )
        # 调用 poller_bot.start 完成当前步骤需要的业务处理。
        poller_bot.start()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Application startup completed")

    @app.on_event("shutdown")
    # 定义 shutdown_event 相关的处理逻辑，供流程或外部调用复用。
    async def shutdown_event() -> None:
        # 服务关闭前收尾观测数据，尽量把模型调用链路和诊断 trace 写完整。
        logger.info("Application shutdown begin")
        # 将 observability_clients 的值保存下来，供后续流程判断或组装响应时使用。
        observability_clients: list[LLMExecutor] = []
        # 关闭前主动 flush Langfuse 等可观测客户端，减少进程退出时丢失 trace 的概率。
        if aiops_workflow is not None:
            # 把当前结果追加到集合中，逐步构建最终输出。
            observability_clients.append(aiops_workflow.llm)
        # 根据 interactive_topic_workflow is not None and interacti... 判断当前流程该进入哪个处理分支。
        if interactive_topic_workflow is not None and interactive_topic_workflow.llm is not None:
            # 把当前结果追加到集合中，逐步构建最终输出。
            observability_clients.append(interactive_topic_workflow.llm)
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for llm in observability_clients:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 等待异步操作完成，再继续推进当前业务流程。
                await llm.flush_observability()
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Failed to flush observability client", exc_info=True)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Application shutdown completed")

    @app.get("/health", response_model=HealthResponse)
    # 定义 health 相关的处理逻辑，供流程或外部调用复用。
    def health() -> HealthResponse:
        # 提供健康检查接口，外部系统可用它判断服务是否还活着。
        return HealthResponse(status="ok", app=settings.app_name)

    @app.post("/chat/once", response_model=ChatResponse)
    # 定义 chat_once 相关的处理逻辑，供流程或外部调用复用。
    def chat_once(payload: ChatRequest) -> ChatResponse:
        # 提供一次性聊天接口，用于验证模型配置或做简单问答，不进入 AIOps 诊断流程。
        try:
            # 将 answer 的值保存下来，供后续流程判断或组装响应时使用。
            answer = get_chat_service().reply_once(payload.message)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return ChatResponse(answer=answer, model=settings.openai_model)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Single-turn chat failed")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("chat_once_error")
            # 调用 report_exception 完成当前步骤需要的业务处理。
            report_exception("/chat/once", exc)
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Chat request failed.") from exc

    @app.post("/alerts/test")
    # 定义 test_alert 相关的处理逻辑，供流程或外部调用复用。
    def test_alert() -> dict[str, str]:
        # 手动触发一条测试告警，用来验证飞书告警通道是否配置正确。
        try:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            dispatched, deduplicated = alert_service.report(
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source="agent-sentinel",
                # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
                level="INFO",
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary="Manual alert test",
                # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
                details=f"Application {settings.app_name} triggered a manual alert test.",
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=["manual-test"],
            )
            # 根据 deduplicated 判断当前流程该进入哪个处理分支。
            if deduplicated:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "deduplicated"}
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "sent" if dispatched else "skipped"}
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to send test alert")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("alert_test_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Failed to send alert.") from exc

    @app.post("/alerts/report", response_model=AlertReportResponse)
    # 定义 report_alert 相关的处理逻辑，供流程或外部调用复用。
    def report_alert(
        # 执行当前业务步骤，推动流程继续向下游推进。
        payload: AlertReportRequest,
        # 将 x_alert_token 的值保存下来，供后续流程判断或组装响应时使用。
        x_alert_token: str | None = Header(default=None),
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> AlertReportResponse:
        # 接收外部系统上报的告警，只负责记录/通知，不启动完整 AI 诊断。
        try:
            # 调用 verify_alert_token 完成当前步骤需要的业务处理。
            verify_alert_token(x_alert_token)
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            dispatched, deduplicated = alert_service.report(
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source=payload.source,
                # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
                level=payload.level,
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary=payload.summary,
                # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
                details=payload.details,
                # 将 dedupe_key 的值保存下来，供后续流程判断或组装响应时使用。
                dedupe_key=payload.dedupe_key,
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=payload.tags,
            )
            # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
            status = "deduplicated" if deduplicated else "sent" if dispatched else "skipped"
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return AlertReportResponse(
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status=status,
                # 将 dispatched 的值保存下来，供后续流程判断或组装响应时使用。
                dispatched=dispatched,
                # 将 deduplicated 的值保存下来，供后续流程判断或组装响应时使用。
                deduplicated=deduplicated,
            )
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except HTTPException:
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to report alert")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("alert_report_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Failed to report alert.") from exc

    @app.get("/alerts/recent", response_model=list[AlertRecordResponse])
    # 定义 recent_alerts 相关的处理逻辑，供流程或外部调用复用。
    def recent_alerts(
        # 将 limit 的值保存下来，供后续流程判断或组装响应时使用。
        limit: int = Query(default=20, ge=1, le=100),
        # 将 x_alert_token 的值保存下来，供后续流程判断或组装响应时使用。
        x_alert_token: str | None = Header(default=None),
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[AlertRecordResponse]:
        # 查询最近的告警记录，方便排查“告警是否已发送、是否被去重”。
        verify_alert_token(x_alert_token)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [AlertRecordResponse(**item) for item in alert_service.recent_alerts(limit=limit)]

    @app.post("/alerts/analyze", response_model=AlertAnalyzeResponse)
    # 定义 analyze_alert 相关的处理逻辑，供流程或外部调用复用。
    async def analyze_alert(
        # 执行当前业务步骤，推动流程继续向下游推进。
        payload: AlertAnalyzeRequest,
        # 将 x_alert_token 的值保存下来，供后续流程判断或组装响应时使用。
        x_alert_token: str | None = Header(default=None),
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> AlertAnalyzeResponse:
        # 接收外部告警并启动 AI 诊断，最后返回诊断文本以及是否已发送飞书。
        try:
            # 调用 verify_alert_token 完成当前步骤需要的业务处理。
            verify_alert_token(x_alert_token)
            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id = payload.thread_root_message_id or payload.message_id
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            analysis, sent = await run_workflow_and_send(
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=payload.chat_id,
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source=payload.source,
                # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
                level=payload.level,
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary=payload.summary,
                # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
                details=payload.details,
                # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
                raw_text=payload.raw_text,
                # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
                trigger_type=payload.trigger_type,
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=payload.tags,
                # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                thread_root_message_id=thread_root_message_id,
                # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
                mention_open_id=payload.mention_open_id,
                # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
                mention_name=payload.mention_name,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return AlertAnalyzeResponse(
                # 将 status 的值保存下来，供后续流程判断或组装响应时使用。
                status="sent" if sent else "generated",
                # 将 analysis 的值保存下来，供后续流程判断或组装响应时使用。
                analysis=analysis,
                # 将 sent_to_feishu 的值保存下来，供后续流程判断或组装响应时使用。
                sent_to_feishu=sent,
            )
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except HTTPException:
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to analyze alert")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("alert_analyze_error")
            # 调用 report_exception 完成当前步骤需要的业务处理。
            report_exception("/alerts/analyze", exc)
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Failed to analyze alert.") from exc

    @app.post("/aiops/diagnose")
    # 定义 aiops_diagnose 相关的处理逻辑，供流程或外部调用复用。
    async def aiops_diagnose(
        # 执行当前业务步骤，推动流程继续向下游推进。
        payload: AlertAnalyzeRequest,
        # 将 x_alert_token 的值保存下来，供后续流程判断或组装响应时使用。
        x_alert_token: str | None = Header(default=None),
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> dict[str, Any]:
        # 直接运行 LangGraph 诊断并返回完整状态，适合调试每个诊断节点的中间结果。
        try:
            # 调用 verify_alert_token 完成当前步骤需要的业务处理。
            verify_alert_token(x_alert_token)
            # 将 initial_state 的值保存下来，供后续流程判断或组装响应时使用。
            initial_state = build_diagnosis_state(
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=payload.chat_id,
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source=payload.source,
                # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
                level=payload.level,
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary=payload.summary,
                # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
                details=payload.details,
                # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
                raw_text=payload.raw_text,
                # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
                trigger_type=payload.trigger_type,
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=payload.tags,
                # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                thread_root_message_id=payload.thread_root_message_id or payload.message_id,
                # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
                mention_open_id=payload.mention_open_id,
                # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
                mention_name=payload.mention_name,
            )
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Starting direct LangGraph diagnosis source=%s level=%s trigger_type=%s chat_id=%s thread_root_message_id=%s workflow_thread_id=%s summary=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.source,
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.level,
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.trigger_type,
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.thread_root_message_id or payload.message_id,
                # 调用 initial_state.get 完成当前步骤需要的业务处理。
                initial_state.get("workflow_thread_id", ""),
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.summary,
            )
            # 将 final_state 的值保存下来，供后续流程判断或组装响应时使用。
            final_state = await get_aiops_workflow().run_streaming(initial_state)
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Completed direct LangGraph diagnosis source=%s trigger_type=%s chat_id=%s validation_result=%s human_decision=%s evidence=%s final_text_chars=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.source,
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.trigger_type,
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.chat_id,
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                final_state.get("validation_result", False),
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                final_state.get("human_decision", "unknown"),
                # 调用 len 完成当前步骤需要的业务处理。
                len(final_state.get("evidence", [])),
                # 调用 len 完成当前步骤需要的业务处理。
                len(str(final_state.get("final_text", ""))),
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "status": "ok",
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                "summary": final_state.get("alert_summary", ""),
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                "recommended_plan": final_state.get("recommended_plan", {}),
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                "evidence": final_state.get("evidence", []),
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                "validation_result": final_state.get("validation_result", False),
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                "human_decision": final_state.get("human_decision", "unknown"),
                # 调用 final_state.get 完成当前步骤需要的业务处理。
                "final_text": final_state.get("final_text", ""),
            }
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except HTTPException:
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("AIOps diagnosis failed")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("aiops_diagnose_error")
            # 调用 report_exception 完成当前步骤需要的业务处理。
            report_exception("/aiops/diagnose", exc)
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="AIOps diagnosis failed.") from exc

    @app.post("/feishu/card/callback")
    # 定义 feishu_card_callback 相关的处理逻辑，供流程或外部调用复用。
    async def feishu_card_callback(request: Request) -> dict[str, str]:
        # 接收飞书卡片按钮事件，例如“批准方案”“拒绝方案”等人工操作。
        try:
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = await request.json()
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("HTTP Feishu card callback accepted payload_keys=%s", sorted(payload.keys()))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return await resume_workflow_from_callback(payload, transport="http")
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to process Feishu card callback")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Failed to process card callback.") from exc

    @app.post("/webhook/card")
    # 定义 webhook_card_callback 相关的处理逻辑，供流程或外部调用复用。
    async def webhook_card_callback(request: Request) -> dict[str, str]:
        # 提供兼容 webhook 的卡片回调入口，最终仍复用同一套恢复工作流逻辑。
        try:
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = await request.json()
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("HTTP webhook card callback accepted payload_keys=%s", sorted(payload.keys()))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return await resume_workflow_from_callback(payload, transport="http")
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to process webhook card callback")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Failed to process card callback.") from exc

    @app.post("/feishu/events")
    # 定义 feishu_events 相关的处理逻辑，供流程或外部调用复用。
    async def feishu_events(payload: FeishuEventEnvelope) -> dict[str, object]:
        # 接收飞书普通消息事件，把用户在群里的 @ 消息转换成一次诊断请求。
        try:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "HTTP Feishu event received type=%s challenge=%s event_type=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.type,
                # 调用 bool 完成当前步骤需要的业务处理。
                bool(payload.challenge),
                # 执行当前业务步骤，推动流程继续向下游推进。
                payload.header.event_type if payload.header else None,
            )
            # 根据 payload.challenge 判断当前流程该进入哪个处理分支。
            if payload.challenge:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"challenge": payload.challenge}
            # 根据 payload.type == "url_verification" 判断当前流程该进入哪个处理分支。
            if payload.type == "url_verification":
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"challenge": payload.challenge or ""}

            # 调用 verify_feishu_event_token 完成当前步骤需要的业务处理。
            verify_feishu_event_token(payload)

            # 根据 settings.feishu_event_encrypt_key and payload.event ... 判断当前流程该进入哪个处理分支。
            if settings.feishu_event_encrypt_key and payload.event is None:
                # 当前实现只处理明文事件；启用飞书加密后应在这里接入解密逻辑。
                raise HTTPException(
                    # 将 status_code 的值保存下来，供后续流程判断或组装响应时使用。
                    status_code=400,
                    # 将 detail 的值保存下来，供后续流程判断或组装响应时使用。
                    detail=(
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "Encrypted Feishu events are not supported in this scaffold yet. "
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "Disable event encryption or add decryption support."
                    ),
                )

            # 将 header 的值保存下来，供后续流程判断或组装响应时使用。
            header = payload.header
            # 根据 header is None or header.event_type != "im.message.r... 判断当前流程该进入哪个处理分支。
            if header is None or header.event_type != "im.message.receive_v1":
                # 只处理消息接收事件，卡片回调由独立接口负责，避免不同事件结构混在一起解析。
                logger.info("HTTP Feishu event ignored unsupported event_type=%s", header.event_type if header else None)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ignored"}

            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event = payload.event or {}
            # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
            sender = event.get("sender") or {}
            # 将 sender_type 的值保存下来，供后续流程判断或组装响应时使用。
            sender_type = sender.get("sender_type")
            # 根据 sender_type and sender_type != "user" 判断当前流程该进入哪个处理分支。
            if sender_type and sender_type != "user":
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("HTTP Feishu event ignored sender_type=%s", sender_type)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ignored"}

            # 将 message 的值保存下来，供后续流程判断或组装响应时使用。
            message = event.get("message") or {}
            # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
            chat_id = str(message.get("chat_id") or "")
            # 根据 not chat_id 判断当前流程该进入哪个处理分支。
            if not chat_id:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("HTTP Feishu event ignored missing chat_id")
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ignored"}

            # 根据 settings.feishu_allowed_chat_ids and chat_id not in ... 判断当前流程该进入哪个处理分支。
            if settings.feishu_allowed_chat_ids and chat_id not in settings.feishu_allowed_chat_ids:
                # 群白名单是业务边界控制，防止机器人响应非预期群聊中的 @ 消息。
                logger.info("HTTP Feishu event ignored chat_id not allowed chat_id=%s", chat_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ignored"}

            # 将 mentions 的值保存下来，供后续流程判断或组装响应时使用。
            mentions = message.get("mentions") or []
            # 根据 settings.feishu_analyze_mention_only and not mention... 判断当前流程该进入哪个处理分支。
            if settings.feishu_analyze_mention_only and not mentions:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("HTTP Feishu event ignored because mention is required chat_id=%s", chat_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ignored"}

            # 将 content_text 的值保存下来，供后续流程判断或组装响应时使用。
            content_text = extract_text_from_message_content(message.get("content"))
            # 根据 not content_text.strip() 判断当前流程该进入哪个处理分支。
            if not content_text.strip():
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("HTTP Feishu event ignored empty content chat_id=%s", chat_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ignored"}

            # 将 sender_open_id 的值保存下来，供后续流程判断或组装响应时使用。
            sender_open_id = None
            # 将 sender_name 的值保存下来，供后续流程判断或组装响应时使用。
            sender_name = None
            # 根据 isinstance(sender, dict) 判断当前流程该进入哪个处理分支。
            if isinstance(sender, dict):
                # 将 sender_open_id 的值保存下来，供后续流程判断或组装响应时使用。
                sender_open_id = str(sender.get("sender_id") or sender.get("id") or "").strip() or None
                # 将 sender_name 的值保存下来，供后续流程判断或组装响应时使用。
                sender_name = str(sender.get("name") or "").strip() or None

            # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            thread_root_message_id = (
                # 调用 str 完成当前步骤需要的业务处理。
                str(message.get("root_id") or "").strip()
                # 调用 str 完成当前步骤需要的业务处理。
                or str(message.get("message_id") or "").strip()
                # 执行当前业务步骤，推动流程继续向下游推进。
                or None
            )
            # 根据 settings.interactive_topic_enabled 判断当前流程该进入哪个处理分支。
            if settings.interactive_topic_enabled:
                # 交互式流程优先复用原消息线程，在同一话题中维护诊断进度。
                if not thread_root_message_id:
                    # 记录关键运行信息，方便排查流程进展和异常现场。
                    logger.info("HTTP Feishu event ignored missing root message id chat_id=%s", chat_id)
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return {"status": "ignored", "reason": "missing root message id"}
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "HTTP Feishu event starts interactive topic chat_id=%s root_message_id=%s content_chars=%s mentions=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    chat_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    thread_root_message_id,
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(content_text),
                    # 调用 len 完成当前步骤需要的业务处理。
                    len(mentions),
                )
                # 将 task_id 的值保存下来，供后续流程判断或组装响应时使用。
                task_id = await get_interactive_topic_workflow().start(chat_id, thread_root_message_id, content_text)
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("HTTP Feishu event interactive topic started task_id=%s chat_id=%s", task_id, chat_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return {"status": "ok", "sent_to_feishu": True, "task_id": task_id}

            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            analysis, sent = await run_workflow_and_send(
                # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
                chat_id=chat_id,
                # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
                source="feishu-user",
                # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
                level="INFO",
                # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
                summary=f"{settings.feishu_bot_name} received a mention request",
                # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
                details=content_text,
                # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
                raw_text=content_text,
                # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
                trigger_type="user_message",
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=["feishu", "mention"],
                # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
                thread_root_message_id=thread_root_message_id,
                # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
                mention_open_id=sender_open_id,
                # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
                mention_name=sender_name,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"status": "ok", "sent_to_feishu": sent, "analysis_preview": analysis[:120]}
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except HTTPException:
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as exc:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to process Feishu event")
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 调用 report_exception 完成当前步骤需要的业务处理。
            report_exception("/feishu/events", exc)
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=500, detail="Failed to process Feishu event.") from exc

    @app.get("/metrics")
    # 定义 metrics 相关的处理逻辑，供流程或外部调用复用。
    def metrics() -> Response:
        # 暴露 Prometheus 指标文本，供监控系统定时抓取服务运行数据。
        if not settings.metrics_enabled:
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise HTTPException(status_code=404, detail="Metrics are disabled.")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return Response(content=monitor.render_latest(), media_type=CONTENT_TYPE_LATEST)

    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return app


# 定义 run_cli_once 相关的处理逻辑，供流程或外部调用复用。
def run_cli_once(settings: Settings, message: str) -> int:
    # 命令行模式下跑一次普通聊天，方便本地快速验证模型是否可用。
    chat_service = SingleTurnChatService(settings)
    # 将 answer 的值保存下来，供后续流程判断或组装响应时使用。
    answer = chat_service.reply_once(message)
    # 调用 print 完成当前步骤需要的业务处理。
    print(answer)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return 0


# 定义 run 相关的处理逻辑，供流程或外部调用复用。
def run() -> int:
    # 命令行入口；带 message 参数时跑一次聊天，否则启动 Web 服务。
    parser = argparse.ArgumentParser(description="Agent Sentinel runner")
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    parser.add_argument("--message", help="Run a single-turn chat in CLI mode.")
    # 将 args 的值保存下来，供后续流程判断或组装响应时使用。
    args = parser.parse_args()

    # 将 settings 的值保存下来，供后续流程判断或组装响应时使用。
    settings = get_settings()
    # 调用 configure_logging 完成当前步骤需要的业务处理。
    configure_logging(settings.log_level)

    # 根据 args.message 判断当前流程该进入哪个处理分支。
    if args.message:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return run_cli_once(settings, args.message)

    # 将 app 的值保存下来，供后续流程判断或组装响应时使用。
    app = build_app(settings)
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return 0


# 根据 __name__ == "__main__" 判断当前流程该进入哪个处理分支。
if __name__ == "__main__":
    # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
    raise SystemExit(run())
