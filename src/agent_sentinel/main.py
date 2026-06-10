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
    AlertAnalyzeRequest,
    AlertAnalyzeResponse,
    AlertRecordResponse,
    AlertReportRequest,
    AlertReportResponse,
    ChatRequest,
    ChatResponse,
    FeishuEventEnvelope,
    HealthResponse,
)
from agent_sentinel.service import SingleTurnChatService
from agent_sentinel.utils.logger import configure_logger


def configure_logging(log_level: str) -> None:
    # 方法说明：设置整套服务的日志级别，让后续每个模块都按同一规则输出运行记录。
    configure_logger(log_level)


def build_app(settings: Settings | None = None) -> FastAPI:
    # 方法说明：创建 Web 应用，并把配置、监控、飞书、诊断工作流等核心组件接到一起。
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    # Prometheus 指标开关在应用构建时统一生效，避免各业务节点重复读取环境变量。
    configure_monitoring(enabled=settings.metrics_enabled)
    logger = logging.getLogger(__name__)
    logger.info(
        "App build started app=%s env=%s interactive_topic=%s rag_provider=%s feishu_longconn=%s feishu_polling=%s metrics_enabled=%s",
        settings.app_name,
        settings.app_env,
        settings.interactive_topic_enabled,
        settings.rag_provider,
        settings.feishu_long_connection_enabled,
        settings.feishu_message_polling_enabled,
        settings.metrics_enabled,
    )

    notifier = FeishuWebhookNotifier(
        webhook_url=settings.feishu_webhook_url,
        secret=settings.feishu_secret,
        enabled=settings.feishu_alert_enabled,
    )
    alert_service = RealtimeAlertService(
        notifier=notifier,
        app_env=settings.app_env,
        title_prefix=settings.feishu_alert_title_prefix,
        dedup_window_seconds=settings.alert_dedup_window_seconds,
        store_limit=settings.alert_store_limit,
    )

    chat_service: SingleTurnChatService | None = None
    feishu_bot_client = FeishuBotClient(settings)
    feishu_sender = FeishuSender(feishu_bot_client)
    decision_store = HumanDecisionStore(settings.redis_url)
    card_handler = FeishuCardHandler(decision_store)
    aiops_workflow: DiagnosisWorkflow | None = None
    interactive_topic_workflow: InteractiveTopicWorkflow | None = None
    longconn_bot: FeishuLongConnectionBot | None = None
    poller_bot: FeishuMessagePoller | None = None

    app = FastAPI(title=settings.app_name)

    def get_chat_service() -> SingleTurnChatService:
        # 方法说明：按需创建普通聊天服务；只有真正调用聊天接口时才初始化模型对象。
        nonlocal chat_service
        # 单轮聊天服务只在首次请求时初始化，避免启动阶段就创建不一定会用到的模型客户端。
        if chat_service is None:
            chat_service = SingleTurnChatService(settings)
        return chat_service

    def get_aiops_workflow() -> DiagnosisWorkflow:
        # 方法说明：按需创建 AIOps 诊断工作流；它负责把告警理解、检索、生成方案等步骤串起来。
        nonlocal aiops_workflow
        # 诊断工作流依赖 LLM、RAG、飞书发送器等对象，采用懒加载可以降低健康检查和简单接口的启动成本。
        if aiops_workflow is None:
            logger.info("Initializing diagnosis workflow rag_provider=%s mock_llm=%s", settings.rag_provider, settings.aiops_mock_llm_enabled)
            aiops_workflow = DiagnosisWorkflow(
                settings=settings,
                llm=build_llm_executor(settings),
                sender=feishu_sender,
                decision_store=decision_store,
            )
        return aiops_workflow

    def get_interactive_topic_workflow() -> InteractiveTopicWorkflow:
        # 方法说明：按需创建飞书单卡片交互流程；它负责在同一张卡片里持续更新诊断进度。
        nonlocal interactive_topic_workflow
        # 交互式话题工作流有独立的状态存储和卡片更新逻辑，仅在启用并首次触发时创建。
        if interactive_topic_workflow is None:
            logger.info(
                "Initializing interactive topic workflow wait_seconds=%s rag_provider=%s static_top_k=%s history_top_k=%s final_top_k=%s",
                settings.interactive_topic_wait_seconds,
                settings.rag_provider,
                settings.rag_static_top_k,
                settings.rag_message_top_k,
                settings.rag_final_top_k,
            )
            interactive_topic_workflow = InteractiveTopicWorkflow(
                sender=InteractiveTopicSender(
                    feishu_bot_client,
                    wait_seconds=settings.interactive_topic_wait_seconds,
                ),
                wait_seconds=settings.interactive_topic_wait_seconds,
                retriever=build_retriever(settings),
                case_store=build_history_case_store(settings),
                llm=build_llm_executor(settings),
            )
        return interactive_topic_workflow

    def verify_alert_token(provided_token: str | None) -> None:
        # 方法说明：检查外部告警接口的访问令牌，防止未授权系统随意触发诊断。
        expected_token = settings.alert_api_token
        # 未配置 token 时保持开发环境易用；一旦配置，则所有告警入口都必须携带正确凭证。
        if expected_token and provided_token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid alert token.")

    def verify_feishu_event_token(payload: FeishuEventEnvelope) -> None:
        # 方法说明：检查飞书推送事件中的校验令牌，确认消息确实来自可信飞书应用。
        expected_token = settings.feishu_event_verification_token
        if not expected_token:
            return

        provided_token = None
        if payload.header and payload.header.token:
            provided_token = payload.header.token
        elif payload.token:
            provided_token = payload.token

        if provided_token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid Feishu event token.")

    def report_exception(scene: str, exc: Exception) -> None:
        # 方法说明：把接口内部异常转成飞书告警，便于运维人员第一时间知道服务自身出错。
        try:
            alert_service.report_exception(scene, exc)
        except Exception as notify_exc:  # pragma: no cover
            logger.exception("Failed to send Feishu alert for %s: %s", scene, notify_exc)

    def build_diagnosis_state(
        *,
        chat_id: str,
        source: str,
        level: str,
        summary: str,
        details: str | None = None,
        raw_text: str | None = None,
        trigger_type: str = "unknown",
        tags: list[str] | None = None,
        thread_root_message_id: str | None = None,
        mention_open_id: str | None = None,
        mention_name: str | None = None,
        workflow_thread_id: str | None = None,
        workflow_run_id: str | None = None,
    ) -> DiagnosisState:
        # 方法说明：把不同来源的告警统一整理成诊断状态，后面的每个节点都围绕这份状态继续补充信息。
        trace_id = trace_id_from_parts(chat_id, thread_root_message_id)
        # 所有诊断入口最终都归一化成同一份 LangGraph state，后续节点只依赖这个状态契约。
        return {
            "raw_alert": {
                "source": source,
                "level": level,
                "summary": summary,
                "details": details,
                "raw_text": raw_text,
                "trigger_type": trigger_type,
                "tags": tags,
            },
            "chat_id": chat_id,
            "thread_root_message_id": thread_root_message_id,
            "mention_open_id": mention_open_id,
            "mention_name": mention_name,
            "workflow_thread_id": workflow_thread_id or str(uuid.uuid4()),
            "workflow_run_id": workflow_run_id or str(uuid.uuid4()),
            # trace_id 用于日志和可观测性关联，不作为 Prometheus label，避免指标高基数。
            "trace_id": trace_id,
            "messages": [],
            "evidence": [],
            "retrieved_docs": [],
            "live_data": {},
            "recommended_plan": {},
            "validation_result": False,
            "need_human": True,
            "human_card_sent": False,
            "human_feedback": None,
        }

    async def run_workflow_and_send(
        *,
        chat_id: str,
        source: str,
        level: str,
        summary: str,
        details: str | None = None,
        raw_text: str | None = None,
        trigger_type: str = "unknown",
        tags: list[str] | None = None,
        thread_root_message_id: str | None = None,
        mention_open_id: str | None = None,
        mention_name: str | None = None,
    ) -> tuple[str, bool]:
        # 方法说明：执行完整诊断链路，并在需要时把最终结果发送回飞书会话。
        if not settings.alert_analysis_enabled:
            logger.info("Diagnosis workflow skipped because alert analysis is disabled chat_id=%s source=%s", chat_id, source)
            return "Alert analysis is disabled.", False

        started = time.perf_counter()
        # 这里先构造初始状态再进入工作流，便于统一记录 trace、线程 ID 和告警原文。
        initial_state = build_diagnosis_state(
            chat_id=chat_id,
            source=source,
            level=level,
            summary=summary,
            details=details,
            raw_text=raw_text,
            trigger_type=trigger_type,
            tags=tags,
            thread_root_message_id=thread_root_message_id,
            mention_open_id=mention_open_id,
            mention_name=mention_name,
        )
        logger.info(
            "Diagnosis workflow start trace_id=%s source=%s level=%s trigger_type=%s chat_id=%s thread_root_message_id=%s workflow_thread_id=%s workflow_run_id=%s summary_chars=%s raw_chars=%s tags=%s",
            initial_state.get("trace_id", ""),
            source,
            level,
            trigger_type,
            chat_id,
            thread_root_message_id,
            initial_state.get("workflow_thread_id", ""),
            initial_state.get("workflow_run_id", ""),
            len(summary or ""),
            len(raw_text or details or ""),
            tags or [],
        )
        final_state = await get_aiops_workflow().run_streaming(initial_state)
        final_text = str(final_state.get("final_text", ""))
        # 是否真实发送飞书消息只取决于 chat_id 和机器人配置，工作流本身仍然可以在无 chat_id 场景运行。
        sent = bool(chat_id and feishu_bot_client.is_configured())
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Diagnosis workflow completed trace_id=%s source=%s trigger_type=%s chat_id=%s workflow_thread_id=%s validation_result=%s human_decision=%s evidence=%s final_text_chars=%s sent=%s elapsed_ms=%s",
            final_state.get("trace_id", initial_state.get("trace_id", "")),
            source,
            trigger_type,
            chat_id,
            final_state.get("workflow_thread_id", ""),
            final_state.get("validation_result", False),
            final_state.get("human_decision", "unknown"),
            len(final_state.get("evidence", [])),
            len(final_text),
            sent,
            elapsed_ms,
        )
        return final_text, sent

    async def resume_workflow_from_callback(payload: dict[str, Any], transport: str) -> dict[str, str]:
        # 方法说明：处理飞书卡片按钮回调，把“人工确认/拒绝”的结果送回暂停中的工作流。
        logger.info("Card callback received transport=%s payload_keys=%s", transport, sorted(payload.keys()))
        if settings.interactive_topic_enabled:
            # 新版单卡片流程优先消费回调；如果不是它的 payload，再回退到旧版人审卡片解析。
            interactive_result = await get_interactive_topic_workflow().handle_card_callback(payload, source=transport)
            logger.info("Interactive topic callback result transport=%s status=%s", transport, interactive_result.get("status"))
            if interactive_result.get("status") != "ignored":
                return interactive_result

        context = await card_handler.parse_callback(payload)
        if context is None:
            return {"status": "ignored"}
        logger.info(
            "Received Feishu card callback transport=%s decision_id=%s workflow_thread_id=%s status=%s",
            transport,
            context.decision_id,
            context.workflow_thread_id,
            context.status,
        )
        if context.feedback:
            await get_aiops_workflow().update_state(
                context.workflow_thread_id,
                {"human_feedback": context.feedback},
            )
        await get_aiops_workflow().resume(
            context.workflow_thread_id,
            {"decision": context.status, "feedback": context.feedback},
        )
        logger.info(
            "Resumed workflow from Feishu callback transport=%s decision_id=%s workflow_thread_id=%s",
            transport,
            context.decision_id,
            context.workflow_thread_id,
        )
        return {"status": "ok"}

    def analyze_and_send_to_chat(
        chat_id: str,
        source: str,
        level: str,
        summary: str,
        details: str | None = None,
        raw_text: str | None = None,
        trigger_type: str = "unknown",
        tags: list[str] | None = None,
        thread_root_message_id: str | None = None,
        mention_open_id: str | None = None,
        mention_name: str | None = None,
    ) -> tuple[str, bool]:
        # 方法说明：把同步入口包装成统一诊断调用；普通流程直接跑工作流，单卡片模式走交互式话题。
        logger.info(
            "Analyze request routed chat_id=%s source=%s trigger_type=%s interactive_topic=%s root_message_id=%s query_chars=%s",
            chat_id,
            source,
            trigger_type,
            settings.interactive_topic_enabled,
            thread_root_message_id,
            len(raw_text or details or summary or ""),
        )
        if settings.interactive_topic_enabled:
            # 单卡片模式依赖原消息话题 ID 来持续更新同一张卡片，缺失时无法保证不刷屏。
            root_message_id = thread_root_message_id
            if not root_message_id:
                logger.warning("Interactive topic skipped because root message id is missing chat_id=%s", chat_id)
                return "Interactive topic skipped: root message id is missing.", False
            return get_interactive_topic_workflow().start_sync(
                chat_id,
                root_message_id,
                raw_text or details or summary,
            )
        return asyncio.run(
            run_workflow_and_send(
                chat_id=chat_id,
                source=source,
                level=level,
                summary=summary,
                details=details,
                raw_text=raw_text,
                trigger_type=trigger_type,
                tags=tags,
                thread_root_message_id=thread_root_message_id,
                mention_open_id=mention_open_id,
                mention_name=mention_name,
            )
        )

    @app.on_event("startup")
    def startup_event() -> None:
        # 方法说明：服务启动时准备飞书长连接和轮询组件，让机器人能主动接收群消息。
        nonlocal longconn_bot, poller_bot
        logger.info("Application startup begin")
        # 长连接和轮询都封装为可配置组件，组件内部会根据开关决定是否真正启动。
        if longconn_bot is None:
            longconn_bot = FeishuLongConnectionBot(
                settings=settings,
                analyze_callback=analyze_and_send_to_chat,
                card_action_callback=lambda payload: resume_workflow_from_callback(payload, transport="longconn"),
            )
        longconn_bot.start()

        if poller_bot is None:
            poller_bot = FeishuMessagePoller(
                settings=settings,
                feishu_bot_client=feishu_bot_client,
                analyze_callback=analyze_and_send_to_chat,
            )
        poller_bot.start()
        logger.info("Application startup completed")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        # 方法说明：服务关闭前收尾观测数据，尽量把模型调用链路和诊断 trace 写完整。
        logger.info("Application shutdown begin")
        observability_clients: list[LLMExecutor] = []
        # 关闭前主动 flush Langfuse 等可观测客户端，减少进程退出时丢失 trace 的概率。
        if aiops_workflow is not None:
            observability_clients.append(aiops_workflow.llm)
        if interactive_topic_workflow is not None and interactive_topic_workflow.llm is not None:
            observability_clients.append(interactive_topic_workflow.llm)
        for llm in observability_clients:
            try:
                await llm.flush_observability()
            except Exception:
                logger.warning("Failed to flush observability client", exc_info=True)
        logger.info("Application shutdown completed")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # 方法说明：提供健康检查接口，外部系统可用它判断服务是否还活着。
        return HealthResponse(status="ok", app=settings.app_name)

    @app.post("/chat/once", response_model=ChatResponse)
    def chat_once(payload: ChatRequest) -> ChatResponse:
        # 方法说明：提供一次性聊天接口，用于验证模型配置或做简单问答，不进入 AIOps 诊断流程。
        try:
            answer = get_chat_service().reply_once(payload.message)
            return ChatResponse(answer=answer, model=settings.openai_model)
        except Exception as exc:
            logger.exception("Single-turn chat failed")
            monitor.record_error("chat_once_error")
            report_exception("/chat/once", exc)
            raise HTTPException(status_code=500, detail="Chat request failed.") from exc

    @app.post("/alerts/test")
    def test_alert() -> dict[str, str]:
        # 方法说明：手动触发一条测试告警，用来验证飞书告警通道是否配置正确。
        try:
            dispatched, deduplicated = alert_service.report(
                source="agent-sentinel",
                level="INFO",
                summary="Manual alert test",
                details=f"Application {settings.app_name} triggered a manual alert test.",
                tags=["manual-test"],
            )
            if deduplicated:
                return {"status": "deduplicated"}
            return {"status": "sent" if dispatched else "skipped"}
        except Exception as exc:
            logger.exception("Failed to send test alert")
            monitor.record_error("alert_test_error")
            raise HTTPException(status_code=500, detail="Failed to send alert.") from exc

    @app.post("/alerts/report", response_model=AlertReportResponse)
    def report_alert(
        payload: AlertReportRequest,
        x_alert_token: str | None = Header(default=None),
    ) -> AlertReportResponse:
        # 方法说明：接收外部系统上报的告警，只负责记录/通知，不启动完整 AI 诊断。
        try:
            verify_alert_token(x_alert_token)
            dispatched, deduplicated = alert_service.report(
                source=payload.source,
                level=payload.level,
                summary=payload.summary,
                details=payload.details,
                dedupe_key=payload.dedupe_key,
                tags=payload.tags,
            )
            status = "deduplicated" if deduplicated else "sent" if dispatched else "skipped"
            return AlertReportResponse(
                status=status,
                dispatched=dispatched,
                deduplicated=deduplicated,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to report alert")
            monitor.record_error("alert_report_error")
            raise HTTPException(status_code=500, detail="Failed to report alert.") from exc

    @app.get("/alerts/recent", response_model=list[AlertRecordResponse])
    def recent_alerts(
        limit: int = Query(default=20, ge=1, le=100),
        x_alert_token: str | None = Header(default=None),
    ) -> list[AlertRecordResponse]:
        # 方法说明：查询最近的告警记录，方便排查“告警是否已发送、是否被去重”。
        verify_alert_token(x_alert_token)
        return [AlertRecordResponse(**item) for item in alert_service.recent_alerts(limit=limit)]

    @app.post("/alerts/analyze", response_model=AlertAnalyzeResponse)
    async def analyze_alert(
        payload: AlertAnalyzeRequest,
        x_alert_token: str | None = Header(default=None),
    ) -> AlertAnalyzeResponse:
        # 方法说明：接收外部告警并启动 AI 诊断，最后返回诊断文本以及是否已发送飞书。
        try:
            verify_alert_token(x_alert_token)
            thread_root_message_id = payload.thread_root_message_id or payload.message_id
            analysis, sent = await run_workflow_and_send(
                chat_id=payload.chat_id,
                source=payload.source,
                level=payload.level,
                summary=payload.summary,
                details=payload.details,
                raw_text=payload.raw_text,
                trigger_type=payload.trigger_type,
                tags=payload.tags,
                thread_root_message_id=thread_root_message_id,
                mention_open_id=payload.mention_open_id,
                mention_name=payload.mention_name,
            )
            return AlertAnalyzeResponse(
                status="sent" if sent else "generated",
                analysis=analysis,
                sent_to_feishu=sent,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to analyze alert")
            monitor.record_error("alert_analyze_error")
            report_exception("/alerts/analyze", exc)
            raise HTTPException(status_code=500, detail="Failed to analyze alert.") from exc

    @app.post("/aiops/diagnose")
    async def aiops_diagnose(
        payload: AlertAnalyzeRequest,
        x_alert_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        # 方法说明：直接运行 LangGraph 诊断并返回完整状态，适合调试每个诊断节点的中间结果。
        try:
            verify_alert_token(x_alert_token)
            initial_state = build_diagnosis_state(
                chat_id=payload.chat_id,
                source=payload.source,
                level=payload.level,
                summary=payload.summary,
                details=payload.details,
                raw_text=payload.raw_text,
                trigger_type=payload.trigger_type,
                tags=payload.tags,
                thread_root_message_id=payload.thread_root_message_id or payload.message_id,
                mention_open_id=payload.mention_open_id,
                mention_name=payload.mention_name,
            )
            logger.info(
                "Starting direct LangGraph diagnosis source=%s level=%s trigger_type=%s chat_id=%s thread_root_message_id=%s workflow_thread_id=%s summary=%s",
                payload.source,
                payload.level,
                payload.trigger_type,
                payload.chat_id,
                payload.thread_root_message_id or payload.message_id,
                initial_state.get("workflow_thread_id", ""),
                payload.summary,
            )
            final_state = await get_aiops_workflow().run_streaming(initial_state)
            logger.info(
                "Completed direct LangGraph diagnosis source=%s trigger_type=%s chat_id=%s validation_result=%s human_decision=%s evidence=%s final_text_chars=%s",
                payload.source,
                payload.trigger_type,
                payload.chat_id,
                final_state.get("validation_result", False),
                final_state.get("human_decision", "unknown"),
                len(final_state.get("evidence", [])),
                len(str(final_state.get("final_text", ""))),
            )
            return {
                "status": "ok",
                "summary": final_state.get("alert_summary", ""),
                "recommended_plan": final_state.get("recommended_plan", {}),
                "evidence": final_state.get("evidence", []),
                "validation_result": final_state.get("validation_result", False),
                "human_decision": final_state.get("human_decision", "unknown"),
                "final_text": final_state.get("final_text", ""),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("AIOps diagnosis failed")
            monitor.record_error("aiops_diagnose_error")
            report_exception("/aiops/diagnose", exc)
            raise HTTPException(status_code=500, detail="AIOps diagnosis failed.") from exc

    @app.post("/feishu/card/callback")
    async def feishu_card_callback(request: Request) -> dict[str, str]:
        # 方法说明：接收飞书卡片按钮事件，例如“批准方案”“拒绝方案”等人工操作。
        try:
            payload = await request.json()
            logger.info("HTTP Feishu card callback accepted payload_keys=%s", sorted(payload.keys()))
            return await resume_workflow_from_callback(payload, transport="http")
        except Exception as exc:
            logger.exception("Failed to process Feishu card callback")
            monitor.record_error("feishu_api_error")
            raise HTTPException(status_code=500, detail="Failed to process card callback.") from exc

    @app.post("/webhook/card")
    async def webhook_card_callback(request: Request) -> dict[str, str]:
        # 方法说明：提供兼容 webhook 的卡片回调入口，最终仍复用同一套恢复工作流逻辑。
        try:
            payload = await request.json()
            logger.info("HTTP webhook card callback accepted payload_keys=%s", sorted(payload.keys()))
            return await resume_workflow_from_callback(payload, transport="http")
        except Exception as exc:
            logger.exception("Failed to process webhook card callback")
            monitor.record_error("feishu_api_error")
            raise HTTPException(status_code=500, detail="Failed to process card callback.") from exc

    @app.post("/feishu/events")
    async def feishu_events(payload: FeishuEventEnvelope) -> dict[str, object]:
        # 方法说明：接收飞书普通消息事件，把用户在群里的 @ 消息转换成一次诊断请求。
        try:
            logger.info(
                "HTTP Feishu event received type=%s challenge=%s event_type=%s",
                payload.type,
                bool(payload.challenge),
                payload.header.event_type if payload.header else None,
            )
            if payload.challenge:
                return {"challenge": payload.challenge}
            if payload.type == "url_verification":
                return {"challenge": payload.challenge or ""}

            verify_feishu_event_token(payload)

            if settings.feishu_event_encrypt_key and payload.event is None:
                # 当前实现只处理明文事件；启用飞书加密后应在这里接入解密逻辑。
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Encrypted Feishu events are not supported in this scaffold yet. "
                        "Disable event encryption or add decryption support."
                    ),
                )

            header = payload.header
            if header is None or header.event_type != "im.message.receive_v1":
                # 只处理消息接收事件，卡片回调由独立接口负责，避免不同事件结构混在一起解析。
                logger.info("HTTP Feishu event ignored unsupported event_type=%s", header.event_type if header else None)
                return {"status": "ignored"}

            event = payload.event or {}
            sender = event.get("sender") or {}
            sender_type = sender.get("sender_type")
            if sender_type and sender_type != "user":
                logger.info("HTTP Feishu event ignored sender_type=%s", sender_type)
                return {"status": "ignored"}

            message = event.get("message") or {}
            chat_id = str(message.get("chat_id") or "")
            if not chat_id:
                logger.info("HTTP Feishu event ignored missing chat_id")
                return {"status": "ignored"}

            if settings.feishu_allowed_chat_ids and chat_id not in settings.feishu_allowed_chat_ids:
                # 群白名单是业务边界控制，防止机器人响应非预期群聊中的 @ 消息。
                logger.info("HTTP Feishu event ignored chat_id not allowed chat_id=%s", chat_id)
                return {"status": "ignored"}

            mentions = message.get("mentions") or []
            if settings.feishu_analyze_mention_only and not mentions:
                logger.info("HTTP Feishu event ignored because mention is required chat_id=%s", chat_id)
                return {"status": "ignored"}

            content_text = extract_text_from_message_content(message.get("content"))
            if not content_text.strip():
                logger.info("HTTP Feishu event ignored empty content chat_id=%s", chat_id)
                return {"status": "ignored"}

            sender_open_id = None
            sender_name = None
            if isinstance(sender, dict):
                sender_open_id = str(sender.get("sender_id") or sender.get("id") or "").strip() or None
                sender_name = str(sender.get("name") or "").strip() or None

            thread_root_message_id = (
                str(message.get("root_id") or "").strip()
                or str(message.get("message_id") or "").strip()
                or None
            )
            if settings.interactive_topic_enabled:
                # 交互式流程优先复用原消息线程，在同一话题中维护诊断进度。
                if not thread_root_message_id:
                    logger.info("HTTP Feishu event ignored missing root message id chat_id=%s", chat_id)
                    return {"status": "ignored", "reason": "missing root message id"}
                logger.info(
                    "HTTP Feishu event starts interactive topic chat_id=%s root_message_id=%s content_chars=%s mentions=%s",
                    chat_id,
                    thread_root_message_id,
                    len(content_text),
                    len(mentions),
                )
                task_id = await get_interactive_topic_workflow().start(chat_id, thread_root_message_id, content_text)
                logger.info("HTTP Feishu event interactive topic started task_id=%s chat_id=%s", task_id, chat_id)
                return {"status": "ok", "sent_to_feishu": True, "task_id": task_id}

            analysis, sent = await run_workflow_and_send(
                chat_id=chat_id,
                source="feishu-user",
                level="INFO",
                summary=f"{settings.feishu_bot_name} received a mention request",
                details=content_text,
                raw_text=content_text,
                trigger_type="user_message",
                tags=["feishu", "mention"],
                thread_root_message_id=thread_root_message_id,
                mention_open_id=sender_open_id,
                mention_name=sender_name,
            )
            return {"status": "ok", "sent_to_feishu": sent, "analysis_preview": analysis[:120]}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to process Feishu event")
            monitor.record_error("feishu_api_error")
            report_exception("/feishu/events", exc)
            raise HTTPException(status_code=500, detail="Failed to process Feishu event.") from exc

    @app.get("/metrics")
    def metrics() -> Response:
        # 方法说明：暴露 Prometheus 指标文本，供监控系统定时抓取服务运行数据。
        if not settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="Metrics are disabled.")
        return Response(content=monitor.render_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def run_cli_once(settings: Settings, message: str) -> int:
    # 方法说明：命令行模式下跑一次普通聊天，方便本地快速验证模型是否可用。
    chat_service = SingleTurnChatService(settings)
    answer = chat_service.reply_once(message)
    print(answer)
    return 0


def run() -> int:
    # 方法说明：命令行入口；带 message 参数时跑一次聊天，否则启动 Web 服务。
    parser = argparse.ArgumentParser(description="Agent Sentinel runner")
    parser.add_argument("--message", help="Run a single-turn chat in CLI mode.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.message:
        return run_cli_once(settings, args.message)

    app = build_app(settings)
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
