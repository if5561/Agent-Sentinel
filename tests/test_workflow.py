from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from langgraph.types import Command

from agent_sentinel.config import Settings
from agent_sentinel.agents.generate_plan import PlanOutput
from agent_sentinel.feishu.card_handler import HumanDecisionStore
from agent_sentinel.feishu.sender import FeishuSender
from agent_sentinel.graph.workflow import DiagnosisWorkflow, build_llm_executor


def make_settings() -> Settings:
    return Settings(
        app_name="Agent Sentinel",
        app_host="127.0.0.1",
        app_port=8000,
        app_env="test",
        log_level="INFO",
        openai_api_key="",
        openai_base_url=None,
        openai_model="gpt-4o-mini",
        openai_temperature=0.0,
        openai_http_trust_env=False,
        feishu_api_base_url="https://open.feishu.cn",
        feishu_app_id=None,
        feishu_app_secret=None,
        feishu_event_verification_token=None,
        feishu_event_encrypt_key=None,
        feishu_bot_name="Analysis Bot",
        feishu_allowed_chat_ids=[],
        feishu_analyze_mention_only=True,
        feishu_long_connection_enabled=False,
        feishu_message_polling_enabled=False,
        feishu_message_polling_interval_seconds=5,
        feishu_message_polling_page_size=20,
        feishu_webhook_url=None,
        feishu_secret=None,
        feishu_alert_enabled=False,
        feishu_alert_title_prefix="Agent Sentinel",
        alert_analysis_enabled=True,
        alert_analysis_title_prefix="Alert Analysis",
        alert_api_token="test-token",
        alert_dedup_window_seconds=60,
        alert_store_limit=100,
        aiops_mock_llm_enabled=True,
        aiops_human_confirm_enabled=False,
        redis_url=None,
    )


def test_langgraph_workflow_runs_mock_diagnosis() -> None:
    async def run() -> None:
        settings = make_settings()
        workflow = DiagnosisWorkflow(
            settings=settings,
            llm=build_llm_executor(settings),
            sender=FeishuSender(None),
            decision_store=HumanDecisionStore(None),
        )
        final_state = await workflow.run_streaming(
            {
                "raw_alert": {
                    "source": "unit-test",
                    "level": "ERROR",
                    "summary": "order sync timeout",
                    "details": "timeout after 3 retries",
                },
                "chat_id": "",
                "workflow_thread_id": "wf-no-chat",
                "workflow_run_id": "run-no-chat",
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
        )
        assert final_state["alert_summary"]
        assert final_state["recommended_plan"]
        assert final_state["evidence"]
        assert final_state["validation_result"] is True
        assert final_state["human_decision"] == "approved"
        assert "AIOps Diagnosis" in final_state["final_text"]

    asyncio.run(run())


def test_langgraph_workflow_sends_progress_updates_for_feishu_chat() -> None:
    async def run() -> None:
        settings = make_settings()
        settings.aiops_human_confirm_enabled = True
        sender = FeishuSender(None)
        sender.send_message = AsyncMock(return_value=True)
        sender.send_card = AsyncMock(return_value=True)
        workflow = DiagnosisWorkflow(
            settings=settings,
            llm=build_llm_executor(settings),
            sender=sender,
            decision_store=HumanDecisionStore(None),
        )

        interrupted_state = {
            "raw_alert": {
                "source": "unit-test",
                "level": "ERROR",
                "summary": "order sync timeout",
                "details": "timeout after 3 retries",
            },
            "chat_id": "oc_test_chat",
            "thread_root_message_id": "om_thread_root",
            "mention_open_id": "ou_user",
            "mention_name": "Fe",
            "workflow_thread_id": "wf-chat",
            "workflow_run_id": "run-chat",
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
        await workflow.run_streaming(interrupted_state)

        sender.send_card.assert_awaited_once()
        resumed_state = await workflow.resume(
            "wf-chat",
            {"decision": "approved", "feedback": ""},
        )

        assert resumed_state["human_decision"] == "approved"
        assert sender.send_message.await_count >= 6
        first_call = sender.send_message.await_args_list[0]
        assert first_call.args[0] == "oc_test_chat"
        assert "已完成告警理解" in first_call.args[1]
        assert first_call.kwargs["thread_root_message_id"] == "om_thread_root"
        final_result_calls = [
            call for call in sender.send_message.await_args_list if "[AIOps Diagnosis]" in call.args[1]
        ]
        assert len(final_result_calls) == 1
        final_call = final_result_calls[0]
        assert final_call.args[0] == "oc_test_chat"
        assert final_call.kwargs["thread_root_message_id"] == "om_thread_root"
        assert final_call.kwargs["mention_open_id"] == "ou_user"
        assert final_call.kwargs["mention_name"] == "Fe"

    asyncio.run(run())


def test_langgraph_workflow_rejected_feedback_returns_to_generate_plan() -> None:
    async def run() -> None:
        settings = make_settings()
        settings.aiops_human_confirm_enabled = True
        sender = FeishuSender(None)
        sender.send_message = AsyncMock(return_value=True)
        sender.send_card = AsyncMock(return_value=True)
        workflow = DiagnosisWorkflow(
            settings=settings,
            llm=build_llm_executor(settings),
            sender=sender,
            decision_store=HumanDecisionStore(None),
        )

        await workflow.run_streaming(
            {
                "raw_alert": {
                    "source": "unit-test",
                    "level": "ERROR",
                    "summary": "order sync timeout",
                    "details": "timeout after 3 retries",
                },
                "chat_id": "oc_test_chat",
                "thread_root_message_id": "om_thread_root",
                "mention_open_id": "ou_user",
                "mention_name": "Fe",
                "workflow_thread_id": "wf-reject",
                "workflow_run_id": "run-reject",
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
        )
        resumed_state = await workflow.resume(
            "wf-reject",
            {"decision": "rejected", "feedback": "请补充回滚影响说明"},
        )

        assert resumed_state["human_decision"] in {"approved", "rejected"}
        assert resumed_state["human_feedback"] == "请补充回滚影响说明"
        assert any(message["content"] == "请补充回滚影响说明" for message in resumed_state["messages"])
        assert sender.send_card.await_count >= 1

    asyncio.run(run())


def test_plan_output_accepts_action_list_from_llm() -> None:
    parsed = PlanOutput.model_validate(
        {
            "recommended_plan": [{"action": "check downstream timeout"}],
            "evidence": ["payment-api timeout"],
            "need_human": True,
        }
    )

    assert parsed.recommended_plan == {"actions": [{"action": "check downstream timeout"}]}
