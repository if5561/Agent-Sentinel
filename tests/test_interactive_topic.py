from __future__ import annotations

import asyncio
import json
import logging

from agent_sentinel.interactive_topic.topic_sender import WORKFLOW_STEPS
from agent_sentinel.interactive_topic.workflow import (
    InteractiveTopicWorkflow,
    _build_evidence_review_prompt,
    _build_summary_prompt,
)
from agent_sentinel.rag.models import RetrievedDoc
from agent_sentinel.utils.logger import configure_logger


class FakeSender:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []
        self.sent_cards: list[dict[str, object]] = []

    async def send_workflow_card(self, chat_id: str, root_message_id: str, task_id: str, query: str) -> str:
        self.sent_cards.append(
            {
                "chat_id": chat_id,
                "root_message_id": root_message_id,
                "task_id": task_id,
                "query": query,
            }
        )
        return f"card-{task_id}"

    async def update_workflow_card(self, message_id: str | None, **kwargs: object) -> bool:
        self.updates.append({"message_id": message_id, **kwargs})
        return True


class FakeRetriever:
    async def retrieve(self, query, filters=None):
        return [
            RetrievedDoc(
                id="doc-1",
                text=f"真实检索片段：{query} 可能与 MQ 堆积有关。",
                source_type="static_doc",
                score=0.91,
                title="消息堆积排查",
                source_uri="data/static_docs/manual.md",
                metadata={
                    "section": "第一章_消息队列_1.1_消息堆积",
                    "alert_category": "MQ",
                    "severity_level": "P0",
                },
            )
        ]


class MultiDocRetriever:
    async def retrieve(self, query, filters=None):
        return [
            RetrievedDoc(id=f"doc-{index}", text=f"片段 {index}", source_type="static_doc", score=0.9 - index / 100)
            for index in range(1, 4)
        ]


class FakeCaseStore:
    def __init__(self) -> None:
        self.saved_state = None

    async def search_similar_cases(self, alert_text, *, top_k=2, threshold=None):
        return [
            RetrievedDoc(
                id="alert-case-1",
                text=alert_text,
                source_type="message_history",
                score=0.96,
                title="CPU 飙升成功案例",
                metadata={"recommended_plan": {"summary": "扩容 worker"}},
            )
        ]

    async def save_case_to_history(self, state):
        self.saved_state = state
        return "saved-case-1"


class FailingCaseStore(FakeCaseStore):
    async def save_case_to_history(self, state):
        self.saved_state = state
        raise RuntimeError("milvus varchar too long")


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def call(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        return "LLM 诊断总结"


class WorkflowLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def call(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        if len(self.prompts) == 3:
            return "PASS"
        if len(self.prompts) == 2:
            return (
                '{"recommended_plan":{"summary":"数据库连接超时导致接口错误率升高",'
                '"actions":["检查连接池耗尽情况","查看慢查询和锁等待"],"risk_level":"medium"},'
                '"evidence":["RAG 命中数据库连接池排查文档"],"need_human":true}'
            )
        return "订单接口出现数据库连接超时，需要结合 RAG 和实时数据继续诊断。"



class RouterLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if kwargs.get("prompt_name") == "tool_router":
            return json.dumps(
                {
                    "need_tools": True,
                    "reason": "Need metrics and logs.",
                    "tool_calls": [
                        {
                            "tool": "prometheus_query_metrics",
                            "purpose": "Check service up.",
                            "arguments": {"promql": "up"},
                        },
                        {
                            "tool": "aliyun_sls_query_logs",
                            "purpose": "Check node events.",
                            "arguments": {"query": "event_type: node_event"},
                        },
                    ],
                }
            )
        if kwargs.get("prompt_name") == "evidence_review":
            return json.dumps(
                {
                    "is_sufficient": True,
                    "reason": "Metrics and logs returned data.",
                    "missing_evidence": [],
                    "tool_success_count": 2,
                    "tool_failure_count": 0,
                    "evidence_count": 2,
                }
            )
        return "OK"


class FakeMetricsProvider:
    async def get_metrics(self, alert_summary, group_id=None):
        return {"provider": "fake_prometheus", "series": [{"metric": {"job": "agent-sentinel"}, "value": [1, "1"]}], "summary": "1 series"}


class FakeLogsProvider:
    async def query_logs(self, alert_summary, group_id=None):
        return {"provider": "fake_sls", "matches": [{"node": "understand"}], "total": 1, "query": alert_summary}


class FakeTopologyProvider:
    async def get_topology(self, alert_summary, group_id=None):
        return {"provider": "fake_topology", "service": "agent-sentinel", "dependencies": ["litellm"]}


class FakeToolsProvider:
    def __init__(self) -> None:
        self.metrics = FakeMetricsProvider()
        self.logs = FakeLogsProvider()
        self.topology = FakeTopologyProvider()

    async def fetch_all(self, alert_summary, group_id=None):
        return {
            "metrics": await self.metrics.get_metrics(alert_summary, group_id),
            "logs": await self.logs.query_logs(alert_summary, group_id),
            "topology": await self.topology.get_topology(alert_summary, group_id),
        }


def test_interactive_topic_routes_retry_twice_to_next() -> None:
    workflow = InteractiveTopicWorkflow(sender=FakeSender())  # type: ignore[arg-type]

    route = workflow._route_after_confirm(  # noqa: SLF001
        {
            "current_node": "rag_retrieve",
            "last_action": "retry",
            "retry_counts": {"rag_retrieve": 2},
        }
    )

    assert route == "next"


def test_interactive_topic_node_event_log_is_structured(caplog) -> None:
    workflow = InteractiveTopicWorkflow(sender=FakeSender())  # type: ignore[arg-type]

    with caplog.at_level(logging.INFO, logger="agent_sentinel.interactive_topic.workflow"):
        workflow._log_node_event(  # noqa: SLF001
            event="completed",
            node_name="rag_retrieve",
            state={
                "trace_id": "oc-chat_om-root",
                "task_id": "topic-1",
                "chat_id": "oc-chat",
                "root_message_id": "om-root",
                "retry_counts": {"rag_retrieve": 1},
                "node_results": [{"node_name": "understand", "result": "ok"}],
            },
            status="done",
            result_chars=12,
            elapsed_ms=345,
        )

    message = next(record.getMessage() for record in caplog.records if record.getMessage().startswith("node_event "))
    payload = json.loads(message.removeprefix("node_event "))
    assert payload["event_type"] == "node_event"
    assert payload["workflow_type"] == "interactive_topic"
    assert payload["event"] == "completed"
    assert payload["status"] == "done"
    assert payload["trace_id"] == "oc-chat_om-root"
    assert payload["business_trace_id"] == "oc-chat_om-root"
    assert payload["task_id"] == "topic-1"
    assert payload["node"] == "rag_retrieve"
    assert payload["retry_count"] == 1
    assert payload["previous_results"] == 1
    assert payload["result_chars"] == 12
    assert payload["elapsed_ms"] == 345


def test_interactive_topic_node_event_also_logs_plain_json(caplog) -> None:
    workflow = InteractiveTopicWorkflow(sender=FakeSender())  # type: ignore[arg-type]

    with caplog.at_level(logging.INFO, logger="agent_sentinel.node_event_json"):
        workflow._log_node_event(  # noqa: SLF001
            event="start",
            node_name="summary",
            state={
                "trace_id": "business-1",
                "task_id": "topic-json",
                "chat_id": "oc-chat",
                "root_message_id": "om-root",
                "retry_counts": {},
                "node_results": [],
            },
            status="running",
        )

    message = next(record.getMessage() for record in caplog.records if record.name == "agent_sentinel.node_event_json")
    payload = json.loads(message)
    assert payload["event_type"] == "node_event"
    assert payload["trace_id"] == "business-1"
    assert payload["node"] == "summary"
    assert payload["event"] == "start"
    assert payload["status"] == "running"


def test_interactive_topic_node_event_writes_jsonl_file(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "node_events.jsonl"
    logger = logging.getLogger("agent_sentinel.node_event_json")
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate
    for handler in old_handlers:
        logger.removeHandler(handler)
    monkeypatch.setenv("NODE_EVENT_JSONL_PATH", str(log_path))
    try:
        configure_logger("INFO")
        workflow = InteractiveTopicWorkflow(sender=FakeSender())  # type: ignore[arg-type]
        workflow._log_node_event(  # noqa: SLF001
            event="waiting",
            node_name="validate",
            state={
                "trace_id": "business-jsonl",
                "task_id": "topic-jsonl",
                "chat_id": "oc-chat",
                "root_message_id": "om-root",
                "retry_counts": {},
                "node_results": [],
            },
            status="waiting",
            timeout_seconds=15,
        )
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.propagate = old_propagate

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "node_event"
    assert payload["trace_id"] == "business-jsonl"
    assert payload["node"] == "validate"
    assert payload["event"] == "waiting"
    assert payload["timeout_seconds"] == 15


def test_interactive_topic_duplicate_start_reuses_existing_task_without_new_card() -> None:
    async def run() -> None:
        sender = FakeSender()
        workflow = InteractiveTopicWorkflow(sender=sender)  # type: ignore[arg-type]
        drive_calls = 0

        async def fake_drive(_task_id, _state):
            nonlocal drive_calls
            drive_calls += 1

        workflow._drive = fake_drive  # type: ignore[method-assign]  # noqa: SLF001

        first = await workflow.start("oc-chat", "om-root", "first query")
        second = await workflow.start("oc-chat", "om-root", "duplicate query")

        assert first == second
        assert drive_calls == 1
        assert len(sender.sent_cards) == 1
        assert sender.sent_cards[0]["root_message_id"] == "om-root"

    asyncio.run(run())


def test_interactive_topic_includes_llm_diagnosis_nodes() -> None:
    async def run() -> None:
        llm = WorkflowLLM()
        workflow = InteractiveTopicWorkflow(
            sender=FakeSender(),  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
        )
        state = {
            "query": "订单接口 timeout",
            "task_id": "topic-llm",
            "chat_id": "oc-chat",
            "root_message_id": "om-root",
            "node_results": [],
            "retry_counts": {},
        }

        understand_result, diagnosis_state = await workflow._understand(state)  # noqa: SLF001
        state["diagnosis_state"] = diagnosis_state
        plan_result, diagnosis_state = await workflow._generate_plan(state)  # noqa: SLF001
        state["diagnosis_state"] = diagnosis_state
        validate_result, diagnosis_state = await workflow._validate(state)  # noqa: SLF001

        step_names = [step.node_name for step in WORKFLOW_STEPS]
        assert "understand" in step_names
        assert "generate_plan" in step_names
        assert "validate" in step_names
        assert "告警理解完成" in understand_result
        assert "方案生成完成" in plan_result
        assert "方案校验完成：通过" in validate_result
        assert diagnosis_state["validation_result"] is True
        assert len(llm.prompts) == 3

    asyncio.run(run())



def test_interactive_topic_react_router_logs_decision_and_metadata(caplog) -> None:
    async def run() -> None:
        llm = RouterLLM()
        workflow = InteractiveTopicWorkflow(sender=FakeSender(), llm=llm)  # type: ignore[arg-type]
        state = {
            "query": "checkout-api 5xx",
            "task_id": "topic-react",
            "chat_id": "oc-chat",
            "root_message_id": "om-root",
            "trace_id": "oc-chat_om-root",
            "node_results": [{"node_name": "rag_retrieve", "result": "RAG evidence"}],
            "retry_counts": {},
            "diagnosis_state": {"alert_summary": "checkout-api 5xx", "evidence": []},
        }

        with caplog.at_level(logging.INFO, logger="agent_sentinel.node_event_json"):
            result, diagnosis_state = await workflow._tool_router(state)  # noqa: SLF001

        assert "prometheus_query_metrics" in result
        assert diagnosis_state["tool_plan"]["need_tools"] is True
        router_call = llm.calls[0]
        assert router_call["prompt_name"] == "tool_router"
        assert router_call["metadata"]["node_name"] == "tool_router"
        event = next(json.loads(record.getMessage()) for record in caplog.records if '"event_type": "agent_decision"' in record.getMessage())
        assert event["business_trace_id"] == "oc-chat_om-root"
        assert event["selected_tools"] == ["prometheus_query_metrics", "aliyun_sls_query_logs"]
        assert event["tool_call_count"] == 2

    asyncio.run(run())


def test_interactive_topic_react_executor_and_review_log_events(monkeypatch, caplog) -> None:
    async def run() -> None:
        import agent_sentinel.interactive_topic.workflow as workflow_module

        monkeypatch.setattr(workflow_module, "_tools_provider", FakeToolsProvider())
        llm = RouterLLM()
        workflow = InteractiveTopicWorkflow(sender=FakeSender(), llm=llm)  # type: ignore[arg-type]
        state = {
            "query": "checkout-api 5xx",
            "task_id": "topic-react",
            "chat_id": "oc-chat",
            "root_message_id": "om-root",
            "trace_id": "oc-chat_om-root",
            "node_results": [],
            "retry_counts": {},
            "tool_plan": {
                "need_tools": True,
                "reason": "Need metrics and logs.",
                "tool_calls": [
                    {"tool": "prometheus_query_metrics", "purpose": "Check up", "arguments": {"promql": "up"}},
                    {"tool": "aliyun_sls_query_logs", "purpose": "Check logs", "arguments": {"query": "event_type: node_event"}},
                ],
            },
            "diagnosis_state": {"alert_summary": "checkout-api 5xx", "evidence": []},
        }

        with caplog.at_level(logging.INFO, logger="agent_sentinel.node_event_json"):
            result, diagnosis_state = await workflow._tool_executor(state)  # noqa: SLF001
            state["tool_results"] = diagnosis_state["tool_results"]
            state["diagnosis_state"] = diagnosis_state
            review_result, reviewed_state = await workflow._evidence_review(state)  # noqa: SLF001

        assert "Tool executor completed" in result
        assert len(diagnosis_state["tool_results"]) == 2
        assert reviewed_state["evidence_review"]["is_sufficient"] is True
        assert "Evidence review completed" in review_result
        messages = [record.getMessage() for record in caplog.records]
        tool_events = [json.loads(message) for message in messages if '"event_type": "tool_call"' in message]
        review_events = [json.loads(message) for message in messages if '"event_type": "evidence_review"' in message]
        assert [event["tool_name"] for event in tool_events] == ["prometheus_query_metrics", "aliyun_sls_query_logs"]
        assert all(event["status"] == "success" for event in tool_events)
        assert review_events[-1]["is_sufficient"] is True
        assert llm.calls[-1]["prompt_name"] == "evidence_review"
        assert llm.calls[-1]["metadata"]["tool_success_count"] == 2

    asyncio.run(run())


def test_evidence_review_prompt_falls_back_to_business_trace_id_parts() -> None:
    prompt = _build_evidence_review_prompt(
        {
            "query": "health check",
            "chat_id": "oc-chat",
            "root_message_id": "om-root",
        },
        {"evidence": []},
        [],
        {"is_sufficient": False},
    )

    assert '"business_trace_id": "oc-chat_om-root"' in prompt


def test_interactive_topic_failed_node_is_skipped() -> None:
    async def run() -> None:
        sender = FakeSender()
        workflow = InteractiveTopicWorkflow(sender=sender)  # type: ignore[arg-type]

        async def failing_runner(_state):
            raise RuntimeError("boom")

        result = await workflow._interactive_node(  # noqa: SLF001
            "rag_retrieve",
            failing_runner,
            {
                "task_id": "topic-1",
                "query": "CPU 飙升",
                "node_results": [],
                "retry_counts": {},
            },
        )

        assert result["last_action"] == "skip"
        assert "执行失败" in result["node_result"]
        assert sender.updates[-1]["node_statuses"] == {"rag_retrieve": "skipped"}

    asyncio.run(run())


def test_interactive_topic_confirmation_replay_does_not_rerun_node() -> None:
    async def run() -> None:
        sender = FakeSender()
        workflow = InteractiveTopicWorkflow(sender=sender)  # type: ignore[arg-type]
        task = workflow.task_store.create_task("topic-replay", "oc-chat", "om-root", "数据库连接超时")
        workflow.task_store.set_card_message_id(task.task_id, "om-card")
        workflow.task_store.mark_waiting(
            task.task_id,
            "tool_call",
            "om-card",
            lambda _task_id, _node_name: None,
            "实时数据结果",
        )
        workflow.task_store.confirm_action(task.task_id, "tool_call", "next", source="test")

        called = False

        async def runner(_state):
            nonlocal called
            called = True
            raise AssertionError("runner should not be called during confirmation replay")

        result = await workflow._interactive_node(  # noqa: SLF001
            "tool_call",
            runner,
            {
                "task_id": task.task_id,
                "query": "数据库连接超时",
                "node_results": [],
                "retry_counts": {},
            },
        )

        assert called is False
        assert result["last_action"] == "next"
        assert result["node_result"] == "实时数据结果"
        assert result["node_results"] == [{"node_name": "tool_call", "result": "实时数据结果"}]
        assert sender.updates == []

    asyncio.run(run())


def test_interactive_topic_rag_uses_real_retriever() -> None:
    async def run() -> None:
        workflow = InteractiveTopicWorkflow(
            sender=FakeSender(),  # type: ignore[arg-type]
            retriever=FakeRetriever(),  # type: ignore[arg-type]
        )

        result, diagnosis_state = await workflow._rag_retrieve({"query": "订单消息堆积"})  # noqa: SLF001

        assert "文档/混合召回 Top 1 / 共命中 1 条" in result
        assert "消息堆积排查" in result
        assert "section=第一章_消息队列_1.1_消息堆积" in result
        assert "真实检索片段" in result
        assert diagnosis_state["retrieved_docs"]

    asyncio.run(run())


def test_interactive_topic_cache_check_uses_history_case_store() -> None:
    async def run() -> None:
        workflow = InteractiveTopicWorkflow(
            sender=FakeSender(),  # type: ignore[arg-type]
            case_store=FakeCaseStore(),  # type: ignore[arg-type]
        )

        result, diagnosis_state = await workflow._cache_check({"query": "CPU 飙升"})  # noqa: SLF001

        assert "命中 1 条可复用历史成功案例" in result
        assert "alert-case-1" in result
        assert "扩容 worker" in result
        assert diagnosis_state["cache_hit"] is True

    asyncio.run(run())


def test_interactive_topic_rag_displays_top2_only() -> None:
    async def run() -> None:
        workflow = InteractiveTopicWorkflow(
            sender=FakeSender(),  # type: ignore[arg-type]
            retriever=MultiDocRetriever(),  # type: ignore[arg-type]
        )

        result, diagnosis_state = await workflow._rag_retrieve({"query": "CPU 飙升"})  # noqa: SLF001

        assert "文档/混合召回 Top 2 / 共命中 3 条" in result
        assert "doc-1" in result
        assert "doc-2" in result
        assert "doc-3" not in result
        assert len(diagnosis_state["retrieved_docs"]) == 3

    asyncio.run(run())


def test_interactive_topic_rag_also_shows_history_cases() -> None:
    async def run() -> None:
        workflow = InteractiveTopicWorkflow(
            sender=FakeSender(),  # type: ignore[arg-type]
            retriever=FakeRetriever(),  # type: ignore[arg-type]
            case_store=FakeCaseStore(),  # type: ignore[arg-type]
        )

        result, diagnosis_state = await workflow._rag_retrieve({"query": "CPU 飙升"})  # noqa: SLF001

        assert "文档/混合召回 Top 1 / 共命中 1 条" in result
        assert "历史案例召回 Top 1" in result
        assert "alert-case-1" in result
        assert "扩容 worker" in result
        assert len(diagnosis_state["retrieved_docs"]) == 2

    asyncio.run(run())


def test_interactive_topic_summary_uses_llm_context() -> None:
    async def run() -> None:
        llm = FakeLLM()
        workflow = InteractiveTopicWorkflow(
            sender=FakeSender(),  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
        )

        result = await workflow._summary(  # noqa: SLF001
            {
                "query": "数据库连接超时",
                "node_results": [
                    {"node_name": "rag_retrieve", "result": "RAG 命中数据库连接池排查文档"},
                    {"node_name": "tool_call", "result": "实时数据发现 timeout 错误率升高"},
                ],
            }
        )

        assert result == "LLM 诊断总结"
        assert "数据库连接超时" in llm.prompts[0]
        assert "rag_retrieve" in llm.prompts[0]
        assert "tool_call" in llm.prompts[0]
        assert "实时数据发现 timeout 错误率升高" in llm.prompts[0]

    asyncio.run(run())


def test_interactive_topic_summary_prompt_forbids_json_and_requires_markdown_sections() -> None:
    prompt = _build_summary_prompt(
        {
            "query": "数据库连接超时",
            "node_results": [
                {"node_name": "rag_retrieve", "result": "RAG 命中数据库连接池排查文档"},
                {"node_name": "tool_call", "result": "实时数据发现 timeout 错误率升高"},
            ],
        }
    )

    assert "不要输出 JSON" in prompt
    assert "不要输出代码块" in prompt
    assert "## 故障判断" in prompt
    assert "## 关键证据" in prompt
    assert "## 处置建议" in prompt
    assert "## 风险与观察项" in prompt
    assert "当前证据不足" in prompt


def test_interactive_topic_final_text_only_shows_summary_result() -> None:
    workflow = InteractiveTopicWorkflow(sender=FakeSender())  # type: ignore[arg-type]

    result = workflow._build_final_text(  # noqa: SLF001
        {
            "node_results": [
                {"node_name": "cache_check", "result": "cache details should be hidden"},
                {"node_name": "rag_retrieve", "result": "rag details should be hidden"},
                {"node_name": "tool_call", "result": "tool details should be hidden"},
                {"node_name": "summary", "result": "final diagnosis only"},
            ]
        }
    )

    assert result == "final diagnosis only"
    assert "rag details" not in result
    assert "tool details" not in result


def test_interactive_topic_feedback_valid_saves_case() -> None:
    async def run() -> None:
        sender = FakeSender()
        case_store = FakeCaseStore()
        workflow = InteractiveTopicWorkflow(
            sender=sender,  # type: ignore[arg-type]
            case_store=case_store,  # type: ignore[arg-type]
        )
        task = workflow.task_store.create_task("topic-1", "oc-chat", "om-root", "CPU 飙升")
        workflow.task_store.set_card_message_id(task.task_id, "om-card")
        workflow.task_store.mark_feedback_waiting(task.task_id, "om-card")
        app = workflow.compile()
        await app.aupdate_state(
            {"configurable": {"thread_id": task.task_id}, "recursion_limit": 50},
            {
                "query": "CPU 飙升",
                "task_id": task.task_id,
                "node_results": [
                    {"node_name": "summary", "result": "建议扩容 worker"},
                ],
            },
        )

        result = await workflow.handle_card_callback(
            {"value": {"task_id": task.task_id, "node_name": "feedback_learning", "action": "feedback_valid"}}
        )

        assert result == {"status": "ok", "saved_case_id": "saved-case-1"}
        assert case_store.saved_state["recommended_plan"]["summary"] == "建议扩容 worker"
        assert case_store.saved_state["alert_summary"] == "CPU 飙升"
        assert "raw_text" not in case_store.saved_state["raw_alert"]
        assert sender.updates[-1]["feedback_buttons"] is False
        assert "已存入历史案例知识库" in sender.updates[-1]["current_result"]

    asyncio.run(run())


def test_interactive_topic_feedback_save_failure_can_retry() -> None:
    async def run() -> None:
        sender = FakeSender()
        workflow = InteractiveTopicWorkflow(
            sender=sender,  # type: ignore[arg-type]
            case_store=FailingCaseStore(),  # type: ignore[arg-type]
        )
        task = workflow.task_store.create_task("topic-2", "oc-chat", "om-root", "CPU 飙升")
        workflow.task_store.set_card_message_id(task.task_id, "om-card")
        workflow.task_store.mark_feedback_waiting(task.task_id, "om-card")
        app = workflow.compile()
        await app.aupdate_state(
            {"configurable": {"thread_id": task.task_id}, "recursion_limit": 50},
            {
                "query": "CPU 飙升",
                "task_id": task.task_id,
                "node_results": [
                    {"node_name": "summary", "result": "建议扩容 worker"},
                ],
            },
        )

        result = await workflow.handle_card_callback(
            {"value": {"task_id": task.task_id, "node_name": "feedback_learning", "action": "feedback_valid"}}
        )

        assert result == {"status": "error", "reason": "save_failed"}
        assert workflow.task_store.get_task(task.task_id).status == "pending_feedback"
        assert sender.updates[-1]["feedback_buttons"] is True
        assert "入库失败" in sender.updates[-1]["current_result"]

    asyncio.run(run())
