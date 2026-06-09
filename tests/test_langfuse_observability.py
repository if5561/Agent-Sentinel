from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_sentinel.config import LangfuseConfig
from agent_sentinel.llm.client import build_chat_model
from agent_sentinel.observability.langfuse import (
    LangfusePrompt,
    LangfusePromptService,
    _build_langfuse_metadata,
    render_prompt,
)


def test_render_prompt_supports_braces_and_escaped_variables() -> None:
    prompt = LangfusePrompt(
        name="understand",
        prompt_type="text",
        content="alert={{raw_alert}} legacy={raw_alert} escaped=\\{{raw_alert}}",
    )

    rendered = render_prompt(prompt, {"raw_alert": "cpu timeout"})

    assert rendered == "alert=cpu timeout legacy=cpu timeout escaped={{raw_alert}}"


def test_render_chat_prompt_supports_message_list() -> None:
    prompt = LangfusePrompt(
        name="chat",
        prompt_type="chat",
        content=[
            {"role": "system", "content": "service={{service}}"},
            {"role": "human", "content": "summary={{summary}}"},
        ],
    )

    rendered = render_prompt(prompt, {"service": "order", "summary": "timeout"})

    assert rendered == [
        {"role": "system", "content": "service=order"},
        {"role": "human", "content": "summary=timeout"},
    ]


def test_build_langfuse_metadata_keeps_business_trace_id_and_node_generation_name() -> None:
    prompt = LangfusePrompt(
        name="interactive_summary",
        prompt_type="text",
        content="summary={{query}}",
        version=3,
        prompt_id="prompt-123",
        label="production",
    )

    metadata = _build_langfuse_metadata(
        trace_context={
            "trace_id": "oc_chat_om_root",
            "workflow_type": "interactive_topic",
            "tags": ["agent-sentinel", "feishu"],
        },
        metadata={"node_name": "interactive_summary", "task_id": "topic-1"},
        prompt=prompt,
    )

    assert metadata["trace_id"] == "oc_chat_om_root"
    assert metadata["business_trace_id"] == "oc_chat_om_root"
    assert "langfuse_trace_id" not in metadata
    assert "traceparent" not in metadata
    assert metadata["otel_trace_source"] == "litellm"
    assert metadata["trace_name"] == "interactive_topic"
    assert metadata["generation_name"] == "interactive_summary"
    assert metadata["node_name"] == "interactive_summary"
    assert metadata["workflow_type"] == "interactive_topic"
    assert metadata["prompt_name"] == "interactive_summary"
    assert metadata["prompt_version"] == 3
    assert metadata["prompt_id"] == "prompt-123"
    assert metadata["prompt_label"] == "production"
    assert metadata["tags"] == [
        "agent-sentinel",
        "feishu",
        "interactive_topic",
        "prompt:interactive_summary",
    ]


class _FakeLLM:
    def __init__(self) -> None:
        self.extra_headers: dict[str, str] | None = None
        self.extra_body: dict | None = None

    async def ainvoke(self, input: object, config: dict | None = None, **kwargs: object) -> object:
        self.extra_headers = kwargs.get("extra_headers")  # type: ignore[assignment]
        self.extra_body = kwargs.get("extra_body")  # type: ignore[assignment]
        return "ok"


@pytest.mark.anyio
async def test_call_llm_skips_generation_prompt_linking_when_disabled() -> None:
    service = LangfusePromptService(
        LangfuseConfig(
            enabled=True,
            host="https://langfuse.example",
            public_key="pk",
            secret_key="sk",
            link_generations=False,
        )
    )

    async def fake_resolve_prompt(
        prompt_name: str | None,
        prompt_label: str | None,
        fallback_prompt: str,
    ) -> LangfusePrompt:
        return LangfusePrompt(
            name=prompt_name or "interactive_summary",
            prompt_type="text",
            content=fallback_prompt,
            version=1,
            prompt_id="prompt-123",
            label=prompt_label or "production",
        )

    scheduled = False

    def fake_schedule(coroutine: object) -> None:
        nonlocal scheduled
        scheduled = True

    service._resolve_prompt = fake_resolve_prompt  # type: ignore[method-assign]
    service._schedule = fake_schedule  # type: ignore[method-assign]

    try:
        llm = _FakeLLM()
        result = await service.call_llm_with_trace(
            llm=llm,
            fallback_prompt="hello {{name}}",
            prompt_name="interactive_summary",
            variables={"name": "Fe"},
            metadata={"trace_id": "business-1", "node_name": "interactive_summary"},
        )
    finally:
        await service.flush()

    assert result == "ok"
    assert scheduled is False
    assert llm.extra_headers is None
    assert llm.extra_body is not None
    assert llm.extra_body["metadata"]["business_trace_id"] == "business-1"
    assert llm.extra_body["metadata"]["otel_trace_source"] == "litellm"


@pytest.mark.anyio
async def test_remote_prompts_can_be_disabled_for_local_fallback() -> None:
    service = LangfusePromptService(
        LangfuseConfig(
            enabled=True,
            host="https://langfuse.example",
            public_key="pk",
            secret_key="sk",
            remote_prompts_enabled=False,
        )
    )

    async def fail_remote_prompt(name: str, label: str) -> LangfusePrompt | None:
        raise AssertionError("remote prompt fetch should be disabled")

    service.client.get_prompt = fail_remote_prompt  # type: ignore[method-assign]
    try:
        prompt = await service.get_prompt("missing-remote", fallback_prompt="local {{name}}")
    finally:
        await service.flush()

    assert render_prompt(prompt, {"name": "Fe"}) == "local Fe"


@pytest.mark.anyio
async def test_chat_openai_forwards_traceparent_and_litellm_metadata() -> None:
    captured: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            captured["path"] = self.path
            captured["headers"] = dict(self.headers)
            captured["body"] = json.loads(body.decode("utf-8"))
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        model = build_chat_model(
            api_key="sk-test",
            model="qwen-plus",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )

        await model.ainvoke(
            "hello",
            extra_headers={"traceparent": traceparent},
            extra_body={"metadata": {"business_trace_id": "business-1"}},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert captured["path"] == "/v1/chat/completions"
    assert captured["headers"]["traceparent"] == traceparent  # type: ignore[index]
    assert captured["body"]["metadata"] == {"business_trace_id": "business-1"}  # type: ignore[index]
