from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input for a single-turn conversation.")


class ChatResponse(BaseModel):
    answer: str
    model: str


class HealthResponse(BaseModel):
    status: str
    app: str


class AlertReportRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Alert source, for example app, job, or service name.")
    level: Literal["INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    summary: str = Field(..., min_length=1, description="Short summary for the alert.")
    details: str | None = Field(default=None, description="Optional detailed context.")
    dedupe_key: str | None = Field(default=None, description="Optional dedupe key for short-window suppression.")
    tags: list[str] = Field(default_factory=list, description="Optional tags for grouping and filtering.")


class AlertReportResponse(BaseModel):
    status: str
    dispatched: bool
    deduplicated: bool


class AlertRecordResponse(BaseModel):
    source: str
    level: str
    summary: str
    details: str | None
    dedupe_key: str | None
    tags: list[str]
    created_at: str


class AlertAnalyzeRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, description="Feishu chat_id to send the analysis result to.")
    message_id: str | None = Field(
        default=None,
        description="Optional original Feishu message_id.",
    )
    thread_root_message_id: str | None = Field(
        default=None,
        description="Optional Feishu root_id for native thread reply. Falls back to message_id when omitted.",
    )
    source: str = Field(..., min_length=1, description="Alert source such as service or bot name.")
    level: Literal["INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    summary: str = Field(..., min_length=1, description="Short summary of the alert.")
    details: str | None = Field(default=None, description="Optional detailed context.")
    raw_text: str | None = Field(default=None, description="Optional raw alert message text.")
    trigger_type: str = Field(default="bot_alert", description="Trigger source such as bot_alert or user_message.")
    tags: list[str] = Field(default_factory=list, description="Optional tags for grouping and filtering.")
    mention_open_id: str | None = Field(default=None, description="Optional open_id to mention in the reply.")
    mention_name: str | None = Field(default=None, description="Optional display name for the mentioned user.")


class AlertAnalyzeResponse(BaseModel):
    status: str
    analysis: str
    sent_to_feishu: bool


class FeishuEventHeader(BaseModel):
    event_type: str | None = None
    token: str | None = None


class FeishuEventEnvelope(BaseModel):
    schema_: str | None = Field(default=None, alias="schema")
    header: FeishuEventHeader | None = None
    event: dict[str, object] | None = None
    challenge: str | None = None
    type: str | None = None
    token: str | None = None
