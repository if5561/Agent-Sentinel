from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# 定义 ChatRequest 组件，集中管理这个模块的状态和行为。
class ChatRequest(BaseModel):
    # 将 message 的值保存下来，供后续流程判断或组装响应时使用。
    message: str = Field(..., min_length=1, description="User input for a single-turn conversation.")


# 定义 ChatResponse 组件，集中管理这个模块的状态和行为。
class ChatResponse(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    answer: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    model: str


# 定义 HealthResponse 组件，集中管理这个模块的状态和行为。
class HealthResponse(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    status: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    app: str


# 定义 AlertReportRequest 组件，集中管理这个模块的状态和行为。
class AlertReportRequest(BaseModel):
    # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
    source: str = Field(..., min_length=1, description="Alert source, for example app, job, or service name.")
    # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
    level: Literal["INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
    summary: str = Field(..., min_length=1, description="Short summary for the alert.")
    # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
    details: str | None = Field(default=None, description="Optional detailed context.")
    # 将 dedupe_key 的值保存下来，供后续流程判断或组装响应时使用。
    dedupe_key: str | None = Field(default=None, description="Optional dedupe key for short-window suppression.")
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags: list[str] = Field(default_factory=list, description="Optional tags for grouping and filtering.")


# 定义 AlertReportResponse 组件，集中管理这个模块的状态和行为。
class AlertReportResponse(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    status: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    dispatched: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    deduplicated: bool


# 定义 AlertRecordResponse 组件，集中管理这个模块的状态和行为。
class AlertRecordResponse(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    source: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    level: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    summary: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    details: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    dedupe_key: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    tags: list[str]
    # 执行当前业务步骤，推动流程继续向下游推进。
    created_at: str


# 定义 AlertAnalyzeRequest 组件，集中管理这个模块的状态和行为。
class AlertAnalyzeRequest(BaseModel):
    # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
    chat_id: str = Field(..., min_length=1, description="Feishu chat_id to send the analysis result to.")
    # 将 message_id 的值保存下来，供后续流程判断或组装响应时使用。
    message_id: str | None = Field(
        # 将 default 的值保存下来，供后续流程判断或组装响应时使用。
        default=None,
        # 将 description 的值保存下来，供后续流程判断或组装响应时使用。
        description="Optional original Feishu message_id.",
    )
    # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
    thread_root_message_id: str | None = Field(
        # 将 default 的值保存下来，供后续流程判断或组装响应时使用。
        default=None,
        # 将 description 的值保存下来，供后续流程判断或组装响应时使用。
        description="Optional Feishu root_id for native thread reply. Falls back to message_id when omitted.",
    )
    # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
    source: str = Field(..., min_length=1, description="Alert source such as service or bot name.")
    # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
    level: Literal["INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
    summary: str = Field(..., min_length=1, description="Short summary of the alert.")
    # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
    details: str | None = Field(default=None, description="Optional detailed context.")
    # 将 raw_text 的值保存下来，供后续流程判断或组装响应时使用。
    raw_text: str | None = Field(default=None, description="Optional raw alert message text.")
    # 将 trigger_type 的值保存下来，供后续流程判断或组装响应时使用。
    trigger_type: str = Field(default="bot_alert", description="Trigger source such as bot_alert or user_message.")
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags: list[str] = Field(default_factory=list, description="Optional tags for grouping and filtering.")
    # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
    mention_open_id: str | None = Field(default=None, description="Optional open_id to mention in the reply.")
    # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
    mention_name: str | None = Field(default=None, description="Optional display name for the mentioned user.")


# 定义 AlertAnalyzeResponse 组件，集中管理这个模块的状态和行为。
class AlertAnalyzeResponse(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    status: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    analysis: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    sent_to_feishu: bool


# 定义 FeishuEventHeader 组件，集中管理这个模块的状态和行为。
class FeishuEventHeader(BaseModel):
    # 将 event_type 的值保存下来，供后续流程判断或组装响应时使用。
    event_type: str | None = None
    # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
    token: str | None = None


# 定义 FeishuEventEnvelope 组件，集中管理这个模块的状态和行为。
class FeishuEventEnvelope(BaseModel):
    # 将 schema_ 的值保存下来，供后续流程判断或组装响应时使用。
    schema_: str | None = Field(default=None, alias="schema")
    # 将 header 的值保存下来，供后续流程判断或组装响应时使用。
    header: FeishuEventHeader | None = None
    # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
    event: dict[str, object] | None = None
    # 将 challenge 的值保存下来，供后续流程判断或组装响应时使用。
    challenge: str | None = None
    # 将 type 的值保存下来，供后续流程判断或组装响应时使用。
    type: str | None = None
    # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
    token: str | None = None
