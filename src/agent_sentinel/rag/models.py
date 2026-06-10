from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# 将 SourceType 的值保存下来，供后续流程判断或组装响应时使用。
SourceType = Literal["static_doc", "message_history", "mock"]


# 定义 RetrievedDoc 组件，集中管理这个模块的状态和行为。
class RetrievedDoc(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    text: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    source_type: SourceType
    # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
    score: float = 0.0
    # 将 weighted_score 的值保存下来，供后续流程判断或组装响应时使用。
    weighted_score: float = 0.0
    # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
    service: str | None = None
    # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
    title: str | None = None
    # 将 created_at 的值保存下来，供后续流程判断或组装响应时使用。
    created_at: int | None = None
    # 将 source_uri 的值保存下来，供后续流程判断或组装响应时使用。
    source_uri: str | None = None
    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
    embedding: list[float] = Field(default_factory=list, exclude=True)

    # 定义 to_prompt_text 相关的处理逻辑，供流程或外部调用复用。
    def to_prompt_text(self) -> str:
        # 把检索文档整理成适合放进提示词的文本，历史案例会额外带上根因和方案。
        label = "固定知识" if self.source_type == "static_doc" else "历史消息"
        # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
        title = self.title or self.id
        # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
        service = f" service={self.service}" if self.service else ""
        # 将 uri 的值保存下来，供后续流程判断或组装响应时使用。
        uri = f" source={self.source_uri}" if self.source_uri else ""
        # 根据 self.source_type == "message_history" 判断当前流程该进入哪个处理分支。
        if self.source_type == "message_history":
            # 将 plan 的值保存下来，供后续流程判断或组装响应时使用。
            plan = self.metadata.get("recommended_plan") or self.metadata.get("final_text") or ""
            # 将 root_cause 的值保存下来，供后续流程判断或组装响应时使用。
            root_cause = self.metadata.get("root_cause") or self.metadata.get("cause") or ""
            # 将 extra 的值保存下来，供后续流程判断或组装响应时使用。
            extra = []
            # 根据 root_cause 判断当前流程该进入哪个处理分支。
            if root_cause:
                # 把当前结果追加到集合中，逐步构建最终输出。
                extra.append(f"历史根因: {root_cause}")
            # 根据 plan 判断当前流程该进入哪个处理分支。
            if plan:
                # 把当前结果追加到集合中，逐步构建最终输出。
                extra.append(f"历史方案: {plan}")
            # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
            suffix = "\n" + "\n".join(extra) if extra else ""
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return f"[{label}] {title}{service}{uri}\n{self.text}{suffix}"
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f"[{label}] {title}{service}{uri}\n{self.text}"


# 定义 RagFilters 组件，集中管理这个模块的状态和行为。
class RagFilters(BaseModel):
    # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
    service: str | None = None
    # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
    level: str | None = None
    # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
    chat_id: str | None = None
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags: list[str] = Field(default_factory=list)
