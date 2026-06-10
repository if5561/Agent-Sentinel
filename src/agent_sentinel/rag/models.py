from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["static_doc", "message_history", "mock"]


class RetrievedDoc(BaseModel):
    id: str
    text: str
    source_type: SourceType
    score: float = 0.0
    weighted_score: float = 0.0
    service: str | None = None
    title: str | None = None
    created_at: int | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list, exclude=True)

    def to_prompt_text(self) -> str:
        # 方法说明：把检索文档整理成适合放进提示词的文本，历史案例会额外带上根因和方案。
        label = "固定知识" if self.source_type == "static_doc" else "历史消息"
        title = self.title or self.id
        service = f" service={self.service}" if self.service else ""
        uri = f" source={self.source_uri}" if self.source_uri else ""
        if self.source_type == "message_history":
            plan = self.metadata.get("recommended_plan") or self.metadata.get("final_text") or ""
            root_cause = self.metadata.get("root_cause") or self.metadata.get("cause") or ""
            extra = []
            if root_cause:
                extra.append(f"历史根因: {root_cause}")
            if plan:
                extra.append(f"历史方案: {plan}")
            suffix = "\n" + "\n".join(extra) if extra else ""
            return f"[{label}] {title}{service}{uri}\n{self.text}{suffix}"
        return f"[{label}] {title}{service}{uri}\n{self.text}"


class RagFilters(BaseModel):
    service: str | None = None
    level: str | None = None
    chat_id: str | None = None
    tags: list[str] = Field(default_factory=list)
