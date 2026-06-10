from __future__ import annotations

from typing import Protocol

from agent_sentinel.rag.models import RagFilters, RetrievedDoc


class BaseRetriever(Protocol):
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 方法说明：从配置的后端或数据集中检索匹配内容。
        ...
