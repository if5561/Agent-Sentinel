from __future__ import annotations

from typing import Protocol

from agent_sentinel.rag.models import RagFilters, RetrievedDoc


class BaseRetriever(Protocol):
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 方法说明：定义所有检索器都要实现的统一入口，输入查询文本和过滤条件，返回检索文档列表。
        ...
