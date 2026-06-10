from __future__ import annotations

from typing import Protocol

from agent_sentinel.rag.models import RagFilters, RetrievedDoc


# 定义 BaseRetriever 组件，集中管理这个模块的状态和行为。
class BaseRetriever(Protocol):
    # 定义 retrieve 相关的处理逻辑，供流程或外部调用复用。
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 定义所有检索器都要实现的统一入口，输入查询文本和过滤条件，返回检索文档列表。
        ...
