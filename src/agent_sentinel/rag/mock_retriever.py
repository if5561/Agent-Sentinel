from __future__ import annotations

import logging

from agent_sentinel.rag.models import RagFilters, RetrievedDoc

logger = logging.getLogger(__name__)


class MockRetriever:
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 在未接入真实向量库时返回空检索结果，让诊断流程可以继续走 mock 或本地演示。
        logger.info("Mock RAG retrieval skipped query_chars=%s", len(query))
        # TODO: Replace with a real vector store or hybrid retrieval backend.
        return []
