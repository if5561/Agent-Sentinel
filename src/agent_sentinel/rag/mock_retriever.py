from __future__ import annotations

import logging

from agent_sentinel.rag.models import RagFilters, RetrievedDoc

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 MockRetriever 组件，集中管理这个模块的状态和行为。
class MockRetriever:
    # 定义 retrieve 相关的处理逻辑，供流程或外部调用复用。
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 在未接入真实向量库时返回空检索结果，让诊断流程可以继续走 mock 或本地演示。
        logger.info("Mock RAG retrieval skipped query_chars=%s", len(query))
        # TODO: Replace with a real vector store or hybrid retrieval backend.
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
