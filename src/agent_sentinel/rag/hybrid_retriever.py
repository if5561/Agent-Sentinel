from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from agent_sentinel.rag.base import BaseRetriever
from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.mmr import select_mmr
from agent_sentinel.rag.models import RagFilters, RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HybridRetriever:
    embedding: EmbeddingClient
    retrievers: list[BaseRetriever] = field(default_factory=list)
    final_top_k: int = 6
    mmr_lambda: float = 0.55

    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 方法说明：并行调用多个 RAG 检索器，再去重并用 MMR 选出最终要给模型看的上下文。
        if not self.retrievers:
            logger.info("Hybrid RAG skipped because no retrievers configured query_chars=%s", len(query or ""))
            return []

        started = time.perf_counter()
        logger.info(
            "Hybrid RAG start query_chars=%s retrievers=%s final_top_k=%s mmr_lambda=%s",
            len(query or ""),
            len(self.retrievers),
            self.final_top_k,
            self.mmr_lambda,
        )
        query_embedding = await self.embedding.embed(query)
        # 多个 retriever 并行召回，单路失败不影响其他知识源返回结果。
        results = await asyncio.gather(
            *(retriever.retrieve(query, filters) for retriever in self.retrievers),
            return_exceptions=True,
        )
        docs: list[RetrievedDoc] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Hybrid RAG retriever failed index=%s error=%s", index, result)
                continue
            logger.info("Hybrid RAG retriever result index=%s docs=%s scores=%s", index, len(result), _format_doc_scores(result))
            docs.extend(result)

        deduped = _dedupe_docs(docs)
        logger.info("Hybrid RAG dedupe completed before=%s after=%s", len(docs), len(deduped))
        # MMR 在相关性和多样性之间折中，避免最终上下文都来自高度重复的文档片段。
        selected = select_mmr(
            query_embedding=query_embedding,
            docs=deduped,
            top_k=self.final_top_k,
            lambda_mult=self.mmr_lambda,
        )
        logger.info(
            "Hybrid RAG completed candidates=%s deduped=%s selected=%s selected_scores=%s elapsed_ms=%s",
            len(docs),
            len(deduped),
            len(selected),
            _format_doc_scores(selected),
            int((time.perf_counter() - started) * 1000),
        )
        return selected


def _dedupe_docs(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    # 方法说明：合并不同检索源返回的重复文档，只保留同一内容里得分最高的一条。
    by_key: dict[str, RetrievedDoc] = {}
    for doc in docs:
        # 不同检索源可能召回同一文档，按 source_type + id/text 前缀做轻量去重。
        key = f"{doc.source_type}:{doc.id or doc.text[:80]}"
        existing = by_key.get(key)
        if existing is None or doc.weighted_score > existing.weighted_score:
            by_key[key] = doc
    return list(by_key.values())


def _format_doc_scores(docs: list[RetrievedDoc], limit: int = 5) -> str:
    # 方法说明：把候选文档的来源、编号和分数压缩成日志字符串，方便排查检索排序。
    if not docs:
        return "[]"
    values = [f"{doc.source_type}:{doc.id}:{(doc.weighted_score or doc.score):.4f}" for doc in docs[:limit]]
    suffix = ", ..." if len(docs) > limit else ""
    return "[" + ", ".join(values) + suffix + "]"
