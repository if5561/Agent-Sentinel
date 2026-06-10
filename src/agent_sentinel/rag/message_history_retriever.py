from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.milvus_client import MilvusVectorClient
from agent_sentinel.rag.models import RagFilters, RetrievedDoc
from agent_sentinel.rag.pruning import prune_by_differential_strategy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MessageHistoryRetriever:
    milvus: MilvusVectorClient
    embedding: EmbeddingClient
    collection_name: str
    top_k: int = 12
    weight: float = 0.45
    default_days: int = 30
    prune_top_k: int | None = 1
    pruning_enabled: bool = True
    pruning_config: dict | None = None

    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 方法说明：从配置的后端或数据集中检索匹配内容。
        started = time.perf_counter()
        logger.info(
            "Message history RAG start collection=%s query_chars=%s recall_top_k=%s prune_top_k=%s pruning_enabled=%s default_days=%s",
            self.collection_name,
            len(query or ""),
            self.top_k,
            self.prune_top_k,
            self.pruning_enabled,
            self.default_days,
        )
        query_embedding = await self.embedding.embed(query)
        expr = self._build_message_expr(filters)
        docs = await self.milvus.search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=self.top_k,
            source_type="message_history",
            expr=expr,
        )
        recalled_count = len(docs)
        docs = self._prune(query, docs)
        now = int(time.time())
        for doc in docs:
            recency_boost = self._recency_boost(now, doc.created_at)
            score = float(doc.metadata.get("combined_score") or doc.score)
            doc.weighted_score = score * self.weight * recency_boost
        logger.info(
            "Message history RAG completed collection=%s recalled=%s returned=%s scores=%s elapsed_ms=%s",
            self.collection_name,
            recalled_count,
            len(docs),
            _format_doc_scores(docs),
            int((time.perf_counter() - started) * 1000),
        )
        return docs

    def _prune(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        top_k = self.prune_top_k or self.top_k
        if not self.pruning_enabled or len(docs) <= top_k:
            logger.info(
                "Message history RAG pruning skipped recalled=%s top_k=%s pruning_enabled=%s",
                len(docs),
                top_k,
                self.pruning_enabled,
            )
            return docs
        candidates = [_doc_to_candidate(index, doc, "history") for index, doc in enumerate(docs)]
        pruned = prune_by_differential_strategy(
            query,
            candidates,
            top_k=top_k,
            config=self.pruning_config,
        )
        logger.info("Message history RAG pruning completed before=%s after=%s", len(docs), len(pruned))
        return [_apply_pruning_scores(docs[int(item["_index"])], item) for item in pruned]

    def _build_message_expr(self, filters: RagFilters | None) -> str | None:
        # 方法说明：构建并返回调用方需要的对象。
        clauses: list[str] = ['doc_type == "alert_case"']
        if self.default_days > 0:
            min_created_at = int(time.time()) - self.default_days * 86400
            clauses.append(f"created_at >= {min_created_at}")
        return " and ".join(clauses) if clauses else None

    def _recency_boost(self, now: int, created_at: int | None) -> float:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if not created_at:
            return 1.0
        age_days = max((now - created_at) / 86400, 0)
        if age_days <= 7:
            return 1.2
        if age_days <= 30:
            return 1.0
        return 0.8


def _doc_to_candidate(index: int, doc: RetrievedDoc, source_type: str) -> dict:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    return {
        "_index": index,
        "text": doc.text,
        "score": doc.score,
        "metadata": doc.metadata,
        "source_type": source_type,
    }


def _apply_pruning_scores(doc: RetrievedDoc, candidate: dict) -> RetrievedDoc:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    metadata = dict(doc.metadata)
    for key in ("metadata_score", "combined_score", "rerank_score"):
        if key in candidate:
            metadata[key] = candidate[key]
    doc.metadata = metadata
    return doc


def _format_doc_scores(docs: list[RetrievedDoc], limit: int = 5) -> str:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if not docs:
        return "[]"
    values = []
    for doc in docs[:limit]:
        score = doc.metadata.get("combined_score") or doc.score
        values.append(f"{doc.id}:{float(score):.4f}")
    suffix = ", ..." if len(docs) > limit else ""
    return "[" + ", ".join(values) + suffix + "]"
