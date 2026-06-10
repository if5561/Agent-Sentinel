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
        # 方法说明：从历史告警案例集合里检索相似处理记录，并按新近程度和权重调整排序。
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
        # 方法说明：对历史消息召回结果做二次筛选，减少相似但不够有用的旧案例进入上下文。
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
        # 方法说明：生成 Milvus 过滤表达式，只检索告警案例，并限制在最近一段时间内。
        clauses: list[str] = ['doc_type == "alert_case"']
        if self.default_days > 0:
            min_created_at = int(time.time()) - self.default_days * 86400
            clauses.append(f"created_at >= {min_created_at}")
        return " and ".join(clauses) if clauses else None

    def _recency_boost(self, now: int, created_at: int | None) -> float:
        # 方法说明：按案例新旧程度调整分数，让近期处理经验在排序里更容易靠前。
        if not created_at:
            return 1.0
        age_days = max((now - created_at) / 86400, 0)
        if age_days <= 7:
            return 1.2
        if age_days <= 30:
            return 1.0
        return 0.8


def _doc_to_candidate(index: int, doc: RetrievedDoc, source_type: str) -> dict:
    # 方法说明：把检索文档转换成剪枝算法需要的候选格式，并保留原始下标方便映射回文档。
    return {
        "_index": index,
        "text": doc.text,
        "score": doc.score,
        "metadata": doc.metadata,
        "source_type": source_type,
    }


def _apply_pruning_scores(doc: RetrievedDoc, candidate: dict) -> RetrievedDoc:
    # 方法说明：把剪枝或重排产生的新分数写回文档元数据，后续加权排序会继续使用。
    metadata = dict(doc.metadata)
    for key in ("metadata_score", "combined_score", "rerank_score"):
        if key in candidate:
            metadata[key] = candidate[key]
    doc.metadata = metadata
    return doc


def _format_doc_scores(docs: list[RetrievedDoc], limit: int = 5) -> str:
    # 方法说明：把历史案例的编号和分数整理成短日志，便于观察哪些案例被召回。
    if not docs:
        return "[]"
    values = []
    for doc in docs[:limit]:
        score = doc.metadata.get("combined_score") or doc.score
        values.append(f"{doc.id}:{float(score):.4f}")
    suffix = ", ..." if len(docs) > limit else ""
    return "[" + ", ".join(values) + suffix + "]"
