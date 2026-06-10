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
class StaticDocRetriever:
    milvus: MilvusVectorClient
    embedding: EmbeddingClient
    collection_name: str
    top_k: int = 8
    weight: float = 0.55
    recall_top_k: int | None = None
    pruning_enabled: bool = True
    pruning_config: dict | None = None

    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 方法说明：从固定知识库中检索与告警相关的排障文档，并按配置权重写入最终分数。
        started = time.perf_counter()
        recall_top_k = self.recall_top_k or self.top_k
        logger.info(
            "Static doc RAG start collection=%s query_chars=%s recall_top_k=%s prune_top_k=%s pruning_enabled=%s",
            self.collection_name,
            len(query or ""),
            recall_top_k,
            self.top_k,
            self.pruning_enabled,
        )
        query_embedding = await self.embedding.embed(query)
        expr = _build_static_expr(filters)
        docs = await self.milvus.search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=recall_top_k,
            source_type="static_doc",
            expr=expr,
        )
        recalled_count = len(docs)
        docs = self._prune(query, docs)
        for doc in docs:
            score = float(doc.metadata.get("rerank_score") or doc.metadata.get("combined_score") or doc.score)
            doc.weighted_score = score * self.weight
        logger.info(
            "Static doc RAG completed collection=%s recalled=%s returned=%s scores=%s elapsed_ms=%s",
            self.collection_name,
            recalled_count,
            len(docs),
            _format_doc_scores(docs),
            int((time.perf_counter() - started) * 1000),
        )
        return docs

    def _prune(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        # 方法说明：对固定知识召回结果做二次筛选，把最相关的少量文档留给模型阅读。
        if not self.pruning_enabled or len(docs) <= self.top_k:
            logger.info(
                "Static doc RAG pruning skipped recalled=%s top_k=%s pruning_enabled=%s",
                len(docs),
                self.top_k,
                self.pruning_enabled,
            )
            return docs
        candidates = [_doc_to_candidate(index, doc, "static") for index, doc in enumerate(docs)]
        pruned = prune_by_differential_strategy(
            query,
            candidates,
            top_k=self.top_k,
            config=self.pruning_config,
        )
        logger.info("Static doc RAG pruning completed before=%s after=%s", len(docs), len(pruned))
        return [_apply_pruning_scores(docs[int(item["_index"])], item) for item in pruned]


def _build_static_expr(filters: RagFilters | None) -> str | None:
    # 方法说明：生成固定知识的服务过滤条件，优先查当前服务，同时保留全局通用知识。
    if not filters or not filters.service:
        return None
    return f'service == "{filters.service}" or service == "global"'


def _doc_to_candidate(index: int, doc: RetrievedDoc, source_type: str) -> dict:
    # 方法说明：把固定知识文档转换成剪枝候选项，并记录原始位置方便剪枝后取回。
    return {
        "_index": index,
        "text": doc.text,
        "score": doc.score,
        "metadata": doc.metadata,
        "source_type": source_type,
    }


def _apply_pruning_scores(doc: RetrievedDoc, candidate: dict) -> RetrievedDoc:
    # 方法说明：把重排分数写回文档元数据，最终排序时可以使用更准确的相关性分数。
    metadata = dict(doc.metadata)
    for key in ("metadata_score", "combined_score", "rerank_score"):
        if key in candidate:
            metadata[key] = candidate[key]
    doc.metadata = metadata
    return doc


def _format_doc_scores(docs: list[RetrievedDoc], limit: int = 5) -> str:
    # 方法说明：把固定知识文档的编号和分数整理成日志，便于检查召回质量。
    if not docs:
        return "[]"
    values = []
    for doc in docs[:limit]:
        score = doc.metadata.get("rerank_score") or doc.metadata.get("combined_score") or doc.score
        values.append(f"{doc.id}:{float(score):.4f}")
    suffix = ", ..." if len(docs) > limit else ""
    return "[" + ", ".join(values) + suffix + "]"
