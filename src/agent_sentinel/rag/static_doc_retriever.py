from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.milvus_client import MilvusVectorClient
from agent_sentinel.rag.models import RagFilters, RetrievedDoc
from agent_sentinel.rag.pruning import prune_by_differential_strategy

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 StaticDocRetriever 组件，集中管理这个模块的状态和行为。
class StaticDocRetriever:
    # 执行当前业务步骤，推动流程继续向下游推进。
    milvus: MilvusVectorClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    embedding: EmbeddingClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    collection_name: str
    # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
    top_k: int = 8
    # 将 weight 的值保存下来，供后续流程判断或组装响应时使用。
    weight: float = 0.55
    # 将 recall_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    recall_top_k: int | None = None
    # 将 pruning_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    pruning_enabled: bool = True
    # 将 pruning_config 的值保存下来，供后续流程判断或组装响应时使用。
    pruning_config: dict | None = None

    # 定义 retrieve 相关的处理逻辑，供流程或外部调用复用。
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 从固定知识库中检索与告警相关的排障文档，并按配置权重写入最终分数。
        started = time.perf_counter()
        # 将 recall_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        recall_top_k = self.recall_top_k or self.top_k
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Static doc RAG start collection=%s query_chars=%s recall_top_k=%s prune_top_k=%s pruning_enabled=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.collection_name,
            # 调用 len 完成当前步骤需要的业务处理。
            len(query or ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            recall_top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.pruning_enabled,
        )
        # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
        query_embedding = await self.embedding.embed(query)
        # 将 expr 的值保存下来，供后续流程判断或组装响应时使用。
        expr = _build_static_expr(filters)
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs = await self.milvus.search(
            # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
            collection_name=self.collection_name,
            # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
            query_embedding=query_embedding,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=recall_top_k,
            # 将 source_type 的值保存下来，供后续流程判断或组装响应时使用。
            source_type="static_doc",
            # 将 expr 的值保存下来，供后续流程判断或组装响应时使用。
            expr=expr,
        )
        # 将 recalled_count 的值保存下来，供后续流程判断或组装响应时使用。
        recalled_count = len(docs)
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs = self._prune(query, docs)
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for doc in docs:
            # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
            score = float(doc.metadata.get("rerank_score") or doc.metadata.get("combined_score") or doc.score)
            # 将 doc.weighted_score 的值保存下来，供后续流程判断或组装响应时使用。
            doc.weighted_score = score * self.weight
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Static doc RAG completed collection=%s recalled=%s returned=%s scores=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.collection_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            recalled_count,
            # 调用 len 完成当前步骤需要的业务处理。
            len(docs),
            # 调用 _format_doc_scores 完成当前步骤需要的业务处理。
            _format_doc_scores(docs),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return docs

    # 定义 _prune 相关的处理逻辑，供流程或外部调用复用。
    def _prune(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        # 对固定知识召回结果做二次筛选，把最相关的少量文档留给模型阅读。
        if not self.pruning_enabled or len(docs) <= self.top_k:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Static doc RAG pruning skipped recalled=%s top_k=%s pruning_enabled=%s",
                # 调用 len 完成当前步骤需要的业务处理。
                len(docs),
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.top_k,
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.pruning_enabled,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return docs
        # 将 candidates 的值保存下来，供后续流程判断或组装响应时使用。
        candidates = [_doc_to_candidate(index, doc, "static") for index, doc in enumerate(docs)]
        # 将 pruned 的值保存下来，供后续流程判断或组装响应时使用。
        pruned = prune_by_differential_strategy(
            # 执行当前业务步骤，推动流程继续向下游推进。
            query,
            # 执行当前业务步骤，推动流程继续向下游推进。
            candidates,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=self.top_k,
            # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
            config=self.pruning_config,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Static doc RAG pruning completed before=%s after=%s", len(docs), len(pruned))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [_apply_pruning_scores(docs[int(item["_index"])], item) for item in pruned]


# 定义 _build_static_expr 相关的处理逻辑，供流程或外部调用复用。
def _build_static_expr(filters: RagFilters | None) -> str | None:
    # 生成固定知识的服务过滤条件，优先查当前服务，同时保留全局通用知识。
    if not filters or not filters.service:
        return None
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return f'service == "{filters.service}" or service == "global"'


# 定义 _doc_to_candidate 相关的处理逻辑，供流程或外部调用复用。
def _doc_to_candidate(index: int, doc: RetrievedDoc, source_type: str) -> dict:
    # 把固定知识文档转换成剪枝候选项，并记录原始位置方便剪枝后取回。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "_index": index,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "text": doc.text,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "score": doc.score,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "metadata": doc.metadata,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "source_type": source_type,
    }


# 定义 _apply_pruning_scores 相关的处理逻辑，供流程或外部调用复用。
def _apply_pruning_scores(doc: RetrievedDoc, candidate: dict) -> RetrievedDoc:
    # 把重排分数写回文档元数据，最终排序时可以使用更准确的相关性分数。
    metadata = dict(doc.metadata)
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for key in ("metadata_score", "combined_score", "rerank_score"):
        # 根据 key in candidate 判断当前流程该进入哪个处理分支。
        if key in candidate:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            metadata[key] = candidate[key]
    # 将 doc.metadata 的值保存下来，供后续流程判断或组装响应时使用。
    doc.metadata = metadata
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return doc


# 定义 _format_doc_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_doc_scores(docs: list[RetrievedDoc], limit: int = 5) -> str:
    # 把固定知识文档的编号和分数整理成日志，便于检查召回质量。
    if not docs:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
    values = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for doc in docs[:limit]:
        # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
        score = doc.metadata.get("rerank_score") or doc.metadata.get("combined_score") or doc.score
        # 把当前结果追加到集合中，逐步构建最终输出。
        values.append(f"{doc.id}:{float(score):.4f}")
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(docs) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(values) + suffix + "]"
