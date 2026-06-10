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
# 定义 MessageHistoryRetriever 组件，集中管理这个模块的状态和行为。
class MessageHistoryRetriever:
    # 执行当前业务步骤，推动流程继续向下游推进。
    milvus: MilvusVectorClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    embedding: EmbeddingClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    collection_name: str
    # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
    top_k: int = 12
    # 将 weight 的值保存下来，供后续流程判断或组装响应时使用。
    weight: float = 0.45
    # 将 default_days 的值保存下来，供后续流程判断或组装响应时使用。
    default_days: int = 30
    # 将 prune_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    prune_top_k: int | None = 1
    # 将 pruning_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    pruning_enabled: bool = True
    # 将 pruning_config 的值保存下来，供后续流程判断或组装响应时使用。
    pruning_config: dict | None = None

    # 定义 retrieve 相关的处理逻辑，供流程或外部调用复用。
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 从历史告警案例集合里检索相似处理记录，并按新近程度和权重调整排序。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Message history RAG start collection=%s query_chars=%s recall_top_k=%s prune_top_k=%s pruning_enabled=%s default_days=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.collection_name,
            # 调用 len 完成当前步骤需要的业务处理。
            len(query or ""),
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.prune_top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.pruning_enabled,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.default_days,
        )
        # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
        query_embedding = await self.embedding.embed(query)
        # 将 expr 的值保存下来，供后续流程判断或组装响应时使用。
        expr = self._build_message_expr(filters)
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs = await self.milvus.search(
            # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
            collection_name=self.collection_name,
            # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
            query_embedding=query_embedding,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=self.top_k,
            # 将 source_type 的值保存下来，供后续流程判断或组装响应时使用。
            source_type="message_history",
            # 将 expr 的值保存下来，供后续流程判断或组装响应时使用。
            expr=expr,
        )
        # 将 recalled_count 的值保存下来，供后续流程判断或组装响应时使用。
        recalled_count = len(docs)
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs = self._prune(query, docs)
        # 将 now 的值保存下来，供后续流程判断或组装响应时使用。
        now = int(time.time())
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for doc in docs:
            # 将 recency_boost 的值保存下来，供后续流程判断或组装响应时使用。
            recency_boost = self._recency_boost(now, doc.created_at)
            # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
            score = float(doc.metadata.get("combined_score") or doc.score)
            # 将 doc.weighted_score 的值保存下来，供后续流程判断或组装响应时使用。
            doc.weighted_score = score * self.weight * recency_boost
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Message history RAG completed collection=%s recalled=%s returned=%s scores=%s elapsed_ms=%s",
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
        # 对历史消息召回结果做二次筛选，减少相似但不够有用的旧案例进入上下文。
        top_k = self.prune_top_k or self.top_k
        # 根据 not self.pruning_enabled or len(docs) <= top_k 判断当前流程该进入哪个处理分支。
        if not self.pruning_enabled or len(docs) <= top_k:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Message history RAG pruning skipped recalled=%s top_k=%s pruning_enabled=%s",
                # 调用 len 完成当前步骤需要的业务处理。
                len(docs),
                # 执行当前业务步骤，推动流程继续向下游推进。
                top_k,
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.pruning_enabled,
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return docs
        # 将 candidates 的值保存下来，供后续流程判断或组装响应时使用。
        candidates = [_doc_to_candidate(index, doc, "history") for index, doc in enumerate(docs)]
        # 将 pruned 的值保存下来，供后续流程判断或组装响应时使用。
        pruned = prune_by_differential_strategy(
            # 执行当前业务步骤，推动流程继续向下游推进。
            query,
            # 执行当前业务步骤，推动流程继续向下游推进。
            candidates,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=top_k,
            # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
            config=self.pruning_config,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Message history RAG pruning completed before=%s after=%s", len(docs), len(pruned))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [_apply_pruning_scores(docs[int(item["_index"])], item) for item in pruned]

    # 定义 _build_message_expr 相关的处理逻辑，供流程或外部调用复用。
    def _build_message_expr(self, filters: RagFilters | None) -> str | None:
        # 生成 Milvus 过滤表达式，只检索告警案例，并限制在最近一段时间内。
        clauses: list[str] = ['doc_type == "alert_case"']
        # 根据 self.default_days > 0 判断当前流程该进入哪个处理分支。
        if self.default_days > 0:
            # 将 min_created_at 的值保存下来，供后续流程判断或组装响应时使用。
            min_created_at = int(time.time()) - self.default_days * 86400
            # 把当前结果追加到集合中，逐步构建最终输出。
            clauses.append(f"created_at >= {min_created_at}")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return " and ".join(clauses) if clauses else None

    # 定义 _recency_boost 相关的处理逻辑，供流程或外部调用复用。
    def _recency_boost(self, now: int, created_at: int | None) -> float:
        # 按案例新旧程度调整分数，让近期处理经验在排序里更容易靠前。
        if not created_at:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return 1.0
        # 将 age_days 的值保存下来，供后续流程判断或组装响应时使用。
        age_days = max((now - created_at) / 86400, 0)
        # 根据 age_days <= 7 判断当前流程该进入哪个处理分支。
        if age_days <= 7:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return 1.2
        # 根据 age_days <= 30 判断当前流程该进入哪个处理分支。
        if age_days <= 30:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return 1.0
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.8


# 定义 _doc_to_candidate 相关的处理逻辑，供流程或外部调用复用。
def _doc_to_candidate(index: int, doc: RetrievedDoc, source_type: str) -> dict:
    # 把检索文档转换成剪枝算法需要的候选格式，并保留原始下标方便映射回文档。
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
    # 把剪枝或重排产生的新分数写回文档元数据，后续加权排序会继续使用。
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
    # 把历史案例的编号和分数整理成短日志，便于观察哪些案例被召回。
    if not docs:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
    values = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for doc in docs[:limit]:
        # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
        score = doc.metadata.get("combined_score") or doc.score
        # 把当前结果追加到集合中，逐步构建最终输出。
        values.append(f"{doc.id}:{float(score):.4f}")
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(docs) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(values) + suffix + "]"
