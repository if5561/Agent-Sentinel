from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from agent_sentinel.rag.base import BaseRetriever
from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.mmr import select_mmr
from agent_sentinel.rag.models import RagFilters, RetrievedDoc

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 HybridRetriever 组件，集中管理这个模块的状态和行为。
class HybridRetriever:
    # 执行当前业务步骤，推动流程继续向下游推进。
    embedding: EmbeddingClient
    # 将 retrievers 的值保存下来，供后续流程判断或组装响应时使用。
    retrievers: list[BaseRetriever] = field(default_factory=list)
    # 将 final_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    final_top_k: int = 6
    # 将 mmr_lambda 的值保存下来，供后续流程判断或组装响应时使用。
    mmr_lambda: float = 0.55

    # 定义 retrieve 相关的处理逻辑，供流程或外部调用复用。
    async def retrieve(self, query: str, filters: RagFilters | None = None) -> list[RetrievedDoc]:
        # 并行调用多个 RAG 检索器，再去重并用 MMR 选出最终要给模型看的上下文。
        if not self.retrievers:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Hybrid RAG skipped because no retrievers configured query_chars=%s", len(query or ""))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []

        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Hybrid RAG start query_chars=%s retrievers=%s final_top_k=%s mmr_lambda=%s",
            # 调用 len 完成当前步骤需要的业务处理。
            len(query or ""),
            # 调用 len 完成当前步骤需要的业务处理。
            len(self.retrievers),
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.final_top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.mmr_lambda,
        )
        # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
        query_embedding = await self.embedding.embed(query)
        # 多个 retriever 并行召回，单路失败不影响其他知识源返回结果。
        results = await asyncio.gather(
            # 调用 retriever.retrieve 完成当前步骤需要的业务处理。
            *(retriever.retrieve(query, filters) for retriever in self.retrievers),
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return_exceptions=True,
        )
        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs: list[RetrievedDoc] = []
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for index, result in enumerate(results):
            # 根据 isinstance(result, Exception) 判断当前流程该进入哪个处理分支。
            if isinstance(result, Exception):
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.warning("Hybrid RAG retriever failed index=%s error=%s", index, result)
                # 跳过当前项剩余逻辑，继续处理下一项数据。
                continue
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Hybrid RAG retriever result index=%s docs=%s scores=%s", index, len(result), _format_doc_scores(result))
            # 把一组结果合并到集合中，扩展后续可使用的数据范围。
            docs.extend(result)

        # 将 deduped 的值保存下来，供后续流程判断或组装响应时使用。
        deduped = _dedupe_docs(docs)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Hybrid RAG dedupe completed before=%s after=%s", len(docs), len(deduped))
        # MMR 在相关性和多样性之间折中，避免最终上下文都来自高度重复的文档片段。
        selected = select_mmr(
            # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
            query_embedding=query_embedding,
            # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
            docs=deduped,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=self.final_top_k,
            # 将 lambda_mult 的值保存下来，供后续流程判断或组装响应时使用。
            lambda_mult=self.mmr_lambda,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Hybrid RAG completed candidates=%s deduped=%s selected=%s selected_scores=%s elapsed_ms=%s",
            # 调用 len 完成当前步骤需要的业务处理。
            len(docs),
            # 调用 len 完成当前步骤需要的业务处理。
            len(deduped),
            # 调用 len 完成当前步骤需要的业务处理。
            len(selected),
            # 调用 _format_doc_scores 完成当前步骤需要的业务处理。
            _format_doc_scores(selected),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return selected


# 定义 _dedupe_docs 相关的处理逻辑，供流程或外部调用复用。
def _dedupe_docs(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    # 合并不同检索源返回的重复文档，只保留同一内容里得分最高的一条。
    by_key: dict[str, RetrievedDoc] = {}
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for doc in docs:
        # 不同检索源可能召回同一文档，按 source_type + id/text 前缀做轻量去重。
        key = f"{doc.source_type}:{doc.id or doc.text[:80]}"
        # 将 existing 的值保存下来，供后续流程判断或组装响应时使用。
        existing = by_key.get(key)
        # 根据 existing is None or doc.weighted_score > existing.we... 判断当前流程该进入哪个处理分支。
        if existing is None or doc.weighted_score > existing.weighted_score:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            by_key[key] = doc
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return list(by_key.values())


# 定义 _format_doc_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_doc_scores(docs: list[RetrievedDoc], limit: int = 5) -> str:
    # 把候选文档的来源、编号和分数压缩成日志字符串，方便排查检索排序。
    if not docs:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
    values = [f"{doc.source_type}:{doc.id}:{(doc.weighted_score or doc.score):.4f}" for doc in docs[:limit]]
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(docs) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(values) + suffix + "]"
