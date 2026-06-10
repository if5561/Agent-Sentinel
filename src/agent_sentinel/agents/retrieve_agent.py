from __future__ import annotations

import logging

from agent_sentinel.graph.state import DiagnosisState, append_evidence, append_message
from agent_sentinel.monitoring import monitor
from agent_sentinel.rag.base import BaseRetriever
from agent_sentinel.rag.models import RagFilters

logger = logging.getLogger(__name__)


async def retrieve_node(state: DiagnosisState, retriever: BaseRetriever) -> DiagnosisState:
    # 方法说明：根据告警上下文执行 RAG 检索，把相关知识和历史经验整理成模型可阅读的文本。
    logger.info("Node retrieve started")
    monitor.record_rag_retrieval("hybrid", state.get("chat_id"))
    # 从告警上下文里抽取业务过滤条件，避免只靠语义相似度召回无关文档。
    filters = _build_filters(state)
    docs = await retriever.retrieve(state.get("alert_summary", ""), filters)
    # 节点状态里保存面向 prompt 的文本，而不是保留 retriever 内部模型对象。
    prompt_docs = [doc.to_prompt_text() for doc in docs]
    logger.info("Node retrieve completed docs=%s", len(prompt_docs))
    return {
        "retrieved_docs": prompt_docs,
        "messages": append_message(state, "assistant", f"RAG 检索完成，召回 {len(prompt_docs)} 条上下文。"),
        "evidence": append_evidence(state, f"Hybrid RAG returned {len(prompt_docs)} documents."),
    }


def should_fetch(state: DiagnosisState) -> str:
    # 方法说明：根据告警摘要里的关键词判断是否需要进一步查询实时指标、日志和拓扑。
    text = f"{state.get('alert_summary', '')} {state.get('raw_alert', {})}".lower()
    keywords = ("error", "critical", "timeout", "latency", "失败", "超时", "异常", "错误")
    # 简单规则用于决定是否补充实时数据，避免低风险或信息不足的请求无谓调用外部工具。
    decision = "fetch" if any(keyword in text for keyword in keywords) else "skip"
    logger.info("Route should_fetch decision=%s", decision)
    return decision


def _build_filters(state: DiagnosisState) -> RagFilters:
    # 方法说明：从告警状态里整理检索过滤条件，让 RAG 更优先找同服务、同级别、同群组的资料。
    raw_alert = state.get("raw_alert", {})
    # RAG filter 是软边界：尽量按服务、级别、群组和标签缩小召回范围。
    return RagFilters(
        service=_clean(raw_alert.get("service") or raw_alert.get("source")),
        level=_clean(raw_alert.get("level")),
        chat_id=_clean(state.get("chat_id")),
        tags=[str(tag) for tag in raw_alert.get("tags", [])],
    )


def _clean(value: object) -> str | None:
    # 方法说明：把空字符串、None 等无效过滤值清掉，避免传给检索器造成误筛选。
    text = str(value or "").strip()
    return text or None
