from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from agent_sentinel.config import Settings
from agent_sentinel.graph.state import DiagnosisState
from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.milvus_client import MilvusSearchConfig, MilvusVectorClient
from agent_sentinel.rag.models import RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoryCaseStore:
    milvus: MilvusVectorClient
    embedding: EmbeddingClient
    collection_name: str
    similarity_threshold: float = 0.85

    async def search_similar_cases(
        self,
        alert_text: str,
        *,
        top_k: int = 2,
        threshold: float | None = None,
    ) -> list[RetrievedDoc]:
        # 方法说明：从配置的后端或数据集中检索匹配内容。
        if not alert_text.strip():
            logger.info("History case search skipped empty alert_text")
            return []
        started = time.perf_counter()
        min_score = self.similarity_threshold if threshold is None else threshold
        logger.info(
            "History case search start collection=%s alert_chars=%s top_k=%s threshold=%s",
            self.collection_name,
            len(alert_text),
            top_k,
            min_score,
        )
        query_embedding = await self.embedding.embed(alert_text)
        # 历史案例库只召回已确认有效的 alert_case，避免普通知识文档进入缓存复用判断。
        docs = await self.milvus.search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            source_type="message_history",
            expr='doc_type == "alert_case"',
        )
        # 阈值过滤在业务层完成，便于不同场景动态调整“可复用案例”的相似度门槛。
        filtered = [doc for doc in docs if doc.score >= min_score]
        logger.info(
            "History case search completed collection=%s recalled=%s filtered=%s threshold=%s scores=%s elapsed_ms=%s",
            self.collection_name,
            len(docs),
            len(filtered),
            min_score,
            _format_scores([doc.score for doc in filtered]),
            int((time.perf_counter() - started) * 1000),
        )
        return filtered

    async def save_case_to_history(self, state: DiagnosisState) -> str:
        # 方法说明：把一次被确认有效的诊断结果保存成历史案例，后续相似告警可直接参考。
        started = time.perf_counter()
        alert_text = build_alert_text(state.get("raw_alert", {}))
        if not alert_text.strip():
            alert_text = str(state.get("alert_summary") or "AIOps alert case")
        # Milvus VARCHAR 限制按字节计算，中文内容必须按 UTF-8 字节安全截断。
        alert_text = _truncate_utf8(alert_text, 65535)
        embedding = await self.embedding.embed(alert_text)
        now = int(time.time())
        raw_alert = state.get("raw_alert", {})
        metadata = {
            "alert_summary": state.get("alert_summary", ""),
            "retrieved_docs": state.get("retrieved_docs", []),
            "live_data": state.get("live_data", {}),
            "recommended_plan": state.get("recommended_plan", {}),
            "evidence": state.get("evidence", []),
            "validation_result": state.get("validation_result", False),
            "need_human": state.get("need_human", False),
            "human_decision": state.get("human_decision", ""),
            "final_text": state.get("final_text", ""),
            "full_workflow_log": state.get("messages", []),
        }
        title = _truncate_utf8(str(raw_alert.get("summary") or state.get("alert_summary") or "AIOps alert case"), 512)
        tags = _extract_tags(state)
        # case_id 使用告警文本和时间生成，避免同一秒内不同案例发生主键冲突。
        case_id = f"alert-case-{hashlib.sha256(f'{alert_text}:{now}'.encode('utf-8')).hexdigest()[:24]}"
        record = {
            "id": case_id,
            "text": alert_text,
            "embedding": embedding,
            "doc_type": "alert_case",
            "title": title,
            "service": _truncate_utf8(str(raw_alert.get("service") or raw_alert.get("source") or ""), 128),
            "component": _truncate_utf8(str(raw_alert.get("component") or ""), 128),
            "tags": _truncate_utf8(",".join(tags), 1024),
            "version": "v1",
            "updated_at": now,
            "created_at": now,
            "source_uri": _truncate_utf8(str(raw_alert.get("source_uri") or ""), 1024),
            "source": _truncate_utf8(str(raw_alert.get("source") or "aiops"), 1024),
            "metadata": _metadata_json(metadata),
        }
        await self.milvus.ensure_static_doc_collection(self.collection_name, len(embedding))
        await self.milvus.upsert(self.collection_name, [record])
        logger.info(
            "History case saved collection=%s case_id=%s alert_chars=%s metadata_bytes=%s elapsed_ms=%s",
            self.collection_name,
            case_id,
            len(alert_text),
            len(record["metadata"].encode("utf-8")),
            int((time.perf_counter() - started) * 1000),
        )
        return case_id


def build_history_case_store(settings: Settings) -> HistoryCaseStore | None:
    # 方法说明：根据配置创建历史案例存储；未启用 Milvus 时返回空值表示跳过案例缓存。
    if settings.rag_provider.strip().lower() != "milvus" or not settings.milvus_uri:
        # 只有启用 Milvus 且配置 URI 时才创建历史案例存储，mock 模式下直接跳过缓存复用。
        return None
    milvus = MilvusVectorClient(
        MilvusSearchConfig(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            user=settings.milvus_user,
            password=settings.milvus_password,
            db_name=settings.milvus_db_name,
        )
    )
    embedding = EmbeddingClient(
        api_key=settings.embedding_api_key or "",
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        mock_enabled=settings.embedding_mock_enabled,
    )
    return HistoryCaseStore(
        milvus=milvus,
        embedding=embedding,
        collection_name=settings.rag_message_collection,
        similarity_threshold=settings.rag_case_cache_threshold,
    )


def build_alert_text(raw_alert: dict[str, Any]) -> str:
    # 用摘要、详情、原文拼接成相似案例检索文本，尽量保留告警上下文而不是只用标题。
    # 方法说明：把告警摘要、详情和原文拼成检索文本，用来和历史案例做相似度匹配。
    parts = [
        str(raw_alert.get("summary") or ""),
        str(raw_alert.get("details") or ""),
        str(raw_alert.get("raw_text") or ""),
    ]
    if not any(part.strip() for part in parts):
        parts.append(json.dumps(raw_alert, ensure_ascii=False, sort_keys=True))
    return "\n".join(part for part in parts if part.strip())


def final_plan_from_case(doc: RetrievedDoc) -> dict[str, Any]:
    # 方法说明：从历史案例 metadata 中恢复可复用方案，兼容不同版本保存过的字段名。
    metadata = doc.metadata or {}
    # 历史数据可能来自不同版本 schema，这里兼容多种字段名恢复可复用方案。
    plan = metadata.get("recommended_plan") or metadata.get("final_plan") or metadata.get("final_result")
    if isinstance(plan, dict):
        return plan
    if isinstance(plan, str) and plan.strip():
        return {"summary": plan}
    final_text = metadata.get("final_text")
    if isinstance(final_text, str) and final_text.strip():
        return {"summary": final_text}
    return {"summary": doc.text[:1200]}


def case_to_dict(doc: RetrievedDoc) -> dict[str, Any]:
    # 方法说明：把检索到的历史案例转换成飞书卡片和工作流状态都容易消费的字典。
    return {
        "id": doc.id,
        "title": doc.title or "",
        "text": doc.text,
        "score": doc.score,
        "source_uri": doc.source_uri or "",
        "metadata": doc.metadata,
        "final_plan": final_plan_from_case(doc),
    }


def _extract_tags(state: DiagnosisState) -> list[str]:
    # 方法说明：从告警标签和证据文本中提取案例标签，便于后续检索和筛选。
    raw_alert = state.get("raw_alert", {})
    tags = [str(tag).strip() for tag in raw_alert.get("tags", []) if str(tag).strip()]
    for item in state.get("evidence", []):
        for token in ("MQ", "CPU", "Redis", "MySQL", "JVM", "超时", "堆积", "死锁"):
            if token in item and token not in tags:
                tags.append(token)
    return tags[:10]


def _truncate_utf8(value: str, max_bytes: int) -> str:
    # 方法说明：按 UTF-8 字节数安全截断字符串，避免中文写入 Milvus 时超出字段限制。
    data = value.encode("utf-8")
    if len(data) <= max_bytes:
        return value
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _metadata_json(metadata: dict[str, Any]) -> str:
    # 方法说明：把案例元数据压缩成 Milvus 可存储的 JSON，超长时保留关键诊断信息。
    raw = json.dumps(metadata, ensure_ascii=False)
    if len(raw.encode("utf-8")) <= 8192:
        return raw

    # 超过 Milvus 字段限制时保留关键决策信息，丢弃完整日志等大字段。
    compact = {
        "alert_summary": _truncate_utf8(str(metadata.get("alert_summary") or ""), 700),
        "retrieved_docs": [
            _truncate_utf8(str(item), 900)
            for item in list(metadata.get("retrieved_docs") or [])[:2]
        ],
        "live_data": metadata.get("live_data", {}),
        "recommended_plan": metadata.get("recommended_plan", {}),
        "evidence": [
            _truncate_utf8(str(item), 500)
            for item in list(metadata.get("evidence") or [])[:8]
        ],
        "validation_result": metadata.get("validation_result", False),
        "need_human": metadata.get("need_human", False),
        "human_decision": metadata.get("human_decision", ""),
        "final_text": _truncate_utf8(str(metadata.get("final_text") or ""), 1800),
        "full_workflow_log": "truncated",
    }
    raw = json.dumps(compact, ensure_ascii=False)
    if len(raw.encode("utf-8")) <= 8192:
        return raw
    return json.dumps(
        {"truncated": _truncate_utf8(raw, 7800)},
        ensure_ascii=False,
    )


def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 方法说明：把历史案例相似度分数格式化成日志短文本。
    if not scores:
        return "[]"
    suffix = ", ..." if len(scores) > limit else ""
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + suffix + "]"
