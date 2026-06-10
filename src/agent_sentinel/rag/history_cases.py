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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 HistoryCaseStore 组件，集中管理这个模块的状态和行为。
class HistoryCaseStore:
    # 执行当前业务步骤，推动流程继续向下游推进。
    milvus: MilvusVectorClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    embedding: EmbeddingClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    collection_name: str
    # 将 similarity_threshold 的值保存下来，供后续流程判断或组装响应时使用。
    similarity_threshold: float = 0.85

    # 定义 search_similar_cases 相关的处理逻辑，供流程或外部调用复用。
    async def search_similar_cases(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        alert_text: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
        top_k: int = 2,
        # 将 threshold 的值保存下来，供后续流程判断或组装响应时使用。
        threshold: float | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[RetrievedDoc]:
        # 在历史案例库中查找与当前告警足够相似的已验证案例，用于快速复用处理方案。
        if not alert_text.strip():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("History case search skipped empty alert_text")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 将 min_score 的值保存下来，供后续流程判断或组装响应时使用。
        min_score = self.similarity_threshold if threshold is None else threshold
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "History case search start collection=%s alert_chars=%s top_k=%s threshold=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.collection_name,
            # 调用 len 完成当前步骤需要的业务处理。
            len(alert_text),
            # 执行当前业务步骤，推动流程继续向下游推进。
            top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            min_score,
        )
        # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
        query_embedding = await self.embedding.embed(alert_text)
        # 历史案例库只召回已确认有效的 alert_case，避免普通知识文档进入缓存复用判断。
        docs = await self.milvus.search(
            # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
            collection_name=self.collection_name,
            # 将 query_embedding 的值保存下来，供后续流程判断或组装响应时使用。
            query_embedding=query_embedding,
            # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
            top_k=top_k,
            # 将 source_type 的值保存下来，供后续流程判断或组装响应时使用。
            source_type="message_history",
            # 将 expr 的值保存下来，供后续流程判断或组装响应时使用。
            expr='doc_type == "alert_case"',
        )
        # 阈值过滤在业务层完成，便于不同场景动态调整“可复用案例”的相似度门槛。
        filtered = [doc for doc in docs if doc.score >= min_score]
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "History case search completed collection=%s recalled=%s filtered=%s threshold=%s scores=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.collection_name,
            # 调用 len 完成当前步骤需要的业务处理。
            len(docs),
            # 调用 len 完成当前步骤需要的业务处理。
            len(filtered),
            # 执行当前业务步骤，推动流程继续向下游推进。
            min_score,
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores([doc.score for doc in filtered]),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return filtered

    # 定义 save_case_to_history 相关的处理逻辑，供流程或外部调用复用。
    async def save_case_to_history(self, state: DiagnosisState) -> str:
        # 把一次被确认有效的诊断结果保存成历史案例，后续相似告警可直接参考。
        started = time.perf_counter()
        # 将 alert_text 的值保存下来，供后续流程判断或组装响应时使用。
        alert_text = build_alert_text(state.get("raw_alert", {}))
        # 根据 not alert_text.strip() 判断当前流程该进入哪个处理分支。
        if not alert_text.strip():
            # 将 alert_text 的值保存下来，供后续流程判断或组装响应时使用。
            alert_text = str(state.get("alert_summary") or "AIOps alert case")
        # Milvus VARCHAR 限制按字节计算，中文内容必须按 UTF-8 字节安全截断。
        alert_text = _truncate_utf8(alert_text, 65535)
        # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
        embedding = await self.embedding.embed(alert_text)
        # 将 now 的值保存下来，供后续流程判断或组装响应时使用。
        now = int(time.time())
        # 将 raw_alert 的值保存下来，供后续流程判断或组装响应时使用。
        raw_alert = state.get("raw_alert", {})
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = {
            # 调用 state.get 完成当前步骤需要的业务处理。
            "alert_summary": state.get("alert_summary", ""),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "retrieved_docs": state.get("retrieved_docs", []),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "live_data": state.get("live_data", {}),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "recommended_plan": state.get("recommended_plan", {}),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "evidence": state.get("evidence", []),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "validation_result": state.get("validation_result", False),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "need_human": state.get("need_human", False),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "human_decision": state.get("human_decision", ""),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "final_text": state.get("final_text", ""),
            # 调用 state.get 完成当前步骤需要的业务处理。
            "full_workflow_log": state.get("messages", []),
        }
        # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
        title = _truncate_utf8(str(raw_alert.get("summary") or state.get("alert_summary") or "AIOps alert case"), 512)
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags = _extract_tags(state)
        # case_id 使用告警文本和时间生成，避免同一秒内不同案例发生主键冲突。
        case_id = f"alert-case-{hashlib.sha256(f'{alert_text}:{now}'.encode('utf-8')).hexdigest()[:24]}"
        # 将 record 的值保存下来，供后续流程判断或组装响应时使用。
        record = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "id": case_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text": alert_text,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "embedding": embedding,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "doc_type": "alert_case",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title": title,
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            "service": _truncate_utf8(str(raw_alert.get("service") or raw_alert.get("source") or ""), 128),
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            "component": _truncate_utf8(str(raw_alert.get("component") or ""), 128),
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            "tags": _truncate_utf8(",".join(tags), 1024),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "version": "v1",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "updated_at": now,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "created_at": now,
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            "source_uri": _truncate_utf8(str(raw_alert.get("source_uri") or ""), 1024),
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            "source": _truncate_utf8(str(raw_alert.get("source") or "aiops"), 1024),
            # 调用 _metadata_json 完成当前步骤需要的业务处理。
            "metadata": _metadata_json(metadata),
        }
        # 等待异步操作完成，再继续推进当前业务流程。
        await self.milvus.ensure_static_doc_collection(self.collection_name, len(embedding))
        # 等待异步操作完成，再继续推进当前业务流程。
        await self.milvus.upsert(self.collection_name, [record])
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "History case saved collection=%s case_id=%s alert_chars=%s metadata_bytes=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.collection_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            case_id,
            # 调用 len 完成当前步骤需要的业务处理。
            len(alert_text),
            # 调用 len 完成当前步骤需要的业务处理。
            len(record["metadata"].encode("utf-8")),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return case_id


# 定义 build_history_case_store 相关的处理逻辑，供流程或外部调用复用。
def build_history_case_store(settings: Settings) -> HistoryCaseStore | None:
    # 根据配置创建历史案例存储；未启用 Milvus 时返回空值表示跳过案例缓存。
    if settings.rag_provider.strip().lower() != "milvus" or not settings.milvus_uri:
        # 只有启用 Milvus 且配置 URI 时才创建历史案例存储，mock 模式下直接跳过缓存复用。
        return None
    # 将 milvus 的值保存下来，供后续流程判断或组装响应时使用。
    milvus = MilvusVectorClient(
        # 调用 MilvusSearchConfig 完成当前步骤需要的业务处理。
        MilvusSearchConfig(
            # 将 uri 的值保存下来，供后续流程判断或组装响应时使用。
            uri=settings.milvus_uri,
            # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
            token=settings.milvus_token,
            # 将 user 的值保存下来，供后续流程判断或组装响应时使用。
            user=settings.milvus_user,
            # 将 password 的值保存下来，供后续流程判断或组装响应时使用。
            password=settings.milvus_password,
            # 将 db_name 的值保存下来，供后续流程判断或组装响应时使用。
            db_name=settings.milvus_db_name,
        )
    )
    # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
    embedding = EmbeddingClient(
        # 将 api_key 的值保存下来，供后续流程判断或组装响应时使用。
        api_key=settings.embedding_api_key or "",
        # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
        base_url=settings.embedding_base_url,
        # 将 model 的值保存下来，供后续流程判断或组装响应时使用。
        model=settings.embedding_model,
        # 将 dimension 的值保存下来，供后续流程判断或组装响应时使用。
        dimension=settings.embedding_dimension,
        # 将 mock_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        mock_enabled=settings.embedding_mock_enabled,
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return HistoryCaseStore(
        # 将 milvus 的值保存下来，供后续流程判断或组装响应时使用。
        milvus=milvus,
        # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
        embedding=embedding,
        # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
        collection_name=settings.rag_message_collection,
        # 将 similarity_threshold 的值保存下来，供后续流程判断或组装响应时使用。
        similarity_threshold=settings.rag_case_cache_threshold,
    )


# 定义 build_alert_text 相关的处理逻辑，供流程或外部调用复用。
def build_alert_text(raw_alert: dict[str, Any]) -> str:
    # 用摘要、详情、原文拼接成相似案例检索文本，尽量保留告警上下文而不是只用标题。
    # 把告警摘要、详情和原文拼成检索文本，用来和历史案例做相似度匹配。
    parts = [
        # 调用 str 完成当前步骤需要的业务处理。
        str(raw_alert.get("summary") or ""),
        # 调用 str 完成当前步骤需要的业务处理。
        str(raw_alert.get("details") or ""),
        # 调用 str 完成当前步骤需要的业务处理。
        str(raw_alert.get("raw_text") or ""),
    ]
    # 根据 not any(part.strip() for part in parts) 判断当前流程该进入哪个处理分支。
    if not any(part.strip() for part in parts):
        # 把当前结果追加到集合中，逐步构建最终输出。
        parts.append(json.dumps(raw_alert, ensure_ascii=False, sort_keys=True))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "\n".join(part for part in parts if part.strip())


# 定义 final_plan_from_case 相关的处理逻辑，供流程或外部调用复用。
def final_plan_from_case(doc: RetrievedDoc) -> dict[str, Any]:
    # 从历史案例 metadata 中恢复可复用方案，兼容不同版本保存过的字段名。
    metadata = doc.metadata or {}
    # 历史数据可能来自不同版本 schema，这里兼容多种字段名恢复可复用方案。
    plan = metadata.get("recommended_plan") or metadata.get("final_plan") or metadata.get("final_result")
    # 根据 isinstance(plan, dict) 判断当前流程该进入哪个处理分支。
    if isinstance(plan, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return plan
    # 根据 isinstance(plan, str) and plan.strip() 判断当前流程该进入哪个处理分支。
    if isinstance(plan, str) and plan.strip():
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"summary": plan}
    # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
    final_text = metadata.get("final_text")
    # 根据 isinstance(final_text, str) and final_text.strip() 判断当前流程该进入哪个处理分支。
    if isinstance(final_text, str) and final_text.strip():
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"summary": final_text}
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {"summary": doc.text[:1200]}


# 定义 case_to_dict 相关的处理逻辑，供流程或外部调用复用。
def case_to_dict(doc: RetrievedDoc) -> dict[str, Any]:
    # 把检索到的历史案例转换成飞书卡片和工作流状态都容易消费的字典。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "id": doc.id,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "title": doc.title or "",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "text": doc.text,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "score": doc.score,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "source_uri": doc.source_uri or "",
        # 执行当前业务步骤，推动流程继续向下游推进。
        "metadata": doc.metadata,
        # 调用 final_plan_from_case 完成当前步骤需要的业务处理。
        "final_plan": final_plan_from_case(doc),
    }


# 定义 _extract_tags 相关的处理逻辑，供流程或外部调用复用。
def _extract_tags(state: DiagnosisState) -> list[str]:
    # 从告警标签和证据文本中提取案例标签，便于后续检索和筛选。
    raw_alert = state.get("raw_alert", {})
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags = [str(tag).strip() for tag in raw_alert.get("tags", []) if str(tag).strip()]
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for item in state.get("evidence", []):
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for token in ("MQ", "CPU", "Redis", "MySQL", "JVM", "超时", "堆积", "死锁"):
            # 根据 token in item and token not in tags 判断当前流程该进入哪个处理分支。
            if token in item and token not in tags:
                # 把当前结果追加到集合中，逐步构建最终输出。
                tags.append(token)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return tags[:10]


# 定义 _truncate_utf8 相关的处理逻辑，供流程或外部调用复用。
def _truncate_utf8(value: str, max_bytes: int) -> str:
    # 按 UTF-8 字节数安全截断字符串，避免中文写入 Milvus 时超出字段限制。
    data = value.encode("utf-8")
    # 根据 len(data) <= max_bytes 判断当前流程该进入哪个处理分支。
    if len(data) <= max_bytes:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return value
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return data[:max_bytes].decode("utf-8", errors="ignore")


# 定义 _metadata_json 相关的处理逻辑，供流程或外部调用复用。
def _metadata_json(metadata: dict[str, Any]) -> str:
    # 把案例元数据压缩成 Milvus 可存储的 JSON，超长时保留关键诊断信息。
    raw = json.dumps(metadata, ensure_ascii=False)
    # 根据 len(raw.encode("utf-8")) <= 8192 判断当前流程该进入哪个处理分支。
    if len(raw.encode("utf-8")) <= 8192:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return raw

    # 超过 Milvus 字段限制时保留关键决策信息，丢弃完整日志等大字段。
    compact = {
        # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
        "alert_summary": _truncate_utf8(str(metadata.get("alert_summary") or ""), 700),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "retrieved_docs": [
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            _truncate_utf8(str(item), 900)
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for item in list(metadata.get("retrieved_docs") or [])[:2]
        ],
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        "live_data": metadata.get("live_data", {}),
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        "recommended_plan": metadata.get("recommended_plan", {}),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "evidence": [
            # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
            _truncate_utf8(str(item), 500)
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for item in list(metadata.get("evidence") or [])[:8]
        ],
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        "validation_result": metadata.get("validation_result", False),
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        "need_human": metadata.get("need_human", False),
        # 调用 metadata.get 完成当前步骤需要的业务处理。
        "human_decision": metadata.get("human_decision", ""),
        # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
        "final_text": _truncate_utf8(str(metadata.get("final_text") or ""), 1800),
        # 执行当前业务步骤，推动流程继续向下游推进。
        "full_workflow_log": "truncated",
    }
    # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
    raw = json.dumps(compact, ensure_ascii=False)
    # 根据 len(raw.encode("utf-8")) <= 8192 判断当前流程该进入哪个处理分支。
    if len(raw.encode("utf-8")) <= 8192:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return raw
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return json.dumps(
        # 调用 _truncate_utf8 完成当前步骤需要的业务处理。
        {"truncated": _truncate_utf8(raw, 7800)},
        # 将 ensure_ascii 的值保存下来，供后续流程判断或组装响应时使用。
        ensure_ascii=False,
    )


# 定义 _format_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 把历史案例相似度分数格式化成日志短文本。
    if not scores:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(scores) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + suffix + "]"
