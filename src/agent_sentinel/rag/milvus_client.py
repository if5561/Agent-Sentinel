from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.rag.models import RetrievedDoc, SourceType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MilvusSearchConfig:
    uri: str
    token: str | None = None
    user: str | None = None
    password: str | None = None
    db_name: str | None = None
    timeout_seconds: float = 5.0


class MilvusVectorClient:
    def __init__(self, config: MilvusSearchConfig) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.config = config
        self._client: Any | None = None

    async def search(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
        source_type: SourceType,
        expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[RetrievedDoc]:
        # 方法说明：从配置的后端或数据集中检索匹配内容。
        if not self.config.uri:
            logger.info("Milvus URI missing; search skipped collection=%s", collection_name)
            return []
        logger.info(
            "Milvus search start collection=%s source_type=%s top_k=%s expr=%s embedding_dim=%s",
            collection_name,
            source_type,
            top_k,
            expr or "",
            len(query_embedding),
        )
        return await asyncio.to_thread(
            # pymilvus 客户端是同步接口，放到线程中执行，避免阻塞 FastAPI/工作流事件循环。
            self._search_sync,
            collection_name,
            query_embedding,
            top_k,
            source_type,
            expr,
            output_fields,
        )

    async def ensure_static_doc_collection(self, collection_name: str, dimension: int) -> None:
        # 方法说明：校验输入或状态是否满足继续处理的条件。
        await asyncio.to_thread(self._ensure_static_doc_collection_sync, collection_name, dimension)

    async def upsert(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if not records:
            return
        try:
            await asyncio.to_thread(self._upsert_sync, collection_name, records)
        except Exception:
            monitor.record_error("milvus_error")
            raise

    def _search_sync(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
        source_type: SourceType,
        expr: str | None,
        output_fields: list[str] | None,
    ) -> list[RetrievedDoc]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        started = time.perf_counter()
        client = self._get_client()
        fields = output_fields or [
            "id",
            "text",
            "source",
            "service",
            "level",
            "tags",
            "created_at",
            "metadata",
            "title",
            "source_uri",
            "embedding",
        ]
        try:
            results = client.search(
                collection_name=collection_name,
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                filter=expr,
                output_fields=fields,
                search_params={"metric_type": "COSINE", "params": {}},
                timeout=self.config.timeout_seconds,
            )
        except Exception:
            monitor.record_error("milvus_error")
            # 兼容不同集合 schema：默认字段失败后，移除可能不存在的字段再试一次。
            fallback_fields = [
                "id",
                "text",
                "source",
                "service",
                "tags",
                "created_at",
                "metadata",
                "title",
                "source_uri",
                "embedding",
            ]
            if output_fields is not None or fields == fallback_fields:
                logger.exception("Milvus search failed collection=%s", collection_name)
                return []
            logger.warning(
                "Milvus search failed with default output fields; retrying with schema-safe fields collection=%s",
                collection_name,
            )
            try:
                results = client.search(
                    collection_name=collection_name,
                    data=[query_embedding],
                    anns_field="embedding",
                    limit=top_k,
                    filter=expr,
                    output_fields=fallback_fields,
                    search_params={"metric_type": "COSINE", "params": {}},
                    timeout=self.config.timeout_seconds,
                )
            except Exception:
                logger.exception("Milvus search fallback failed collection=%s", collection_name)
                return []

        docs: list[RetrievedDoc] = []
        for hit in results[0] if results else []:
            docs.append(_hit_to_doc(hit, source_type))
        logger.info(
            "Milvus search completed collection=%s source_type=%s docs=%s scores=%s elapsed_ms=%s",
            collection_name,
            source_type,
            len(docs),
            _format_scores([doc.score for doc in docs]),
            int((time.perf_counter() - started) * 1000),
        )
        return docs

    def _ensure_static_doc_collection_sync(self, collection_name: str, dimension: int) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        client = self._get_client()
        if client.has_collection(collection_name):
            logger.info("Milvus collection already exists collection=%s", collection_name)
            return

        try:
            from pymilvus import DataType
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("pymilvus is required to initialize Milvus collections.") from exc

        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        # 显式字段约束与 history case 写入逻辑保持一致，避免 VARCHAR 超长导致 upsert 失败。
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="service", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="component", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="updated_at", datatype=DataType.INT64)
        schema.add_field(field_name="created_at", datatype=DataType.INT64)
        schema.add_field(field_name="source_uri", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=8192)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Milvus static doc collection created collection=%s dimension=%s", collection_name, dimension)

    def _upsert_sync(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        started = time.perf_counter()
        client = self._get_client()
        client.upsert(collection_name=collection_name, data=records)
        logger.info("Milvus records upserted collection=%s count=%s elapsed_ms=%s", collection_name, len(records), int((time.perf_counter() - started) * 1000))

    def _get_client(self) -> Any:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if self._client is not None:
            return self._client

        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("pymilvus is required when RAG_PROVIDER=milvus.") from exc

        kwargs: dict[str, Any] = {"uri": self.config.uri}
        if self.config.token:
            kwargs["token"] = self.config.token
        if self.config.user:
            kwargs["user"] = self.config.user
        if self.config.password:
            kwargs["password"] = self.config.password
        if self.config.db_name:
            kwargs["db_name"] = self.config.db_name

        self._client = MilvusClient(**kwargs)
        return self._client


def _hit_to_doc(hit: Any, source_type: SourceType) -> RetrievedDoc:
    # pymilvus 不同版本可能返回 dict 或对象形态，这里统一转换成 RetrievedDoc。
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if isinstance(hit, dict):
        entity = hit.get("entity") or {}
        raw_score = hit.get("score", hit.get("distance", 0.0))
        raw_id = hit.get("id", "")
    else:
        entity = getattr(hit, "entity", None) or {}
        raw_score = getattr(hit, "score", getattr(hit, "distance", 0.0))
        raw_id = getattr(hit, "id", "")
    if not isinstance(entity, dict):
        entity = dict(entity)
    metadata = _parse_metadata(entity.get("metadata"))
    doc_id = str(entity.get("id") or raw_id or metadata.get("id", ""))
    return RetrievedDoc(
        id=doc_id,
        text=str(entity.get("text") or ""),
        source_type=source_type,
        score=float(raw_score or 0.0),
        service=_empty_to_none(entity.get("service")),
        title=_empty_to_none(entity.get("title")),
        created_at=_to_int_or_none(entity.get("created_at")),
        source_uri=_empty_to_none(entity.get("source_uri") or entity.get("source")),
        metadata=metadata,
        embedding=[float(item) for item in entity.get("embedding", [])],
    )


def _parse_metadata(value: Any) -> dict[str, Any]:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            # metadata 正常是 JSON 字符串；解析失败时保留原文，方便后续排查脏数据。
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw": value}
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def _empty_to_none(value: Any) -> str | None:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    text = str(value or "").strip()
    return text or None


def _to_int_or_none(value: Any) -> int | None:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    if not scores:
        return "[]"
    suffix = ", ..." if len(scores) > limit else ""
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + suffix + "]"
