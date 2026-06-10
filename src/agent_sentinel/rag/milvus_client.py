from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.rag.models import RetrievedDoc, SourceType

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 MilvusSearchConfig 组件，集中管理这个模块的状态和行为。
class MilvusSearchConfig:
    # 执行当前业务步骤，推动流程继续向下游推进。
    uri: str
    # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
    token: str | None = None
    # 将 user 的值保存下来，供后续流程判断或组装响应时使用。
    user: str | None = None
    # 将 password 的值保存下来，供后续流程判断或组装响应时使用。
    password: str | None = None
    # 将 db_name 的值保存下来，供后续流程判断或组装响应时使用。
    db_name: str | None = None
    # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    timeout_seconds: float = 5.0


# 定义 MilvusVectorClient 组件，集中管理这个模块的状态和行为。
class MilvusVectorClient:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, config: MilvusSearchConfig) -> None:
        # 保存 Milvus 连接配置，并延迟创建客户端，避免程序启动时立即连接向量库。
        self.config = config
        # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
        self._client: Any | None = None

    # 定义 search 相关的处理逻辑，供流程或外部调用复用。
    async def search(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 执行当前业务步骤，推动流程继续向下游推进。
        collection_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        query_embedding: list[float],
        # 执行当前业务步骤，推动流程继续向下游推进。
        top_k: int,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source_type: SourceType,
        # 将 expr 的值保存下来，供后续流程判断或组装响应时使用。
        expr: str | None = None,
        # 将 output_fields 的值保存下来，供后续流程判断或组装响应时使用。
        output_fields: list[str] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[RetrievedDoc]:
        # 用查询向量在指定 Milvus 集合中做相似度搜索，并返回统一格式的检索文档。
        if not self.config.uri:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Milvus URI missing; search skipped collection=%s", collection_name)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Milvus search start collection=%s source_type=%s top_k=%s expr=%s embedding_dim=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            collection_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            source_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            expr or "",
            # 调用 len 完成当前步骤需要的业务处理。
            len(query_embedding),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return await asyncio.to_thread(
            # pymilvus 客户端是同步接口，放到线程中执行，避免阻塞 FastAPI/工作流事件循环。
            self._search_sync,
            # 执行当前业务步骤，推动流程继续向下游推进。
            collection_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            query_embedding,
            # 执行当前业务步骤，推动流程继续向下游推进。
            top_k,
            # 执行当前业务步骤，推动流程继续向下游推进。
            source_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            expr,
            # 执行当前业务步骤，推动流程继续向下游推进。
            output_fields,
        )

    # 定义 ensure_static_doc_collection 相关的处理逻辑，供流程或外部调用复用。
    async def ensure_static_doc_collection(self, collection_name: str, dimension: int) -> None:
        # 确保 Milvus 里有指定集合，没有时按向量维度创建一套可检索的 schema。
        await asyncio.to_thread(self._ensure_static_doc_collection_sync, collection_name, dimension)

    # 定义 upsert 相关的处理逻辑，供流程或外部调用复用。
    async def upsert(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        # 把已经带向量的文档或历史案例写入 Milvus，供后续相似度检索。
        if not records:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 等待异步操作完成，再继续推进当前业务流程。
            await asyncio.to_thread(self._upsert_sync, collection_name, records)
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("milvus_error")
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise

    # 定义 _search_sync 相关的处理逻辑，供流程或外部调用复用。
    def _search_sync(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        collection_name: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        query_embedding: list[float],
        # 执行当前业务步骤，推动流程继续向下游推进。
        top_k: int,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source_type: SourceType,
        # 执行当前业务步骤，推动流程继续向下游推进。
        expr: str | None,
        # 执行当前业务步骤，推动流程继续向下游推进。
        output_fields: list[str] | None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[RetrievedDoc]:
        # 在线程中执行 Milvus 同步搜索，并把返回结果转换成项目统一的检索文档。
        started = time.perf_counter()
        # 将 client 的值保存下来，供后续流程判断或组装响应时使用。
        client = self._get_client()
        # 将 fields 的值保存下来，供后续流程判断或组装响应时使用。
        fields = output_fields or [
            # 执行当前业务步骤，推动流程继续向下游推进。
            "id",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "service",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "level",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tags",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "created_at",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "metadata",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source_uri",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "embedding",
        ]
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 将 results 的值保存下来，供后续流程判断或组装响应时使用。
            results = client.search(
                # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
                collection_name=collection_name,
                # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
                data=[query_embedding],
                # 将 anns_field 的值保存下来，供后续流程判断或组装响应时使用。
                anns_field="embedding",
                # 将 limit 的值保存下来，供后续流程判断或组装响应时使用。
                limit=top_k,
                # 将 filter 的值保存下来，供后续流程判断或组装响应时使用。
                filter=expr,
                # 将 output_fields 的值保存下来，供后续流程判断或组装响应时使用。
                output_fields=fields,
                # 将 search_params 的值保存下来，供后续流程判断或组装响应时使用。
                search_params={"metric_type": "COSINE", "params": {}},
                # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
                timeout=self.config.timeout_seconds,
            )
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("milvus_error")
            # 兼容不同集合 schema：默认字段失败后，移除可能不存在的字段再试一次。
            fallback_fields = [
                # 执行当前业务步骤，推动流程继续向下游推进。
                "id",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "text",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "source",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "service",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "tags",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "created_at",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "metadata",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "title",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "source_uri",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "embedding",
            ]
            # 根据 output_fields is not None or fields == fallback_fiel... 判断当前流程该进入哪个处理分支。
            if output_fields is not None or fields == fallback_fields:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.exception("Milvus search failed collection=%s", collection_name)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return []
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Milvus search failed with default output fields; retrying with schema-safe fields collection=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                collection_name,
            )
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 results 的值保存下来，供后续流程判断或组装响应时使用。
                results = client.search(
                    # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
                    collection_name=collection_name,
                    # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
                    data=[query_embedding],
                    # 将 anns_field 的值保存下来，供后续流程判断或组装响应时使用。
                    anns_field="embedding",
                    # 将 limit 的值保存下来，供后续流程判断或组装响应时使用。
                    limit=top_k,
                    # 将 filter 的值保存下来，供后续流程判断或组装响应时使用。
                    filter=expr,
                    # 将 output_fields 的值保存下来，供后续流程判断或组装响应时使用。
                    output_fields=fallback_fields,
                    # 将 search_params 的值保存下来，供后续流程判断或组装响应时使用。
                    search_params={"metric_type": "COSINE", "params": {}},
                    # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
                    timeout=self.config.timeout_seconds,
                )
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.exception("Milvus search fallback failed collection=%s", collection_name)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return []

        # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
        docs: list[RetrievedDoc] = []
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for hit in results[0] if results else []:
            # 把当前结果追加到集合中，逐步构建最终输出。
            docs.append(_hit_to_doc(hit, source_type))
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Milvus search completed collection=%s source_type=%s docs=%s scores=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            collection_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            source_type,
            # 调用 len 完成当前步骤需要的业务处理。
            len(docs),
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores([doc.score for doc in docs]),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return docs

    # 定义 _ensure_static_doc_collection_sync 相关的处理逻辑，供流程或外部调用复用。
    def _ensure_static_doc_collection_sync(self, collection_name: str, dimension: int) -> None:
        # 同步创建静态文档集合，字段设计同时兼容知识文档和历史案例。
        client = self._get_client()
        # 根据 client.has_collection(collection_name) 判断当前流程该进入哪个处理分支。
        if client.has_collection(collection_name):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Milvus collection already exists collection=%s", collection_name)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            from pymilvus import DataType
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("pymilvus is required to initialize Milvus collections.") from exc

        # 将 schema 的值保存下来，供后续流程判断或组装响应时使用。
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        # 显式字段约束与 history case 写入逻辑保持一致，避免 VARCHAR 超长导致 upsert 失败。
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="service", datatype=DataType.VARCHAR, max_length=128)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="component", datatype=DataType.VARCHAR, max_length=128)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=1024)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=64)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="updated_at", datatype=DataType.INT64)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="created_at", datatype=DataType.INT64)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="source_uri", datatype=DataType.VARCHAR, max_length=1024)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=8192)

        # 将 index_params 的值保存下来，供后续流程判断或组装响应时使用。
        index_params = client.prepare_index_params()
        # 调用 index_params.add_index 完成当前步骤需要的业务处理。
        index_params.add_index(
            # 将 field_name 的值保存下来，供后续流程判断或组装响应时使用。
            field_name="embedding",
            # 将 index_type 的值保存下来，供后续流程判断或组装响应时使用。
            index_type="AUTOINDEX",
            # 将 metric_type 的值保存下来，供后续流程判断或组装响应时使用。
            metric_type="COSINE",
        )
        # 调用 client.create_collection 完成当前步骤需要的业务处理。
        client.create_collection(
            # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
            collection_name=collection_name,
            # 将 schema 的值保存下来，供后续流程判断或组装响应时使用。
            schema=schema,
            # 将 index_params 的值保存下来，供后续流程判断或组装响应时使用。
            index_params=index_params,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Milvus static doc collection created collection=%s dimension=%s", collection_name, dimension)

    # 定义 _upsert_sync 相关的处理逻辑，供流程或外部调用复用。
    def _upsert_sync(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        # 调用 Milvus upsert 写入记录，存在同 ID 时覆盖旧版本。
        started = time.perf_counter()
        # 将 client 的值保存下来，供后续流程判断或组装响应时使用。
        client = self._get_client()
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        client.upsert(collection_name=collection_name, data=records)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Milvus records upserted collection=%s count=%s elapsed_ms=%s", collection_name, len(records), int((time.perf_counter() - started) * 1000))

    # 定义 _get_client 相关的处理逻辑，供流程或外部调用复用。
    def _get_client(self) -> Any:
        # 按需创建并缓存 Milvus 客户端，避免每次检索都重新建立连接。
        if self._client is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return self._client

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            from pymilvus import MilvusClient
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("pymilvus is required when RAG_PROVIDER=milvus.") from exc

        # 将 kwargs 的值保存下来，供后续流程判断或组装响应时使用。
        kwargs: dict[str, Any] = {"uri": self.config.uri}
        # 根据 self.config.token 判断当前流程该进入哪个处理分支。
        if self.config.token:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            kwargs["token"] = self.config.token
        # 根据 self.config.user 判断当前流程该进入哪个处理分支。
        if self.config.user:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            kwargs["user"] = self.config.user
        # 根据 self.config.password 判断当前流程该进入哪个处理分支。
        if self.config.password:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            kwargs["password"] = self.config.password
        # 根据 self.config.db_name 判断当前流程该进入哪个处理分支。
        if self.config.db_name:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            kwargs["db_name"] = self.config.db_name

        # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
        self._client = MilvusClient(**kwargs)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._client


# 定义 _hit_to_doc 相关的处理逻辑，供流程或外部调用复用。
def _hit_to_doc(hit: Any, source_type: SourceType) -> RetrievedDoc:
    # pymilvus 不同版本可能返回 dict 或对象形态，这里统一转换成 RetrievedDoc。
    # 把 Milvus 命中的原始记录转换成业务层可读的 RetrievedDoc。
    if isinstance(hit, dict):
        # 将 entity 的值保存下来，供后续流程判断或组装响应时使用。
        entity = hit.get("entity") or {}
        # 将 raw_score 的值保存下来，供后续流程判断或组装响应时使用。
        raw_score = hit.get("score", hit.get("distance", 0.0))
        # 将 raw_id 的值保存下来，供后续流程判断或组装响应时使用。
        raw_id = hit.get("id", "")
    # 处理前面条件都不满足时的默认分支。
    else:
        # 将 entity 的值保存下来，供后续流程判断或组装响应时使用。
        entity = getattr(hit, "entity", None) or {}
        # 将 raw_score 的值保存下来，供后续流程判断或组装响应时使用。
        raw_score = getattr(hit, "score", getattr(hit, "distance", 0.0))
        # 将 raw_id 的值保存下来，供后续流程判断或组装响应时使用。
        raw_id = getattr(hit, "id", "")
    # 根据 not isinstance(entity, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(entity, dict):
        # 将 entity 的值保存下来，供后续流程判断或组装响应时使用。
        entity = dict(entity)
    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
    metadata = _parse_metadata(entity.get("metadata"))
    # 将 doc_id 的值保存下来，供后续流程判断或组装响应时使用。
    doc_id = str(entity.get("id") or raw_id or metadata.get("id", ""))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return RetrievedDoc(
        # 将 id 的值保存下来，供后续流程判断或组装响应时使用。
        id=doc_id,
        # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
        text=str(entity.get("text") or ""),
        # 将 source_type 的值保存下来，供后续流程判断或组装响应时使用。
        source_type=source_type,
        # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
        score=float(raw_score or 0.0),
        # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
        service=_empty_to_none(entity.get("service")),
        # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
        title=_empty_to_none(entity.get("title")),
        # 将 created_at 的值保存下来，供后续流程判断或组装响应时使用。
        created_at=_to_int_or_none(entity.get("created_at")),
        # 将 source_uri 的值保存下来，供后续流程判断或组装响应时使用。
        source_uri=_empty_to_none(entity.get("source_uri") or entity.get("source")),
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata=metadata,
        # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
        embedding=[float(item) for item in entity.get("embedding", [])],
    )


# 定义 _parse_metadata 相关的处理逻辑，供流程或外部调用复用。
def _parse_metadata(value: Any) -> dict[str, Any]:
    # 解析 Milvus 中保存的 metadata JSON，解析失败时保留原始文本方便排查。
    if isinstance(value, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return value
    # 根据 isinstance(value, str) and value 判断当前流程该进入哪个处理分支。
    if isinstance(value, str) and value:
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # metadata 正常是 JSON 字符串；解析失败时保留原文，方便后续排查脏数据。
            parsed = json.loads(value)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return parsed if isinstance(parsed, dict) else {"raw": value}
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except json.JSONDecodeError:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {"raw": value}
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {}


# 定义 _empty_to_none 相关的处理逻辑，供流程或外部调用复用。
def _empty_to_none(value: Any) -> str | None:
    # 把空字符串统一转成 None，让检索结果里的可选字段更干净。
    text = str(value or "").strip()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return text or None


# 定义 _to_int_or_none 相关的处理逻辑，供流程或外部调用复用。
def _to_int_or_none(value: Any) -> int | None:
    # 把 Milvus 返回的时间字段转成整数，无法转换时返回空值。
    try:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return int(value)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except (TypeError, ValueError):
        return None


# 定义 _format_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 把检索分数格式化成短文本，方便日志里快速判断召回质量。
    if not scores:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(scores) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + suffix + "]"
