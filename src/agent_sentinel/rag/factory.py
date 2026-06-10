from __future__ import annotations

import logging

from agent_sentinel.config import Settings
from agent_sentinel.rag.base import BaseRetriever
from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.hybrid_retriever import HybridRetriever
from agent_sentinel.rag.message_history_retriever import MessageHistoryRetriever
from agent_sentinel.rag.milvus_client import MilvusSearchConfig, MilvusVectorClient
from agent_sentinel.rag.mock_retriever import MockRetriever
from agent_sentinel.rag.static_doc_retriever import StaticDocRetriever

logger = logging.getLogger(__name__)


def build_retriever(settings: Settings) -> BaseRetriever:
    # 根据配置组装 RAG 检索器；未启用 Milvus 或配置不完整时自动退回模拟检索。
    provider = settings.rag_provider.strip().lower()
    if provider != "milvus":
        logger.info("Using mock RAG provider provider=%s", provider)
        return MockRetriever()

    if not settings.milvus_uri:
        logger.warning("RAG_PROVIDER=milvus but MILVUS_URI is missing; falling back to mock RAG.")
        return MockRetriever()

    embedding = EmbeddingClient(
        api_key=settings.embedding_api_key or "",
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        mock_enabled=settings.embedding_mock_enabled,
    )
    milvus = MilvusVectorClient(
        MilvusSearchConfig(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            user=settings.milvus_user,
            password=settings.milvus_password,
            db_name=settings.milvus_db_name,
        )
    )
    pruning_config = {
        "reranker_model_name": settings.rag_reranker_model_name,
        "reranker_device": settings.rag_reranker_device,
        "reranker_api_endpoint": settings.rag_reranker_api_endpoint,
        "reranker_api_key": settings.rag_reranker_api_key,
        "reranker_api_format": settings.rag_reranker_api_format,
        "reranker_timeout_seconds": settings.rag_reranker_timeout_seconds,
    }

    retrievers: list[BaseRetriever] = []
    if settings.rag_static_enabled:
        retrievers.append(
            StaticDocRetriever(
                milvus=milvus,
                embedding=embedding,
                collection_name=settings.rag_static_collection,
                top_k=settings.rag_static_top_k,
                weight=settings.rag_static_weight,
                recall_top_k=settings.rag_static_recall_top_k,
                pruning_enabled=settings.rag_pruning_enabled,
                pruning_config=pruning_config,
            )
        )
    if settings.rag_message_enabled:
        retrievers.append(
            MessageHistoryRetriever(
                milvus=milvus,
                embedding=embedding,
                collection_name=settings.rag_message_collection,
                top_k=settings.rag_message_top_k,
                weight=settings.rag_message_weight,
                default_days=settings.rag_message_default_days,
                prune_top_k=settings.rag_history_prune_top_k,
                pruning_enabled=settings.rag_pruning_enabled,
                pruning_config=pruning_config,
            )
        )

    logger.info("Using Milvus hybrid RAG retrievers=%s", len(retrievers))
    return HybridRetriever(
        embedding=embedding,
        retrievers=retrievers,
        final_top_k=settings.rag_final_top_k,
        mmr_lambda=settings.rag_mmr_lambda,
    )
