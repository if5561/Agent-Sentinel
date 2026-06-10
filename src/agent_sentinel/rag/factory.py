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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 build_retriever 相关的处理逻辑，供流程或外部调用复用。
def build_retriever(settings: Settings) -> BaseRetriever:
    # 根据配置组装 RAG 检索器；未启用 Milvus 或配置不完整时自动退回模拟检索。
    provider = settings.rag_provider.strip().lower()
    # 根据 provider != "milvus" 判断当前流程该进入哪个处理分支。
    if provider != "milvus":
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Using mock RAG provider provider=%s", provider)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return MockRetriever()

    # 根据 not settings.milvus_uri 判断当前流程该进入哪个处理分支。
    if not settings.milvus_uri:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.warning("RAG_PROVIDER=milvus but MILVUS_URI is missing; falling back to mock RAG.")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return MockRetriever()

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
    # 将 pruning_config 的值保存下来，供后续流程判断或组装响应时使用。
    pruning_config = {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reranker_model_name": settings.rag_reranker_model_name,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reranker_device": settings.rag_reranker_device,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reranker_api_endpoint": settings.rag_reranker_api_endpoint,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reranker_api_key": settings.rag_reranker_api_key,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reranker_api_format": settings.rag_reranker_api_format,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "reranker_timeout_seconds": settings.rag_reranker_timeout_seconds,
    }

    # 将 retrievers 的值保存下来，供后续流程判断或组装响应时使用。
    retrievers: list[BaseRetriever] = []
    # 根据 settings.rag_static_enabled 判断当前流程该进入哪个处理分支。
    if settings.rag_static_enabled:
        # 把当前结果追加到集合中，逐步构建最终输出。
        retrievers.append(
            # 调用 StaticDocRetriever 完成当前步骤需要的业务处理。
            StaticDocRetriever(
                # 将 milvus 的值保存下来，供后续流程判断或组装响应时使用。
                milvus=milvus,
                # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
                embedding=embedding,
                # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
                collection_name=settings.rag_static_collection,
                # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
                top_k=settings.rag_static_top_k,
                # 将 weight 的值保存下来，供后续流程判断或组装响应时使用。
                weight=settings.rag_static_weight,
                # 将 recall_top_k 的值保存下来，供后续流程判断或组装响应时使用。
                recall_top_k=settings.rag_static_recall_top_k,
                # 将 pruning_enabled 的值保存下来，供后续流程判断或组装响应时使用。
                pruning_enabled=settings.rag_pruning_enabled,
                # 将 pruning_config 的值保存下来，供后续流程判断或组装响应时使用。
                pruning_config=pruning_config,
            )
        )
    # 根据 settings.rag_message_enabled 判断当前流程该进入哪个处理分支。
    if settings.rag_message_enabled:
        # 把当前结果追加到集合中，逐步构建最终输出。
        retrievers.append(
            # 调用 MessageHistoryRetriever 完成当前步骤需要的业务处理。
            MessageHistoryRetriever(
                # 将 milvus 的值保存下来，供后续流程判断或组装响应时使用。
                milvus=milvus,
                # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
                embedding=embedding,
                # 将 collection_name 的值保存下来，供后续流程判断或组装响应时使用。
                collection_name=settings.rag_message_collection,
                # 将 top_k 的值保存下来，供后续流程判断或组装响应时使用。
                top_k=settings.rag_message_top_k,
                # 将 weight 的值保存下来，供后续流程判断或组装响应时使用。
                weight=settings.rag_message_weight,
                # 将 default_days 的值保存下来，供后续流程判断或组装响应时使用。
                default_days=settings.rag_message_default_days,
                # 将 prune_top_k 的值保存下来，供后续流程判断或组装响应时使用。
                prune_top_k=settings.rag_history_prune_top_k,
                # 将 pruning_enabled 的值保存下来，供后续流程判断或组装响应时使用。
                pruning_enabled=settings.rag_pruning_enabled,
                # 将 pruning_config 的值保存下来，供后续流程判断或组装响应时使用。
                pruning_config=pruning_config,
            )
        )

    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("Using Milvus hybrid RAG retrievers=%s", len(retrievers))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return HybridRetriever(
        # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
        embedding=embedding,
        # 将 retrievers 的值保存下来，供后续流程判断或组装响应时使用。
        retrievers=retrievers,
        # 将 final_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        final_top_k=settings.rag_final_top_k,
        # 将 mmr_lambda 的值保存下来，供后续流程判断或组装响应时使用。
        mmr_lambda=settings.rag_mmr_lambda,
    )
