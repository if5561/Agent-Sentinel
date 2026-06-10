from __future__ import annotations

import math

from agent_sentinel.rag.models import RetrievedDoc


# 定义 cosine_similarity 相关的处理逻辑，供流程或外部调用复用。
def cosine_similarity(left: list[float], right: list[float]) -> float:
    # 计算两个向量的相似度，数值越高表示两段文本语义越接近。
    if not left or not right or len(left) != len(right):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0
    # 将 dot 的值保存下来，供后续流程判断或组装响应时使用。
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    # 将 left_norm 的值保存下来，供后续流程判断或组装响应时使用。
    left_norm = math.sqrt(sum(a * a for a in left))
    # 将 right_norm 的值保存下来，供后续流程判断或组装响应时使用。
    right_norm = math.sqrt(sum(b * b for b in right))
    # 根据 left_norm == 0 or right_norm == 0 判断当前流程该进入哪个处理分支。
    if left_norm == 0 or right_norm == 0:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return dot / (left_norm * right_norm)


# 定义 select_mmr 相关的处理逻辑，供流程或外部调用复用。
def select_mmr(
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    query_embedding: list[float],
    # 执行当前业务步骤，推动流程继续向下游推进。
    docs: list[RetrievedDoc],
    # 执行当前业务步骤，推动流程继续向下游推进。
    top_k: int,
    # 将 lambda_mult 的值保存下来，供后续流程判断或组装响应时使用。
    lambda_mult: float = 0.55,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> list[RetrievedDoc]:
    # 用 MMR 策略挑选最终文档，既保留高相关内容，也避免上下文被重复片段占满。
    if top_k <= 0 or not docs:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # 根据 not query_embedding or not any(doc.embedding for doc... 判断当前流程该进入哪个处理分支。
    if not query_embedding or not any(doc.embedding for doc in docs):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return sorted(docs, key=lambda doc: doc.weighted_score or doc.score, reverse=True)[:top_k]

    # 将 selected 的值保存下来，供后续流程判断或组装响应时使用。
    selected: list[RetrievedDoc] = []
    # 将 candidates 的值保存下来，供后续流程判断或组装响应时使用。
    candidates = docs[:]

    # 在条件仍然成立时持续执行循环逻辑。
    while candidates and len(selected) < top_k:
        # 将 best_doc 的值保存下来，供后续流程判断或组装响应时使用。
        best_doc: RetrievedDoc | None = None
        # 将 best_score 的值保存下来，供后续流程判断或组装响应时使用。
        best_score = float("-inf")

        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for doc in candidates:
            # 将 relevance 的值保存下来，供后续流程判断或组装响应时使用。
            relevance = doc.weighted_score or doc.score or cosine_similarity(query_embedding, doc.embedding)
            # 将 diversity_penalty 的值保存下来，供后续流程判断或组装响应时使用。
            diversity_penalty = max(
                # 调用 cosine_similarity 完成当前步骤需要的业务处理。
                (cosine_similarity(doc.embedding, selected_doc.embedding) for selected_doc in selected),
                # 将 default 的值保存下来，供后续流程判断或组装响应时使用。
                default=0.0,
            )
            # 将 mmr_score 的值保存下来，供后续流程判断或组装响应时使用。
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            # 根据 mmr_score > best_score 判断当前流程该进入哪个处理分支。
            if mmr_score > best_score:
                # 将 best_score 的值保存下来，供后续流程判断或组装响应时使用。
                best_score = mmr_score
                # 将 best_doc 的值保存下来，供后续流程判断或组装响应时使用。
                best_doc = doc

        # 根据 best_doc is None 判断当前流程该进入哪个处理分支。
        if best_doc is None:
            # 满足停止条件后退出循环，避免继续执行无效处理。
            break
        # 把当前结果追加到集合中，逐步构建最终输出。
        selected.append(best_doc)
        # 调用 candidates.remove 完成当前步骤需要的业务处理。
        candidates.remove(best_doc)

    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return selected
