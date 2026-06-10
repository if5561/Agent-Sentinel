from __future__ import annotations

import math

from agent_sentinel.rag.models import RetrievedDoc


def cosine_similarity(left: list[float], right: list[float]) -> float:
    # 计算两个向量的相似度，数值越高表示两段文本语义越接近。
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def select_mmr(
    *,
    query_embedding: list[float],
    docs: list[RetrievedDoc],
    top_k: int,
    lambda_mult: float = 0.55,
) -> list[RetrievedDoc]:
    # 用 MMR 策略挑选最终文档，既保留高相关内容，也避免上下文被重复片段占满。
    if top_k <= 0 or not docs:
        return []
    if not query_embedding or not any(doc.embedding for doc in docs):
        return sorted(docs, key=lambda doc: doc.weighted_score or doc.score, reverse=True)[:top_k]

    selected: list[RetrievedDoc] = []
    candidates = docs[:]

    while candidates and len(selected) < top_k:
        best_doc: RetrievedDoc | None = None
        best_score = float("-inf")

        for doc in candidates:
            relevance = doc.weighted_score or doc.score or cosine_similarity(query_embedding, doc.embedding)
            diversity_penalty = max(
                (cosine_similarity(doc.embedding, selected_doc.embedding) for selected_doc in selected),
                default=0.0,
            )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_doc = doc

        if best_doc is None:
            break
        selected.append(best_doc)
        candidates.remove(best_doc)

    return selected
