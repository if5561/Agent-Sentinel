from __future__ import annotations

import logging
import math
import re
import time
from urllib.parse import urlparse
from typing import Any, Dict, List

import requests

try:  # Optional dependency used when local reranker scores need normalization.
    import numpy as np
except ImportError:  # pragma: no cover - depends on optional runtime package
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


default_config: dict[str, Any] = {
    "alpha_history": 0.6,
    "alpha_static_fallback": 0.7,
    "reranker_model_name": "qwen3-rerank",
    "reranker_device": "cpu",
    "reranker_api_endpoint": None,
    "reranker_api_key": None,
    "reranker_api_format": "auto",
    "reranker_timeout_seconds": 15,
}

HISTORY_FIELD_WEIGHTS = {
    "validation_result": 0.4,
    "need_human": 0.2,
    "human_decision": 0.2,
    "alert_summary": 0.2,
}

STATIC_CATEGORY_KEYWORDS = {
    "mq": {"mq", "kafka", "rocketmq", "rabbitmq", "message", "queue"},
    "redis": {"redis", "cache", "缓存"},
    "mysql": {"mysql", "sql", "database", "db", "数据库"},
    "jvm": {"jvm", "oom", "outofmemory", "gc"},
    "k8s": {"k8s", "kubernetes", "pod", "container", "容器"},
}


def jaccard_similarity(str1: str, str2: str) -> float:
    # 用词集合重叠度粗略衡量两段文本是否相似，作为元数据评分的轻量依据。
    left = set(_tokenize(str1))
    right = set(_tokenize(str2))
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def extract_query_features(query: str) -> dict[str, Any]:
    # 从告警文本中提取类别、级别、错误码和关键词，供静态知识排序时判断匹配度。
    normalized = _normalize_text(query)
    tokens = set(_tokenize(query))
    alert_categories: set[str] = set()
    for category, keywords in STATIC_CATEGORY_KEYWORDS.items():
        if category in normalized or any(keyword.lower() in normalized for keyword in keywords):
            alert_categories.add(category)

    severity_levels = {match.upper() for match in re.findall(r"\bP[0-2]\b", query, flags=re.IGNORECASE)}
    error_codes = set(re.findall(r"\b\d{3}\b", query))
    return {
        "normalized_query": normalized,
        "alert_categories": alert_categories,
        "severity_levels": severity_levels,
        "error_codes": error_codes,
        "tokens": tokens,
    }


def compute_history_metadata_score(metadata: dict, query: str) -> float:
    # 根据历史案例的校验结果、人工决策和告警摘要，计算它是否值得被复用。
    if not isinstance(metadata, dict):
        return 0.0

    scores: dict[str, float] = {}
    if "validation_result" in metadata:
        scores["validation_result"] = 1.0 if metadata.get("validation_result") is True else 0.0
    if "need_human" in metadata:
        scores["need_human"] = 1.0 if metadata.get("need_human") is False else 0.0
    if metadata.get("human_decision") is not None:
        decision = str(metadata.get("human_decision") or "").strip().lower()
        if decision == "approved":
            scores["human_decision"] = 1.0
        elif decision == "rejected":
            scores["human_decision"] = 0.0
        else:
            scores["human_decision"] = 0.5
    if metadata.get("alert_summary"):
        scores["alert_summary"] = jaccard_similarity(query, str(metadata.get("alert_summary") or ""))

    active_weight = sum(HISTORY_FIELD_WEIGHTS[key] for key in scores)
    if active_weight <= 0:
        return 0.0
    return sum(scores[key] * HISTORY_FIELD_WEIGHTS[key] for key in scores) / active_weight


def compute_static_metadata_score(metadata: dict, query_features: dict) -> float:
    # 根据静态文档的故障类型、级别、错误码和关键词，计算它与当前告警的匹配程度。
    if not isinstance(metadata, dict):
        metadata = {}

    normalized_query = str(query_features.get("normalized_query") or "")
    query_tokens = set(query_features.get("tokens") or set())
    query_categories = {str(item).lower() for item in query_features.get("alert_categories", set())}
    query_severities = {str(item).upper() for item in query_features.get("severity_levels", set())}
    query_error_codes = {str(item).lower() for item in query_features.get("error_codes", set())}

    category = _normalize_text(str(metadata.get("alert_category") or ""))
    category_score = 1.0 if category and (category in normalized_query or category in query_categories) else 0.0

    severity = str(metadata.get("severity_level") or "").strip().upper()
    severity_score = 1.0 if severity and severity in query_severities else 0.0

    error_codes = _as_list(metadata.get("error_code"))
    error_hits = sum(1 for code in error_codes if str(code).lower() in query_error_codes)
    error_score = error_hits / max(1, len(error_codes))

    keywords = _as_list(metadata.get("keywords"))
    keyword_hits = 0
    for keyword in keywords:
        normalized_keyword = _normalize_text(str(keyword))
        if normalized_keyword and (
            normalized_keyword in normalized_query
            or normalized_keyword in query_tokens
        ):
            keyword_hits += 1
    keyword_score = keyword_hits / max(1, len(keywords))

    metadata_score = (
        0.3 * category_score
        + 0.2 * severity_score
        + 0.25 * error_score
        + 0.25 * keyword_score
    )

    section_bonus = 0.0
    for token in _tokenize(str(metadata.get("section") or "")):
        if token in query_tokens or token in normalized_query:
            section_bonus = 0.05
            break
    return min(1.0, metadata_score + min(0.1, section_bonus))


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        api_format: str = "auto",
        timeout_seconds: int = 15,
    ) -> None:
        # 保存 reranker 的模型、设备和远程 API 配置，本地模型会在首次重排时再加载。
        self.model_name = model_name
        self.device = device
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_format = api_format
        self.timeout_seconds = timeout_seconds
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Return one relevance score for each document in the same order."""
        # 对候选文档重新打分，让更贴近当前告警的问题排到前面。
        if not documents:
            return []
        logger.info(
            "Reranker start model=%s mode=%s docs=%s query_chars=%s endpoint_configured=%s",
            self.model_name,
            self.api_mode(),
            len(documents),
            len(query or ""),
            bool(self.api_endpoint),
        )
        if self.api_endpoint:
            return self._rerank_api(query, documents)
        return self._rerank_local(query, documents)

    def _rerank_api(self, query: str, documents: list[str]) -> list[float]:
        # 调用外部 reranker API 给候选文档打分，适合线上环境复用远程模型能力。
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif _is_dashscope_endpoint(self.api_endpoint or ""):
            raise RuntimeError("RAG_RERANKER_API_KEY is required for DashScope reranker API.")

        payload = self._build_api_payload(query, documents)
        started = time.perf_counter()
        logger.info(
            "Reranker API request model=%s mode=%s docs=%s endpoint_host=%s",
            self.model_name,
            self.api_mode(),
            len(documents),
            _endpoint_host(self.api_endpoint or ""),
        )
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()
        if self._api_payload_mode().startswith("dashscope"):
            scores = self._parse_dashscope_response(response_payload, len(documents))
            logger.info(
                "Reranker API completed model=%s mode=%s docs=%s scores=%s elapsed_ms=%s",
                self.model_name,
                self.api_mode(),
                len(documents),
                _format_scores(scores),
                int((time.perf_counter() - started) * 1000),
            )
            return scores

        raw_scores = response_payload.get("scores")
        if raw_scores is None and isinstance(response_payload.get("results"), list):
            raw_scores = [_score_from_result(item) for item in response_payload["results"] if isinstance(item, dict)]
        scores = [_bounded_score(value) for value in raw_scores or []]
        if len(scores) != len(documents):
            raise RuntimeError("Reranker API returned a score count that does not match documents.")
        logger.info(
            "Reranker API completed model=%s mode=%s docs=%s scores=%s elapsed_ms=%s",
            self.model_name,
            self.api_mode(),
            len(documents),
            _format_scores(scores),
            int((time.perf_counter() - started) * 1000),
        )
        return scores

    def _build_api_payload(self, query: str, documents: list[str]) -> dict[str, Any]:
        # 按不同 reranker 服务的协议组装请求体，保证同一批候选文档能被统一评分。
        mode = self._api_payload_mode()
        if mode == "dashscope_vl":
            return {
                "model": self.model_name,
                "input": {
                    "query": {"text": query},
                    "documents": [{"text": document} for document in documents],
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(documents),
                },
            }
        if mode == "dashscope_text":
            return {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "return_documents": False,
            }
        return {"model": self.model_name, "query": query, "documents": documents}

    def _parse_dashscope_response(self, payload: dict[str, Any], document_count: int) -> list[float]:
        # 解析 DashScope reranker 返回结果，并按原始文档顺序还原每条文档的分数。
        output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        results = output.get("results") if isinstance(output, dict) else None
        if results is None:
            results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("DashScope reranker response is missing output.results.")

        scores = [0.0 for _ in range(document_count)]
        for fallback_index, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            index = item.get("index", fallback_index)
            try:
                doc_index = int(index)
            except (TypeError, ValueError):
                doc_index = fallback_index
            if 0 <= doc_index < document_count:
                scores[doc_index] = _bounded_score(_score_from_result(item))
        return scores

    def _api_payload_mode(self) -> str:
        # 根据配置和接口地址判断应该使用哪种 reranker 请求格式。
        api_format = str(self.api_format or "auto").strip().lower()
        if api_format in {"dashscope_vl", "dashscope-vl", "dashscope_multimodal", "dashscope-multimodal"}:
            return "dashscope_vl"
        if api_format in {"dashscope", "dashscope_text", "dashscope-text"}:
            return "dashscope_text"
        if api_format in {"generic", "openai", "cohere"}:
            return "generic"
        if _is_dashscope_compatible_endpoint(self.api_endpoint or ""):
            return "generic"
        if _is_dashscope_endpoint(self.api_endpoint or ""):
            return "dashscope_vl" if "vl" in self.model_name.lower() else "dashscope_text"
        return "generic"

    def api_mode(self) -> str:
        # 返回当前 reranker 使用远程 API 还是本地模型，主要用于日志排查。
        return self._api_payload_mode() if self.api_endpoint else "local"

    def _rerank_local(self, query: str, documents: list[str]) -> list[float]:
        # 用本地 transformers 模型给候选文档打分，适合没有远程 reranker 的环境。
        started = time.perf_counter()
        self._ensure_local_model()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        inputs = tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
        if len(logits.shape) == 2 and logits.shape[-1] > 1:
            scores_tensor = torch.softmax(logits, dim=-1)[:, -1]
        else:
            scores_tensor = torch.sigmoid(logits.reshape(-1))
        scores = [float(item) for item in scores_tensor.detach().cpu().tolist()]
        normalized = _normalize_scores(scores)
        logger.info(
            "Reranker local completed model=%s device=%s docs=%s scores=%s elapsed_ms=%s",
            self.model_name,
            self.device,
            len(documents),
            _format_scores(normalized),
            int((time.perf_counter() - started) * 1000),
        )
        return normalized

    def _ensure_local_model(self) -> None:
        # 首次本地重排时才加载 tokenizer 和模型，避免启动阶段占用大量内存。
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("transformers and torch are required for local reranker inference.") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)


def prune_by_differential_strategy(
    query: str,
    candidates: List[Dict],
    top_k: int,
    config: Dict | None = None,
) -> List[Dict]:
    # 根据候选来源选择不同裁剪策略，历史案例重业务元数据，静态文档优先走 reranker。
    if not candidates or len(candidates) <= top_k:
        logger.info("RAG pruning skipped source_type=%s candidates=%s top_k=%s", candidates[0].get("source_type") if candidates else None, len(candidates), top_k)
        return candidates

    merged_config = {**default_config, **(config or {})}
    source_type = str(candidates[0].get("source_type") or "").lower()
    logger.info("RAG pruning start source_type=%s candidates=%s top_k=%s", source_type, len(candidates), top_k)
    if source_type in {"history", "message_history"}:
        return _prune_history(query, candidates, top_k, merged_config)
    if source_type in {"static", "static_doc"}:
        return _prune_static(query, candidates, top_k, merged_config)

    logger.warning("Unknown RAG source_type=%s; falling back to vector score ordering.", source_type)
    return sorted(candidates, key=lambda item: _bounded_score(item.get("score")), reverse=True)[:top_k]


def _prune_history(query: str, candidates: list[dict], top_k: int, config: dict[str, Any]) -> list[dict]:
    # 裁剪历史案例候选，把向量相似度和案例质量元数据合成一个最终排序分。
    started = time.perf_counter()
    alpha = float(config.get("alpha_history", 0.6))
    ranked: list[dict] = []
    for candidate in candidates:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        metadata_score = compute_history_metadata_score(metadata, query)
        original_score = _bounded_score(candidate.get("score"))
        combined_score = alpha * original_score + (1 - alpha) * metadata_score
        ranked.append(
            {
                **candidate,
                "metadata_score": metadata_score,
                "combined_score": combined_score,
            }
        )
    selected = sorted(ranked, key=lambda item: item["combined_score"], reverse=True)[:top_k]
    logger.info(
        "History metadata pruning completed candidates=%s selected=%s alpha=%s selected_scores=%s elapsed_ms=%s",
        len(candidates),
        len(selected),
        alpha,
        _format_candidate_scores(selected, "combined_score"),
        int((time.perf_counter() - started) * 1000),
    )
    return selected


def _prune_static(query: str, candidates: list[dict], top_k: int, config: dict[str, Any]) -> list[dict]:
    # 裁剪静态知识文档，优先用 reranker 判断哪些文档最值得放进提示词。
    started = time.perf_counter()
    documents = [str(candidate.get("text") or "") for candidate in candidates]
    reranker = Reranker(
        model_name=str(config.get("reranker_model_name") or default_config["reranker_model_name"]),
        device=str(config.get("reranker_device") or default_config["reranker_device"]),
        api_endpoint=config.get("reranker_api_endpoint"),
        api_key=config.get("reranker_api_key"),
        api_format=str(config.get("reranker_api_format") or default_config["reranker_api_format"]),
        timeout_seconds=int(config.get("reranker_timeout_seconds") or default_config["reranker_timeout_seconds"]),
    )
    try:
        scores = reranker.rerank(query, documents)
        ranked = [
            {**candidate, "rerank_score": _bounded_score(score)}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        selected = sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        logger.info(
            "Static reranker pruning completed candidates=%s selected=%s model=%s api_mode=%s selected_scores=%s elapsed_ms=%s",
            len(candidates),
            len(selected),
            reranker.model_name,
            reranker.api_mode(),
            _format_candidate_scores(selected, "rerank_score"),
            int((time.perf_counter() - started) * 1000),
        )
        return selected
    except Exception as exc:
        logger.warning("Reranker unavailable; falling back to static metadata pruning: %s", exc)
        return _prune_static_fallback(query, candidates, top_k, config)


def _prune_static_fallback(query: str, candidates: list[dict], top_k: int, config: dict[str, Any]) -> list[dict]:
    # reranker 不可用时，用向量分和文档元数据分做保守排序。
    started = time.perf_counter()
    alpha = float(config.get("alpha_static_fallback", 0.7))
    query_features = extract_query_features(query)
    ranked: list[dict] = []
    for candidate in candidates:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        metadata_score = compute_static_metadata_score(metadata, query_features)
        original_score = _bounded_score(candidate.get("score"))
        combined_score = alpha * original_score + (1 - alpha) * metadata_score
        ranked.append(
            {
                **candidate,
                "metadata_score": metadata_score,
                "combined_score": combined_score,
            }
        )
    selected = sorted(ranked, key=lambda item: item["combined_score"], reverse=True)[:top_k]
    logger.info(
        "Static metadata fallback pruning completed candidates=%s selected=%s alpha=%s selected_scores=%s elapsed_ms=%s",
        len(candidates),
        len(selected),
        alpha,
        _format_candidate_scores(selected, "combined_score"),
        int((time.perf_counter() - started) * 1000),
    )
    return selected


def _normalize_text(text: str) -> str:
    # 把文本统一成小写并去掉标点，方便后续关键词匹配和分词。
    return re.sub(r"[^\w\s]", " ", str(text).lower()).strip()


def _tokenize(text: str) -> list[str]:
    # 从归一化文本中提取词语列表，用于相似度和关键词命中计算。
    normalized = _normalize_text(text)
    return re.findall(r"\w+", normalized)


def _as_list(value: Any) -> list[Any]:
    # 把单个值、元组、集合等统一转成列表，方便按多值字段处理。
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _bounded_score(value: Any) -> float:
    # 把任意分数压到 0 到 1 之间，避免异常值影响排序。
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _score_from_result(item: dict[str, Any]) -> Any:
    # 兼容不同 reranker 返回字段名，提取候选文档的相关性分数。
    return item.get("relevance_score", item.get("score", item.get("rerank_score", 0.0)))


def _is_dashscope_endpoint(endpoint: str) -> bool:
    # 判断 reranker 地址是否是 DashScope 原生接口，用来选择请求和鉴权方式。
    normalized = endpoint.lower()
    return "dashscope" in normalized or "/services/rerank/" in normalized


def _is_dashscope_compatible_endpoint(endpoint: str) -> bool:
    # 判断 reranker 地址是否是 DashScope 兼容模式接口。
    return "/compatible-api/" in endpoint.lower() or "/compatible-mode/" in endpoint.lower()


def _endpoint_host(endpoint: str) -> str:
    # 从完整接口地址中提取主机名，日志里只展示主机可减少噪声。
    try:
        return urlparse(endpoint).netloc or endpoint
    except Exception:
        return endpoint


def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 把前几名分数格式化成短字符串，方便日志观察排序效果。
    if not scores:
        return "[]"
    suffix = ", ..." if len(scores) > limit else ""
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + suffix + "]"


def _format_candidate_scores(candidates: list[dict], score_key: str, limit: int = 5) -> str:
    # 把候选文档的索引和分数格式化到日志，方便定位哪条资料被选中。
    if not candidates:
        return "[]"
    values = []
    for item in candidates[:limit]:
        values.append(f"{item.get('_index', '?')}:{_bounded_score(item.get(score_key)):.4f}")
    suffix = ", ..." if len(candidates) > limit else ""
    return "[" + ", ".join(values) + suffix + "]"


def _normalize_scores(scores: list[float]) -> list[float]:
    # 把本地模型输出的原始分数归一化，保证排序阶段使用稳定的 0 到 1 分数。
    if not scores:
        return []
    bounded = [_bounded_score(score) for score in scores]
    if any(score != 0.0 for score in bounded):
        return bounded
    if np is None:
        return bounded
    array = np.asarray(scores, dtype=float)
    min_score = float(array.min())
    max_score = float(array.max())
    if max_score == min_score:
        return [0.0 for _ in scores]
    return [float(item) for item in ((array - min_score) / (max_score - min_score)).tolist()]


if __name__ == "__main__":
    history_candidates = [
        {
            "text": "Redis timeout resolved by lowering retry concurrency.",
            "score": 0.82,
            "metadata": {
                "alert_summary": "Redis 连接超时",
                "validation_result": True,
                "need_human": False,
                "human_decision": "approved",
            },
            "source_type": "history",
        },
        {
            "text": "Unrelated rejected diagnosis.",
            "score": 0.91,
            "metadata": {
                "alert_summary": "订单同步失败",
                "validation_result": False,
                "need_human": True,
                "human_decision": "rejected",
            },
            "source_type": "history",
        },
    ]
    static_candidates = [
        {
            "text": "Redis connection timeout with 500 errors.",
            "score": 0.72,
            "metadata": {
                "alert_category": "Redis",
                "severity_level": "P1",
                "error_code": ["500"],
                "keywords": ["连接", "超时"],
                "section": "Redis/连接池/超时",
            },
            "source_type": "static",
        },
        {
            "text": "MQ retry backlog.",
            "score": 0.8,
            "metadata": {"alert_category": "MQ", "keywords": ["重试"]},
            "source_type": "static",
        },
    ]
    print(prune_by_differential_strategy("Redis 连接超时，错误码 500", history_candidates, 1))
    print(prune_by_differential_strategy("Redis 连接超时，错误码 500", static_candidates, 1))
