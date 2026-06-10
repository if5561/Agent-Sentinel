from __future__ import annotations

import logging
import math
import re
import time
from urllib.parse import urlparse
from typing import Any, Dict, List

import requests

# 进入可能失败的处理块，便于后续统一捕获和恢复。
try:  # Optional dependency used when local reranker scores need normalization.
    import numpy as np
# 捕获当前步骤中的异常，转换为日志、指标或降级结果。
except ImportError:  # pragma: no cover - depends on optional runtime package
    # 将 np 的值保存下来，供后续流程判断或组装响应时使用。
    np = None  # type: ignore[assignment]

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 将 default_config 的值保存下来，供后续流程判断或组装响应时使用。
default_config: dict[str, Any] = {
    # 执行当前业务步骤，推动流程继续向下游推进。
    "alpha_history": 0.6,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "alpha_static_fallback": 0.7,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "reranker_model_name": "qwen3-rerank",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "reranker_device": "cpu",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "reranker_api_endpoint": None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "reranker_api_key": None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "reranker_api_format": "auto",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "reranker_timeout_seconds": 15,
}

# 将 HISTORY_FIELD_WEIGHTS 的值保存下来，供后续流程判断或组装响应时使用。
HISTORY_FIELD_WEIGHTS = {
    # 执行当前业务步骤，推动流程继续向下游推进。
    "validation_result": 0.4,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "need_human": 0.2,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "human_decision": 0.2,
    # 执行当前业务步骤，推动流程继续向下游推进。
    "alert_summary": 0.2,
}

# 将 STATIC_CATEGORY_KEYWORDS 的值保存下来，供后续流程判断或组装响应时使用。
STATIC_CATEGORY_KEYWORDS = {
    # 执行当前业务步骤，推动流程继续向下游推进。
    "mq": {"mq", "kafka", "rocketmq", "rabbitmq", "message", "queue"},
    # 执行当前业务步骤，推动流程继续向下游推进。
    "redis": {"redis", "cache", "缓存"},
    # 执行当前业务步骤，推动流程继续向下游推进。
    "mysql": {"mysql", "sql", "database", "db", "数据库"},
    # 执行当前业务步骤，推动流程继续向下游推进。
    "jvm": {"jvm", "oom", "outofmemory", "gc"},
    # 执行当前业务步骤，推动流程继续向下游推进。
    "k8s": {"k8s", "kubernetes", "pod", "container", "容器"},
}


# 定义 jaccard_similarity 相关的处理逻辑，供流程或外部调用复用。
def jaccard_similarity(str1: str, str2: str) -> float:
    # 用词集合重叠度粗略衡量两段文本是否相似，作为元数据评分的轻量依据。
    left = set(_tokenize(str1))
    # 将 right 的值保存下来，供后续流程判断或组装响应时使用。
    right = set(_tokenize(str2))
    # 将 union 的值保存下来，供后续流程判断或组装响应时使用。
    union = left | right
    # 根据 not union 判断当前流程该进入哪个处理分支。
    if not union:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return len(left & right) / len(union)


# 定义 extract_query_features 相关的处理逻辑，供流程或外部调用复用。
def extract_query_features(query: str) -> dict[str, Any]:
    # 从告警文本中提取类别、级别、错误码和关键词，供静态知识排序时判断匹配度。
    normalized = _normalize_text(query)
    # 将 tokens 的值保存下来，供后续流程判断或组装响应时使用。
    tokens = set(_tokenize(query))
    # 将 alert_categories 的值保存下来，供后续流程判断或组装响应时使用。
    alert_categories: set[str] = set()
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for category, keywords in STATIC_CATEGORY_KEYWORDS.items():
        # 根据 category in normalized or any(keyword.lower() in nor... 判断当前流程该进入哪个处理分支。
        if category in normalized or any(keyword.lower() in normalized for keyword in keywords):
            # 调用 alert_categories.add 完成当前步骤需要的业务处理。
            alert_categories.add(category)

    # 将 severity_levels 的值保存下来，供后续流程判断或组装响应时使用。
    severity_levels = {match.upper() for match in re.findall(r"\bP[0-2]\b", query, flags=re.IGNORECASE)}
    # 将 error_codes 的值保存下来，供后续流程判断或组装响应时使用。
    error_codes = set(re.findall(r"\b\d{3}\b", query))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "normalized_query": normalized,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "alert_categories": alert_categories,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "severity_levels": severity_levels,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "error_codes": error_codes,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "tokens": tokens,
    }


# 定义 compute_history_metadata_score 相关的处理逻辑，供流程或外部调用复用。
def compute_history_metadata_score(metadata: dict, query: str) -> float:
    # 根据历史案例的校验结果、人工决策和告警摘要，计算它是否值得被复用。
    if not isinstance(metadata, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0

    # 将 scores 的值保存下来，供后续流程判断或组装响应时使用。
    scores: dict[str, float] = {}
    # 根据 "validation_result" in metadata 判断当前流程该进入哪个处理分支。
    if "validation_result" in metadata:
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        scores["validation_result"] = 1.0 if metadata.get("validation_result") is True else 0.0
    # 根据 "need_human" in metadata 判断当前流程该进入哪个处理分支。
    if "need_human" in metadata:
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        scores["need_human"] = 1.0 if metadata.get("need_human") is False else 0.0
    # 根据 metadata.get("human_decision") is not None 判断当前流程该进入哪个处理分支。
    if metadata.get("human_decision") is not None:
        # 将 decision 的值保存下来，供后续流程判断或组装响应时使用。
        decision = str(metadata.get("human_decision") or "").strip().lower()
        # 根据 decision == "approved" 判断当前流程该进入哪个处理分支。
        if decision == "approved":
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            scores["human_decision"] = 1.0
        # 根据 decision == "rejected" 判断当前流程该进入哪个处理分支。
        elif decision == "rejected":
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            scores["human_decision"] = 0.0
        # 处理前面条件都不满足时的默认分支。
        else:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            scores["human_decision"] = 0.5
    # 根据 metadata.get("alert_summary") 判断当前流程该进入哪个处理分支。
    if metadata.get("alert_summary"):
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        scores["alert_summary"] = jaccard_similarity(query, str(metadata.get("alert_summary") or ""))

    # 将 active_weight 的值保存下来，供后续流程判断或组装响应时使用。
    active_weight = sum(HISTORY_FIELD_WEIGHTS[key] for key in scores)
    # 根据 active_weight <= 0 判断当前流程该进入哪个处理分支。
    if active_weight <= 0:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return sum(scores[key] * HISTORY_FIELD_WEIGHTS[key] for key in scores) / active_weight


# 定义 compute_static_metadata_score 相关的处理逻辑，供流程或外部调用复用。
def compute_static_metadata_score(metadata: dict, query_features: dict) -> float:
    # 根据静态文档的故障类型、级别、错误码和关键词，计算它与当前告警的匹配程度。
    if not isinstance(metadata, dict):
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = {}

    # 将 normalized_query 的值保存下来，供后续流程判断或组装响应时使用。
    normalized_query = str(query_features.get("normalized_query") or "")
    # 将 query_tokens 的值保存下来，供后续流程判断或组装响应时使用。
    query_tokens = set(query_features.get("tokens") or set())
    # 将 query_categories 的值保存下来，供后续流程判断或组装响应时使用。
    query_categories = {str(item).lower() for item in query_features.get("alert_categories", set())}
    # 将 query_severities 的值保存下来，供后续流程判断或组装响应时使用。
    query_severities = {str(item).upper() for item in query_features.get("severity_levels", set())}
    # 将 query_error_codes 的值保存下来，供后续流程判断或组装响应时使用。
    query_error_codes = {str(item).lower() for item in query_features.get("error_codes", set())}

    # 将 category 的值保存下来，供后续流程判断或组装响应时使用。
    category = _normalize_text(str(metadata.get("alert_category") or ""))
    # 将 category_score 的值保存下来，供后续流程判断或组装响应时使用。
    category_score = 1.0 if category and (category in normalized_query or category in query_categories) else 0.0

    # 将 severity 的值保存下来，供后续流程判断或组装响应时使用。
    severity = str(metadata.get("severity_level") or "").strip().upper()
    # 将 severity_score 的值保存下来，供后续流程判断或组装响应时使用。
    severity_score = 1.0 if severity and severity in query_severities else 0.0

    # 将 error_codes 的值保存下来，供后续流程判断或组装响应时使用。
    error_codes = _as_list(metadata.get("error_code"))
    # 将 error_hits 的值保存下来，供后续流程判断或组装响应时使用。
    error_hits = sum(1 for code in error_codes if str(code).lower() in query_error_codes)
    # 将 error_score 的值保存下来，供后续流程判断或组装响应时使用。
    error_score = error_hits / max(1, len(error_codes))

    # 将 keywords 的值保存下来，供后续流程判断或组装响应时使用。
    keywords = _as_list(metadata.get("keywords"))
    # 将 keyword_hits 的值保存下来，供后续流程判断或组装响应时使用。
    keyword_hits = 0
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for keyword in keywords:
        # 将 normalized_keyword 的值保存下来，供后续流程判断或组装响应时使用。
        normalized_keyword = _normalize_text(str(keyword))
        # 根据 normalized_keyword and ( 判断当前流程该进入哪个处理分支。
        if normalized_keyword and (
            # 执行当前业务步骤，推动流程继续向下游推进。
            normalized_keyword in normalized_query
            # 执行当前业务步骤，推动流程继续向下游推进。
            or normalized_keyword in query_tokens
        ):
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            keyword_hits += 1
    # 将 keyword_score 的值保存下来，供后续流程判断或组装响应时使用。
    keyword_score = keyword_hits / max(1, len(keywords))

    # 将 metadata_score 的值保存下来，供后续流程判断或组装响应时使用。
    metadata_score = (
        # 执行当前业务步骤，推动流程继续向下游推进。
        0.3 * category_score
        # 执行当前业务步骤，推动流程继续向下游推进。
        + 0.2 * severity_score
        # 执行当前业务步骤，推动流程继续向下游推进。
        + 0.25 * error_score
        # 执行当前业务步骤，推动流程继续向下游推进。
        + 0.25 * keyword_score
    )

    # 将 section_bonus 的值保存下来，供后续流程判断或组装响应时使用。
    section_bonus = 0.0
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for token in _tokenize(str(metadata.get("section") or "")):
        # 根据 token in query_tokens or token in normalized_query 判断当前流程该进入哪个处理分支。
        if token in query_tokens or token in normalized_query:
            # 将 section_bonus 的值保存下来，供后续流程判断或组装响应时使用。
            section_bonus = 0.05
            # 满足停止条件后退出循环，避免继续执行无效处理。
            break
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return min(1.0, metadata_score + min(0.1, section_bonus))


# 定义 Reranker 组件，集中管理这个模块的状态和行为。
class Reranker:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        model_name: str,
        # 将 device 的值保存下来，供后续流程判断或组装响应时使用。
        device: str = "cpu",
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 api_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
        api_endpoint: str | None = None,
        # 将 api_key 的值保存下来，供后续流程判断或组装响应时使用。
        api_key: str | None = None,
        # 将 api_format 的值保存下来，供后续流程判断或组装响应时使用。
        api_format: str = "auto",
        # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        timeout_seconds: int = 15,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 保存 reranker 的模型、设备和远程 API 配置，本地模型会在首次重排时再加载。
        self.model_name = model_name
        # 将 self.device 的值保存下来，供后续流程判断或组装响应时使用。
        self.device = device
        # 将 self.api_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
        self.api_endpoint = api_endpoint
        # 将 self.api_key 的值保存下来，供后续流程判断或组装响应时使用。
        self.api_key = api_key
        # 将 self.api_format 的值保存下来，供后续流程判断或组装响应时使用。
        self.api_format = api_format
        # 将 self.timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        self.timeout_seconds = timeout_seconds
        # 将 self._tokenizer 的值保存下来，供后续流程判断或组装响应时使用。
        self._tokenizer: Any | None = None
        # 将 self._model 的值保存下来，供后续流程判断或组装响应时使用。
        self._model: Any | None = None
        # 将 self._torch 的值保存下来，供后续流程判断或组装响应时使用。
        self._torch: Any | None = None

    # 定义 rerank 相关的处理逻辑，供流程或外部调用复用。
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Return one relevance score for each document in the same order."""
        # 对候选文档重新打分，让更贴近当前告警的问题排到前面。
        if not documents:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return []
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Reranker start model=%s mode=%s docs=%s query_chars=%s endpoint_configured=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.model_name,
            # 调用 self.api_mode 完成当前步骤需要的业务处理。
            self.api_mode(),
            # 调用 len 完成当前步骤需要的业务处理。
            len(documents),
            # 调用 len 完成当前步骤需要的业务处理。
            len(query or ""),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(self.api_endpoint),
        )
        # 根据 self.api_endpoint 判断当前流程该进入哪个处理分支。
        if self.api_endpoint:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return self._rerank_api(query, documents)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._rerank_local(query, documents)

    # 定义 _rerank_api 相关的处理逻辑，供流程或外部调用复用。
    def _rerank_api(self, query: str, documents: list[str]) -> list[float]:
        # 调用外部 reranker API 给候选文档打分，适合线上环境复用远程模型能力。
        headers = {"Content-Type": "application/json"}
        # 根据 self.api_key 判断当前流程该进入哪个处理分支。
        if self.api_key:
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            headers["Authorization"] = f"Bearer {self.api_key}"
        # 根据 _is_dashscope_endpoint(self.api_endpoint or "") 判断当前流程该进入哪个处理分支。
        elif _is_dashscope_endpoint(self.api_endpoint or ""):
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("RAG_RERANKER_API_KEY is required for DashScope reranker API.")

        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = self._build_api_payload(query, documents)
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Reranker API request model=%s mode=%s docs=%s endpoint_host=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.model_name,
            # 调用 self.api_mode 完成当前步骤需要的业务处理。
            self.api_mode(),
            # 调用 len 完成当前步骤需要的业务处理。
            len(documents),
            # 调用 _endpoint_host 完成当前步骤需要的业务处理。
            _endpoint_host(self.api_endpoint or ""),
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.api_endpoint,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers=headers,
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json=payload,
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout_seconds,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 response_payload 的值保存下来，供后续流程判断或组装响应时使用。
        response_payload = response.json()
        # 根据 self._api_payload_mode().startswith("dashscope") 判断当前流程该进入哪个处理分支。
        if self._api_payload_mode().startswith("dashscope"):
            # 将 scores 的值保存下来，供后续流程判断或组装响应时使用。
            scores = self._parse_dashscope_response(response_payload, len(documents))
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Reranker API completed model=%s mode=%s docs=%s scores=%s elapsed_ms=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.model_name,
                # 调用 self.api_mode 完成当前步骤需要的业务处理。
                self.api_mode(),
                # 调用 len 完成当前步骤需要的业务处理。
                len(documents),
                # 调用 _format_scores 完成当前步骤需要的业务处理。
                _format_scores(scores),
                # 调用 int 完成当前步骤需要的业务处理。
                int((time.perf_counter() - started) * 1000),
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return scores

        # 将 raw_scores 的值保存下来，供后续流程判断或组装响应时使用。
        raw_scores = response_payload.get("scores")
        # 根据 raw_scores is None and isinstance(response_payload.g... 判断当前流程该进入哪个处理分支。
        if raw_scores is None and isinstance(response_payload.get("results"), list):
            # 将 raw_scores 的值保存下来，供后续流程判断或组装响应时使用。
            raw_scores = [_score_from_result(item) for item in response_payload["results"] if isinstance(item, dict)]
        # 将 scores 的值保存下来，供后续流程判断或组装响应时使用。
        scores = [_bounded_score(value) for value in raw_scores or []]
        # 根据 len(scores) != len(documents) 判断当前流程该进入哪个处理分支。
        if len(scores) != len(documents):
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("Reranker API returned a score count that does not match documents.")
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Reranker API completed model=%s mode=%s docs=%s scores=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.model_name,
            # 调用 self.api_mode 完成当前步骤需要的业务处理。
            self.api_mode(),
            # 调用 len 完成当前步骤需要的业务处理。
            len(documents),
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores(scores),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return scores

    # 定义 _build_api_payload 相关的处理逻辑，供流程或外部调用复用。
    def _build_api_payload(self, query: str, documents: list[str]) -> dict[str, Any]:
        # 按不同 reranker 服务的协议组装请求体，保证同一批候选文档能被统一评分。
        mode = self._api_payload_mode()
        # 根据 mode == "dashscope_vl" 判断当前流程该进入哪个处理分支。
        if mode == "dashscope_vl":
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "model": self.model_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "input": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "query": {"text": query},
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "documents": [{"text": document} for document in documents],
                },
                # 执行当前业务步骤，推动流程继续向下游推进。
                "parameters": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "return_documents": False,
                    # 调用 len 完成当前步骤需要的业务处理。
                    "top_n": len(documents),
                },
            }
        # 根据 mode == "dashscope_text" 判断当前流程该进入哪个处理分支。
        if mode == "dashscope_text":
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "model": self.model_name,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "query": query,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "documents": documents,
                # 调用 len 完成当前步骤需要的业务处理。
                "top_n": len(documents),
                # 执行当前业务步骤，推动流程继续向下游推进。
                "return_documents": False,
            }
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {"model": self.model_name, "query": query, "documents": documents}

    # 定义 _parse_dashscope_response 相关的处理逻辑，供流程或外部调用复用。
    def _parse_dashscope_response(self, payload: dict[str, Any], document_count: int) -> list[float]:
        # 解析 DashScope reranker 返回结果，并按原始文档顺序还原每条文档的分数。
        output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        # 将 results 的值保存下来，供后续流程判断或组装响应时使用。
        results = output.get("results") if isinstance(output, dict) else None
        # 根据 results is None 判断当前流程该进入哪个处理分支。
        if results is None:
            # 将 results 的值保存下来，供后续流程判断或组装响应时使用。
            results = payload.get("results")
        # 根据 not isinstance(results, list) 判断当前流程该进入哪个处理分支。
        if not isinstance(results, list):
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("DashScope reranker response is missing output.results.")

        # 将 scores 的值保存下来，供后续流程判断或组装响应时使用。
        scores = [0.0 for _ in range(document_count)]
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for fallback_index, item in enumerate(results):
            # 根据 not isinstance(item, dict) 判断当前流程该进入哪个处理分支。
            if not isinstance(item, dict):
                # 跳过当前项剩余逻辑，继续处理下一项数据。
                continue
            # 将 index 的值保存下来，供后续流程判断或组装响应时使用。
            index = item.get("index", fallback_index)
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 将 doc_index 的值保存下来，供后续流程判断或组装响应时使用。
                doc_index = int(index)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except (TypeError, ValueError):
                # 将 doc_index 的值保存下来，供后续流程判断或组装响应时使用。
                doc_index = fallback_index
            # 根据 0 <= doc_index < document_count 判断当前流程该进入哪个处理分支。
            if 0 <= doc_index < document_count:
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                scores[doc_index] = _bounded_score(_score_from_result(item))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return scores

    # 定义 _api_payload_mode 相关的处理逻辑，供流程或外部调用复用。
    def _api_payload_mode(self) -> str:
        # 根据配置和接口地址判断应该使用哪种 reranker 请求格式。
        api_format = str(self.api_format or "auto").strip().lower()
        # 根据 api_format in {"dashscope_vl", "dashscope-vl", "dash... 判断当前流程该进入哪个处理分支。
        if api_format in {"dashscope_vl", "dashscope-vl", "dashscope_multimodal", "dashscope-multimodal"}:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "dashscope_vl"
        # 根据 api_format in {"dashscope", "dashscope_text", "dashs... 判断当前流程该进入哪个处理分支。
        if api_format in {"dashscope", "dashscope_text", "dashscope-text"}:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "dashscope_text"
        # 根据 api_format in {"generic", "openai", "cohere"} 判断当前流程该进入哪个处理分支。
        if api_format in {"generic", "openai", "cohere"}:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "generic"
        # 根据 _is_dashscope_compatible_endpoint(self.api_endpoint ... 判断当前流程该进入哪个处理分支。
        if _is_dashscope_compatible_endpoint(self.api_endpoint or ""):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "generic"
        # 根据 _is_dashscope_endpoint(self.api_endpoint or "") 判断当前流程该进入哪个处理分支。
        if _is_dashscope_endpoint(self.api_endpoint or ""):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return "dashscope_vl" if "vl" in self.model_name.lower() else "dashscope_text"
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "generic"

    # 定义 api_mode 相关的处理逻辑，供流程或外部调用复用。
    def api_mode(self) -> str:
        # 返回当前 reranker 使用远程 API 还是本地模型，主要用于日志排查。
        return self._api_payload_mode() if self.api_endpoint else "local"

    # 定义 _rerank_local 相关的处理逻辑，供流程或外部调用复用。
    def _rerank_local(self, query: str, documents: list[str]) -> list[float]:
        # 用本地 transformers 模型给候选文档打分，适合没有远程 reranker 的环境。
        started = time.perf_counter()
        # 调用 self._ensure_local_model 完成当前步骤需要的业务处理。
        self._ensure_local_model()
        # 将 tokenizer 的值保存下来，供后续流程判断或组装响应时使用。
        tokenizer = self._tokenizer
        # 将 model 的值保存下来，供后续流程判断或组装响应时使用。
        model = self._model
        # 将 torch 的值保存下来，供后续流程判断或组装响应时使用。
        torch = self._torch
        # 将 inputs 的值保存下来，供后续流程判断或组装响应时使用。
        inputs = tokenizer(
            # 调用 len 完成当前步骤需要的业务处理。
            [query] * len(documents),
            # 执行当前业务步骤，推动流程继续向下游推进。
            documents,
            # 将 padding 的值保存下来，供后续流程判断或组装响应时使用。
            padding=True,
            # 将 truncation 的值保存下来，供后续流程判断或组装响应时使用。
            truncation=True,
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return_tensors="pt",
        )
        # 将 inputs 的值保存下来，供后续流程判断或组装响应时使用。
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        # 调用 model.eval 完成当前步骤需要的业务处理。
        model.eval()
        # 进入上下文管理器保护的区域，自动处理资源生命周期。
        with torch.no_grad():
            # 将 logits 的值保存下来，供后续流程判断或组装响应时使用。
            logits = model(**inputs).logits
        # 根据 len(logits.shape) == 2 and logits.shape[-1] > 1 判断当前流程该进入哪个处理分支。
        if len(logits.shape) == 2 and logits.shape[-1] > 1:
            # 将 scores_tensor 的值保存下来，供后续流程判断或组装响应时使用。
            scores_tensor = torch.softmax(logits, dim=-1)[:, -1]
        # 处理前面条件都不满足时的默认分支。
        else:
            # 将 scores_tensor 的值保存下来，供后续流程判断或组装响应时使用。
            scores_tensor = torch.sigmoid(logits.reshape(-1))
        # 将 scores 的值保存下来，供后续流程判断或组装响应时使用。
        scores = [float(item) for item in scores_tensor.detach().cpu().tolist()]
        # 将 normalized 的值保存下来，供后续流程判断或组装响应时使用。
        normalized = _normalize_scores(scores)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Reranker local completed model=%s device=%s docs=%s scores=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.model_name,
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.device,
            # 调用 len 完成当前步骤需要的业务处理。
            len(documents),
            # 调用 _format_scores 完成当前步骤需要的业务处理。
            _format_scores(normalized),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return normalized

    # 定义 _ensure_local_model 相关的处理逻辑，供流程或外部调用复用。
    def _ensure_local_model(self) -> None:
        # 首次本地重排时才加载 tokenizer 和模型，避免启动阶段占用大量内存。
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("transformers and torch are required for local reranker inference.") from exc

        # 将 self._torch 的值保存下来，供后续流程判断或组装响应时使用。
        self._torch = torch
        # 将 self._tokenizer 的值保存下来，供后续流程判断或组装响应时使用。
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 将 self._model 的值保存下来，供后续流程判断或组装响应时使用。
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)


# 定义 prune_by_differential_strategy 相关的处理逻辑，供流程或外部调用复用。
def prune_by_differential_strategy(
    # 执行当前业务步骤，推动流程继续向下游推进。
    query: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    candidates: List[Dict],
    # 执行当前业务步骤，推动流程继续向下游推进。
    top_k: int,
    # 将 config 的值保存下来，供后续流程判断或组装响应时使用。
    config: Dict | None = None,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> List[Dict]:
    # 根据候选来源选择不同裁剪策略，历史案例重业务元数据，静态文档优先走 reranker。
    if not candidates or len(candidates) <= top_k:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("RAG pruning skipped source_type=%s candidates=%s top_k=%s", candidates[0].get("source_type") if candidates else None, len(candidates), top_k)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return candidates

    # 将 merged_config 的值保存下来，供后续流程判断或组装响应时使用。
    merged_config = {**default_config, **(config or {})}
    # 将 source_type 的值保存下来，供后续流程判断或组装响应时使用。
    source_type = str(candidates[0].get("source_type") or "").lower()
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info("RAG pruning start source_type=%s candidates=%s top_k=%s", source_type, len(candidates), top_k)
    # 根据 source_type in {"history", "message_history"} 判断当前流程该进入哪个处理分支。
    if source_type in {"history", "message_history"}:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _prune_history(query, candidates, top_k, merged_config)
    # 根据 source_type in {"static", "static_doc"} 判断当前流程该进入哪个处理分支。
    if source_type in {"static", "static_doc"}:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _prune_static(query, candidates, top_k, merged_config)

    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.warning("Unknown RAG source_type=%s; falling back to vector score ordering.", source_type)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return sorted(candidates, key=lambda item: _bounded_score(item.get("score")), reverse=True)[:top_k]


# 定义 _prune_history 相关的处理逻辑，供流程或外部调用复用。
def _prune_history(query: str, candidates: list[dict], top_k: int, config: dict[str, Any]) -> list[dict]:
    # 裁剪历史案例候选，把向量相似度和案例质量元数据合成一个最终排序分。
    started = time.perf_counter()
    # 将 alpha 的值保存下来，供后续流程判断或组装响应时使用。
    alpha = float(config.get("alpha_history", 0.6))
    # 将 ranked 的值保存下来，供后续流程判断或组装响应时使用。
    ranked: list[dict] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for candidate in candidates:
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        # 将 metadata_score 的值保存下来，供后续流程判断或组装响应时使用。
        metadata_score = compute_history_metadata_score(metadata, query)
        # 将 original_score 的值保存下来，供后续流程判断或组装响应时使用。
        original_score = _bounded_score(candidate.get("score"))
        # 将 combined_score 的值保存下来，供后续流程判断或组装响应时使用。
        combined_score = alpha * original_score + (1 - alpha) * metadata_score
        # 把当前结果追加到集合中，逐步构建最终输出。
        ranked.append(
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                **candidate,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "metadata_score": metadata_score,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "combined_score": combined_score,
            }
        )
    # 将 selected 的值保存下来，供后续流程判断或组装响应时使用。
    selected = sorted(ranked, key=lambda item: item["combined_score"], reverse=True)[:top_k]
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info(
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "History metadata pruning completed candidates=%s selected=%s alpha=%s selected_scores=%s elapsed_ms=%s",
        # 调用 len 完成当前步骤需要的业务处理。
        len(candidates),
        # 调用 len 完成当前步骤需要的业务处理。
        len(selected),
        # 执行当前业务步骤，推动流程继续向下游推进。
        alpha,
        # 调用 _format_candidate_scores 完成当前步骤需要的业务处理。
        _format_candidate_scores(selected, "combined_score"),
        # 调用 int 完成当前步骤需要的业务处理。
        int((time.perf_counter() - started) * 1000),
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return selected


# 定义 _prune_static 相关的处理逻辑，供流程或外部调用复用。
def _prune_static(query: str, candidates: list[dict], top_k: int, config: dict[str, Any]) -> list[dict]:
    # 裁剪静态知识文档，优先用 reranker 判断哪些文档最值得放进提示词。
    started = time.perf_counter()
    # 将 documents 的值保存下来，供后续流程判断或组装响应时使用。
    documents = [str(candidate.get("text") or "") for candidate in candidates]
    # 将 reranker 的值保存下来，供后续流程判断或组装响应时使用。
    reranker = Reranker(
        # 将 model_name 的值保存下来，供后续流程判断或组装响应时使用。
        model_name=str(config.get("reranker_model_name") or default_config["reranker_model_name"]),
        # 将 device 的值保存下来，供后续流程判断或组装响应时使用。
        device=str(config.get("reranker_device") or default_config["reranker_device"]),
        # 将 api_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
        api_endpoint=config.get("reranker_api_endpoint"),
        # 将 api_key 的值保存下来，供后续流程判断或组装响应时使用。
        api_key=config.get("reranker_api_key"),
        # 将 api_format 的值保存下来，供后续流程判断或组装响应时使用。
        api_format=str(config.get("reranker_api_format") or default_config["reranker_api_format"]),
        # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        timeout_seconds=int(config.get("reranker_timeout_seconds") or default_config["reranker_timeout_seconds"]),
    )
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 将 scores 的值保存下来，供后续流程判断或组装响应时使用。
        scores = reranker.rerank(query, documents)
        # 将 ranked 的值保存下来，供后续流程判断或组装响应时使用。
        ranked = [
            # 调用 _bounded_score 完成当前步骤需要的业务处理。
            {**candidate, "rerank_score": _bounded_score(score)}
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        # 将 selected 的值保存下来，供后续流程判断或组装响应时使用。
        selected = sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Static reranker pruning completed candidates=%s selected=%s model=%s api_mode=%s selected_scores=%s elapsed_ms=%s",
            # 调用 len 完成当前步骤需要的业务处理。
            len(candidates),
            # 调用 len 完成当前步骤需要的业务处理。
            len(selected),
            # 执行当前业务步骤，推动流程继续向下游推进。
            reranker.model_name,
            # 调用 reranker.api_mode 完成当前步骤需要的业务处理。
            reranker.api_mode(),
            # 调用 _format_candidate_scores 完成当前步骤需要的业务处理。
            _format_candidate_scores(selected, "rerank_score"),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return selected
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception as exc:
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.warning("Reranker unavailable; falling back to static metadata pruning: %s", exc)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _prune_static_fallback(query, candidates, top_k, config)


# 定义 _prune_static_fallback 相关的处理逻辑，供流程或外部调用复用。
def _prune_static_fallback(query: str, candidates: list[dict], top_k: int, config: dict[str, Any]) -> list[dict]:
    # reranker 不可用时，用向量分和文档元数据分做保守排序。
    started = time.perf_counter()
    # 将 alpha 的值保存下来，供后续流程判断或组装响应时使用。
    alpha = float(config.get("alpha_static_fallback", 0.7))
    # 将 query_features 的值保存下来，供后续流程判断或组装响应时使用。
    query_features = extract_query_features(query)
    # 将 ranked 的值保存下来，供后续流程判断或组装响应时使用。
    ranked: list[dict] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for candidate in candidates:
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        # 将 metadata_score 的值保存下来，供后续流程判断或组装响应时使用。
        metadata_score = compute_static_metadata_score(metadata, query_features)
        # 将 original_score 的值保存下来，供后续流程判断或组装响应时使用。
        original_score = _bounded_score(candidate.get("score"))
        # 将 combined_score 的值保存下来，供后续流程判断或组装响应时使用。
        combined_score = alpha * original_score + (1 - alpha) * metadata_score
        # 把当前结果追加到集合中，逐步构建最终输出。
        ranked.append(
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                **candidate,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "metadata_score": metadata_score,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "combined_score": combined_score,
            }
        )
    # 将 selected 的值保存下来，供后续流程判断或组装响应时使用。
    selected = sorted(ranked, key=lambda item: item["combined_score"], reverse=True)[:top_k]
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.info(
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "Static metadata fallback pruning completed candidates=%s selected=%s alpha=%s selected_scores=%s elapsed_ms=%s",
        # 调用 len 完成当前步骤需要的业务处理。
        len(candidates),
        # 调用 len 完成当前步骤需要的业务处理。
        len(selected),
        # 执行当前业务步骤，推动流程继续向下游推进。
        alpha,
        # 调用 _format_candidate_scores 完成当前步骤需要的业务处理。
        _format_candidate_scores(selected, "combined_score"),
        # 调用 int 完成当前步骤需要的业务处理。
        int((time.perf_counter() - started) * 1000),
    )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return selected


# 定义 _normalize_text 相关的处理逻辑，供流程或外部调用复用。
def _normalize_text(text: str) -> str:
    # 把文本统一成小写并去掉标点，方便后续关键词匹配和分词。
    return re.sub(r"[^\w\s]", " ", str(text).lower()).strip()


# 定义 _tokenize 相关的处理逻辑，供流程或外部调用复用。
def _tokenize(text: str) -> list[str]:
    # 从归一化文本中提取词语列表，用于相似度和关键词命中计算。
    normalized = _normalize_text(text)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return re.findall(r"\w+", normalized)


# 定义 _as_list 相关的处理逻辑，供流程或外部调用复用。
def _as_list(value: Any) -> list[Any]:
    # 把单个值、元组、集合等统一转成列表，方便按多值字段处理。
    if value is None:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # 根据 isinstance(value, list) 判断当前流程该进入哪个处理分支。
    if isinstance(value, list):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return value
    # 根据 isinstance(value, tuple | set) 判断当前流程该进入哪个处理分支。
    if isinstance(value, tuple | set):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return list(value)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return [value]


# 定义 _bounded_score 相关的处理逻辑，供流程或外部调用复用。
def _bounded_score(value: Any) -> float:
    # 把任意分数压到 0 到 1 之间，避免异常值影响排序。
    try:
        # 将 score 的值保存下来，供后续流程判断或组装响应时使用。
        score = float(value)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except (TypeError, ValueError):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0
    # 根据 math.isnan(score) or math.isinf(score) 判断当前流程该进入哪个处理分支。
    if math.isnan(score) or math.isinf(score):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return 0.0
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return max(0.0, min(1.0, score))


# 定义 _score_from_result 相关的处理逻辑，供流程或外部调用复用。
def _score_from_result(item: dict[str, Any]) -> Any:
    # 兼容不同 reranker 返回字段名，提取候选文档的相关性分数。
    return item.get("relevance_score", item.get("score", item.get("rerank_score", 0.0)))


# 定义 _is_dashscope_endpoint 相关的处理逻辑，供流程或外部调用复用。
def _is_dashscope_endpoint(endpoint: str) -> bool:
    # 判断 reranker 地址是否是 DashScope 原生接口，用来选择请求和鉴权方式。
    normalized = endpoint.lower()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "dashscope" in normalized or "/services/rerank/" in normalized


# 定义 _is_dashscope_compatible_endpoint 相关的处理逻辑，供流程或外部调用复用。
def _is_dashscope_compatible_endpoint(endpoint: str) -> bool:
    # 判断 reranker 地址是否是 DashScope 兼容模式接口。
    return "/compatible-api/" in endpoint.lower() or "/compatible-mode/" in endpoint.lower()


# 定义 _endpoint_host 相关的处理逻辑，供流程或外部调用复用。
def _endpoint_host(endpoint: str) -> str:
    # 从完整接口地址中提取主机名，日志里只展示主机可减少噪声。
    try:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return urlparse(endpoint).netloc or endpoint
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return endpoint


# 定义 _format_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_scores(scores: list[float], limit: int = 5) -> str:
    # 把前几名分数格式化成短字符串，方便日志观察排序效果。
    if not scores:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(scores) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(f"{score:.4f}" for score in scores[:limit]) + suffix + "]"


# 定义 _format_candidate_scores 相关的处理逻辑，供流程或外部调用复用。
def _format_candidate_scores(candidates: list[dict], score_key: str, limit: int = 5) -> str:
    # 把候选文档的索引和分数格式化到日志，方便定位哪条资料被选中。
    if not candidates:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "[]"
    # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
    values = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for item in candidates[:limit]:
        # 把当前结果追加到集合中，逐步构建最终输出。
        values.append(f"{item.get('_index', '?')}:{_bounded_score(item.get(score_key)):.4f}")
    # 将 suffix 的值保存下来，供后续流程判断或组装响应时使用。
    suffix = ", ..." if len(candidates) > limit else ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "[" + ", ".join(values) + suffix + "]"


# 定义 _normalize_scores 相关的处理逻辑，供流程或外部调用复用。
def _normalize_scores(scores: list[float]) -> list[float]:
    # 把本地模型输出的原始分数归一化，保证排序阶段使用稳定的 0 到 1 分数。
    if not scores:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # 将 bounded 的值保存下来，供后续流程判断或组装响应时使用。
    bounded = [_bounded_score(score) for score in scores]
    # 根据 any(score != 0.0 for score in bounded) 判断当前流程该进入哪个处理分支。
    if any(score != 0.0 for score in bounded):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return bounded
    # 根据 np is None 判断当前流程该进入哪个处理分支。
    if np is None:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return bounded
    # 将 array 的值保存下来，供后续流程判断或组装响应时使用。
    array = np.asarray(scores, dtype=float)
    # 将 min_score 的值保存下来，供后续流程判断或组装响应时使用。
    min_score = float(array.min())
    # 将 max_score 的值保存下来，供后续流程判断或组装响应时使用。
    max_score = float(array.max())
    # 根据 max_score == min_score 判断当前流程该进入哪个处理分支。
    if max_score == min_score:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [0.0 for _ in scores]
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return [float(item) for item in ((array - min_score) / (max_score - min_score)).tolist()]


# 根据 __name__ == "__main__" 判断当前流程该进入哪个处理分支。
if __name__ == "__main__":
    # 将 history_candidates 的值保存下来，供后续流程判断或组装响应时使用。
    history_candidates = [
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text": "Redis timeout resolved by lowering retry concurrency.",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "score": 0.82,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "metadata": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "alert_summary": "Redis 连接超时",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "validation_result": True,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "need_human": False,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "human_decision": "approved",
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source_type": "history",
        },
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text": "Unrelated rejected diagnosis.",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "score": 0.91,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "metadata": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "alert_summary": "订单同步失败",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "validation_result": False,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "need_human": True,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "human_decision": "rejected",
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source_type": "history",
        },
    ]
    # 将 static_candidates 的值保存下来，供后续流程判断或组装响应时使用。
    static_candidates = [
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text": "Redis connection timeout with 500 errors.",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "score": 0.72,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "metadata": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "alert_category": "Redis",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "severity_level": "P1",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "error_code": ["500"],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "keywords": ["连接", "超时"],
                # 执行当前业务步骤，推动流程继续向下游推进。
                "section": "Redis/连接池/超时",
            },
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source_type": "static",
        },
        {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text": "MQ retry backlog.",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "score": 0.8,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "metadata": {"alert_category": "MQ", "keywords": ["重试"]},
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source_type": "static",
        },
    ]
    # 调用 print 完成当前步骤需要的业务处理。
    print(prune_by_differential_strategy("Redis 连接超时，错误码 500", history_candidates, 1))
    # 调用 print 完成当前步骤需要的业务处理。
    print(prune_by_differential_strategy("Redis 连接超时，错误码 500", static_candidates, 1))
