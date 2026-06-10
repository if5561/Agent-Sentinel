from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.utils.retry_utils import async_retry

logger = logging.getLogger(__name__)

# 日志级别关键词
LOG_LEVELS = ["ERROR", "WARN", "WARNING", "FATAL", "CRITICAL", "EXCEPTION", "EXCEPTION"]

# 常见服务名模式
SERVICE_PATTERNS = [
    r"[\w-]+-service",
    r"[\w-]+-api",
    r"[\w-]+-sync",
    r"[\w-]+-worker",
    r"[\w-]+-consumer",
]


def _iter_sls_logs(result: Any) -> list[Any]:
    # 方法说明：兼容 SLS SDK 对象和测试字典两种结果形态，统一取出日志列表。
    if hasattr(result, "get_logs"):
        return list(result.get_logs())
    if isinstance(result, dict):
        logs = result.get("logs") or result.get("data") or []
        return list(logs) if isinstance(logs, list) else []
    return []


def _log_contents(log: Any) -> Any:
    # 方法说明：兼容不同日志对象结构，统一提取一条日志的字段内容。
    if hasattr(log, "get_contents"):
        return log.get_contents()
    if isinstance(log, dict):
        return log.get("contents") or log.get("content") or log
    return str(log)


def _log_time(log: Any) -> Any:
    # 方法说明：兼容不同日志对象结构，统一提取日志时间戳。
    if hasattr(log, "get_time"):
        return log.get_time()
    if isinstance(log, dict):
        return log.get("time") or log.get("__time__") or log.get("timestamp")
    return None


def _extract_message(contents: dict[str, Any]) -> str:
    # 方法说明：从常见字段里提取日志正文，兼容 message、msg、content 等命名。
    message = contents.get("message") or contents.get("msg") or contents.get("content") or ""
    return str(message)


class AliyunSLSClient:
    """阿里云 SLS 日志服务客户端

    封装 aliyun-log-python-sdk，提供异步接口。
    SDK 是同步的，通过 run_in_executor 异步化。
    """

    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str,
        project: str,
        logstore: str,
        timeout_seconds: int = 10,
        max_lines: int = 100,
    ) -> None:
        # 方法说明：保存 SLS 鉴权、项目、日志库和查询限制，SDK 客户端会在首次查询时懒加载。
        self._access_key_id = access_key_id
        self._access_key_secret = access_key_secret
        self._endpoint = endpoint
        self._project = project
        self._logstore = logstore
        self._timeout_seconds = timeout_seconds
        self._max_lines = max_lines
        self._client = None

    def _get_client(self):
        """延迟初始化 SLS 客户端"""
        # 方法说明：首次查询日志时才创建 SLS SDK 客户端，减少启动时对云 SDK 的依赖。
        if self._client is None:
            try:
                from aliyun.log.logclient import LogClient

                self._client = LogClient(
                    self._endpoint,
                    self._access_key_id,
                    self._access_key_secret,
                )
            except ImportError:
                raise ImportError(
                    "aliyun-log-python-sdk is required for SLS provider. "
                    "Install it with: pip install aliyun-log-python-sdk"
                )
        return self._client

    @async_retry(max_attempts=2)
    async def query_logs(
        self,
        query: str,
        from_time: int | None = None,
        to_time: int | None = None,
        max_lines: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询 SLS 日志

        Args:
            query: SLS 查询语句（LOGQL 语法）
            from_time: 开始时间戳（秒），默认为 1 小时前
            to_time: 结束时间戳（秒），默认为当前时间
            max_lines: 最大返回行数

        Returns:
            日志列表，每条包含 timestamp 和 content
        """
        # 方法说明：执行一条 SLS 查询语句，并把云 SDK 返回的日志整理成统一字典列表。
        from aliyun.log import GetLogsRequest
        from aliyun.log.logclient import LogException

        client = self._get_client()
        max_lines = max_lines or self._max_lines

        # 默认查询最近 1 小时
        now = int(time.time())
        from_time = from_time or (now - 3600)
        to_time = to_time or now

        try:
            async with asyncio.timeout(self._timeout_seconds):
                loop = asyncio.get_event_loop()
                request = GetLogsRequest(
                    project=self._project,
                    logstore=self._logstore,
                    fromTime=from_time,
                    toTime=to_time,
                    topic="",
                    query=query,
                    line=max_lines,
                    offset=0,
                    reverse=False,
                )
                result = await loop.run_in_executor(
                    None,
                    lambda: client.get_logs(request),
                )

                logs = []
                for log in _iter_sls_logs(result):
                    contents = _log_contents(log)
                    log_entry = {
                        "timestamp": _log_time(log),
                        "content": contents,
                    }

                    # 提取常见字段
                    if isinstance(contents, dict):
                        log_entry["level"] = contents.get("level", contents.get("severity", ""))
                        log_entry["service"] = contents.get("service", contents.get("app", ""))
                        log_entry["message"] = _extract_message(contents)
                        log_entry["trace_id"] = contents.get("trace_id", contents.get("traceId", ""))
                        log_entry["span_id"] = contents.get("span_id", contents.get("spanId", ""))
                    else:
                        log_entry["message"] = str(contents)

                    logs.append(log_entry)
                return logs

        except LogException as e:
            logger.error("SLS query failed: code=%s message=%s", e.get_error_code(), e.get_error_message())
            raise
        except Exception as e:
            logger.error("SLS query unexpected error: %s", e)
            raise


class SLSLogsProvider:
    """基于 SLS 的日志查询提供者"""

    def __init__(self, client: AliyunSLSClient) -> None:
        # 方法说明：绑定 SLS 客户端，把阿里云日志查询能力包装成项目统一的日志 provider。
        self._client = client

    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """查询与告警相关的日志

        Args:
            alert_summary: 告警摘要文本
            group_id: 飞书群组 ID（可用于映射到不同的 project/logstore）

        Returns:
            包含 matches 列表和统计信息的字典
        """
        # 方法说明：根据告警摘要生成候选 SLS 查询，返回命中的日志、级别统计和错误摘要。
        logger.info("Querying SLS logs summary_chars=%s group_id=%s", len(alert_summary), group_id)

        queries = self._build_queries(alert_summary)
        logs: list[dict[str, Any]] = []
        query = queries[0]
        for candidate in queries:
            logger.info("SLS query: %s", candidate)
            logs = await self._client.query_logs(query=candidate)
            query = candidate
            if logs:
                break

        monitor.record_tool_call("query_logs", group_id)

        # 统计日志级别分布
        level_stats = {}
        for log in logs:
            level = log.get("level", "UNKNOWN")
            level_stats[level] = level_stats.get(level, 0) + 1

        # 提取错误消息
        error_messages = []
        for log in logs:
            level = log.get("level", "").upper()
            if level in ["ERROR", "FATAL", "CRITICAL", "EXCEPTION"]:
                error_messages.append(log.get("message", str(log.get("content", ""))))

        return {
            "matches": [
                {
                    "timestamp": log.get("timestamp"),
                    "level": log.get("level", ""),
                    "service": log.get("service", ""),
                    "message": log.get("message", str(log.get("content", ""))),
                    "trace_id": log.get("trace_id", ""),
                }
                for log in logs
            ],
            "total": len(logs),
            "level_stats": level_stats,
            "error_messages": error_messages[:5],  # 最多返回 5 条错误消息
            "query": query,
            "provider": "aliyun_sls",
        }

    def _build_queries(self, alert_summary: str) -> list[str]:
        # 方法说明：根据告警摘要生成多条候选查询，先精确匹配原文，再退回关键词模糊查询。
        queries = []
        raw_query = self._build_raw_query(alert_summary)
        if raw_query:
            queries.append(raw_query)

        fuzzy_query = self._build_query(alert_summary)
        if fuzzy_query not in queries:
            queries.append(fuzzy_query)

        return queries or ["*"]

    def _build_raw_query(self, alert_summary: str) -> str | None:
        # 方法说明：把完整告警摘要转成 SLS 精确短语查询，用于优先命中原始日志。
        text = alert_summary.strip()
        if not text:
            return None
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _build_query(self, alert_summary: str) -> str:
        """根据告警摘要构建 SLS 查询语句

        智能分析告警摘要，提取日志级别、服务名和关键词，构建高效的查询语句。

        Args:
            alert_summary: 告警摘要文本

        Returns:
            SLS 查询语句（LOGQL 语法）
        """
        # 方法说明：从告警摘要中提取日志级别、服务名和关键词，拼成可执行的 SLS 查询语句。
        # 提取日志级别
        log_level = self._extract_log_level(alert_summary)

        # 提取服务名
        service_name = self._extract_service_name(alert_summary)

        # 提取关键词
        keywords = self._extract_keywords(alert_summary)

        # 构建查询条件
        conditions = []

        # 添加日志级别条件
        if log_level:
            conditions.append(f'level: {log_level}')

        # 添加服务名条件
        if service_name:
            conditions.append(f'service: "{service_name}"')

        # 添加关键词条件
        if keywords:
            keyword_query = " OR ".join([f'"{kw}"' for kw in keywords[:3]])
            conditions.append(f'({keyword_query})')

        # 组合查询条件
        if not conditions:
            return '* AND (level: ERROR OR level: WARN)'

        if len(conditions) == 1:
            return conditions[0]

        return " AND ".join(conditions)

    def _extract_log_level(self, text: str) -> str | None:
        """从文本中提取日志级别"""
        # 方法说明：从告警文字中识别 ERROR、WARN 等日志级别，帮助缩小日志查询范围。
        text_upper = text.upper()
        for level in LOG_LEVELS:
            if level in text_upper:
                return level
        return None

    def _extract_service_name(self, text: str) -> str | None:
        """从文本中提取服务名"""
        # 方法说明：从告警文字中识别常见服务名，例如 xxx-service 或 xxx-api。
        for pattern in SERVICE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词

        过滤停用词，提取有意义的关键词。
        """
        # 方法说明：从告警文字中抽取可用于日志搜索的关键词，并过滤常见无意义词。
        # 停用词列表
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "让", "用", "对",
            "等", "但", "而", "如果", "虽然", "因为", "所以", "这个", "那个", "什么", "怎么",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "and", "but", "or",
            "nor", "not", "so", "very", "just", "than", "too", "also",
        }

        # 提取单词（支持中英文）
        words = re.findall(r'[\w\u4e00-\u9fff]+', text)

        # 过滤停用词和短词
        keywords = [
            word for word in words
            if len(word) > 2 and word.lower() not in stop_words
        ]

        # 去重并限制数量
        seen = set()
        result = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                result.append(kw)

        return result[:5]
