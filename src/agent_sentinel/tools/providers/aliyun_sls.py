from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from agent_sentinel.monitoring import monitor
from agent_sentinel.utils.retry_utils import async_retry

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 日志级别关键词
LOG_LEVELS = ["ERROR", "WARN", "WARNING", "FATAL", "CRITICAL", "EXCEPTION", "EXCEPTION"]

# 常见服务名模式
SERVICE_PATTERNS = [
    # 执行当前业务步骤，推动流程继续向下游推进。
    r"[\w-]+-service",
    # 执行当前业务步骤，推动流程继续向下游推进。
    r"[\w-]+-api",
    # 执行当前业务步骤，推动流程继续向下游推进。
    r"[\w-]+-sync",
    # 执行当前业务步骤，推动流程继续向下游推进。
    r"[\w-]+-worker",
    # 执行当前业务步骤，推动流程继续向下游推进。
    r"[\w-]+-consumer",
]


# 定义 _iter_sls_logs 相关的处理逻辑，供流程或外部调用复用。
def _iter_sls_logs(result: Any) -> list[Any]:
    # 兼容 SLS SDK 对象和测试字典两种结果形态，统一取出日志列表。
    if hasattr(result, "get_logs"):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return list(result.get_logs())
    # 根据 isinstance(result, dict) 判断当前流程该进入哪个处理分支。
    if isinstance(result, dict):
        # 将 logs 的值保存下来，供后续流程判断或组装响应时使用。
        logs = result.get("logs") or result.get("data") or []
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return list(logs) if isinstance(logs, list) else []
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return []


# 定义 _log_contents 相关的处理逻辑，供流程或外部调用复用。
def _log_contents(log: Any) -> Any:
    # 兼容不同日志对象结构，统一提取一条日志的字段内容。
    if hasattr(log, "get_contents"):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return log.get_contents()
    # 根据 isinstance(log, dict) 判断当前流程该进入哪个处理分支。
    if isinstance(log, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return log.get("contents") or log.get("content") or log
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return str(log)


# 定义 _log_time 相关的处理逻辑，供流程或外部调用复用。
def _log_time(log: Any) -> Any:
    # 兼容不同日志对象结构，统一提取日志时间戳。
    if hasattr(log, "get_time"):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return log.get_time()
    # 根据 isinstance(log, dict) 判断当前流程该进入哪个处理分支。
    if isinstance(log, dict):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return log.get("time") or log.get("__time__") or log.get("timestamp")
    return None


# 定义 _extract_message 相关的处理逻辑，供流程或外部调用复用。
def _extract_message(contents: dict[str, Any]) -> str:
    # 从常见字段里提取日志正文，兼容 message、msg、content 等命名。
    message = contents.get("message") or contents.get("msg") or contents.get("content") or ""
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return str(message)


# 定义 AliyunSLSClient 组件，集中管理这个模块的状态和行为。
class AliyunSLSClient:
    """阿里云 SLS 日志服务客户端

    封装 aliyun-log-python-sdk，提供异步接口。
    SDK 是同步的，通过 run_in_executor 异步化。
    """

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        access_key_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        access_key_secret: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        endpoint: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        project: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        logstore: str,
        # 将 timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        timeout_seconds: int = 10,
        # 将 max_lines 的值保存下来，供后续流程判断或组装响应时使用。
        max_lines: int = 100,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 保存 SLS 鉴权、项目、日志库和查询限制，SDK 客户端会在首次查询时懒加载。
        self._access_key_id = access_key_id
        # 将 self._access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
        self._access_key_secret = access_key_secret
        # 将 self._endpoint 的值保存下来，供后续流程判断或组装响应时使用。
        self._endpoint = endpoint
        # 将 self._project 的值保存下来，供后续流程判断或组装响应时使用。
        self._project = project
        # 将 self._logstore 的值保存下来，供后续流程判断或组装响应时使用。
        self._logstore = logstore
        # 将 self._timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        self._timeout_seconds = timeout_seconds
        # 将 self._max_lines 的值保存下来，供后续流程判断或组装响应时使用。
        self._max_lines = max_lines
        # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
        self._client = None

    # 定义 _get_client 相关的处理逻辑，供流程或外部调用复用。
    def _get_client(self):
        """延迟初始化 SLS 客户端"""
        # 首次查询日志时才创建 SLS SDK 客户端，减少启动时对云 SDK 的依赖。
        if self._client is None:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                from aliyun.log.logclient import LogClient

                # 将 self._client 的值保存下来，供后续流程判断或组装响应时使用。
                self._client = LogClient(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    self._endpoint,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    self._access_key_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    self._access_key_secret,
                )
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except ImportError:
                # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
                raise ImportError(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "aliyun-log-python-sdk is required for SLS provider. "
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Install it with: pip install aliyun-log-python-sdk"
                )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self._client

    @async_retry(max_attempts=2)
    # 定义 query_logs 相关的处理逻辑，供流程或外部调用复用。
    async def query_logs(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        query: str,
        # 将 from_time 的值保存下来，供后续流程判断或组装响应时使用。
        from_time: int | None = None,
        # 将 to_time 的值保存下来，供后续流程判断或组装响应时使用。
        to_time: int | None = None,
        # 将 max_lines 的值保存下来，供后续流程判断或组装响应时使用。
        max_lines: int | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
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
        # 执行一条 SLS 查询语句，并把云 SDK 返回的日志整理成统一字典列表。
        from aliyun.log import GetLogsRequest
        from aliyun.log.logclient import LogException

        # 将 client 的值保存下来，供后续流程判断或组装响应时使用。
        client = self._get_client()
        # 将 max_lines 的值保存下来，供后续流程判断或组装响应时使用。
        max_lines = max_lines or self._max_lines

        # 默认查询最近 1 小时
        now = int(time.time())
        # 将 from_time 的值保存下来，供后续流程判断或组装响应时使用。
        from_time = from_time or (now - 3600)
        # 将 to_time 的值保存下来，供后续流程判断或组装响应时使用。
        to_time = to_time or now

        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            # 进入异步上下文管理区域，确保异步资源按约定释放。
            async with asyncio.timeout(self._timeout_seconds):
                # 将 loop 的值保存下来，供后续流程判断或组装响应时使用。
                loop = asyncio.get_event_loop()
                # 将 request 的值保存下来，供后续流程判断或组装响应时使用。
                request = GetLogsRequest(
                    # 将 project 的值保存下来，供后续流程判断或组装响应时使用。
                    project=self._project,
                    # 将 logstore 的值保存下来，供后续流程判断或组装响应时使用。
                    logstore=self._logstore,
                    # 将 fromTime 的值保存下来，供后续流程判断或组装响应时使用。
                    fromTime=from_time,
                    # 将 toTime 的值保存下来，供后续流程判断或组装响应时使用。
                    toTime=to_time,
                    # 将 topic 的值保存下来，供后续流程判断或组装响应时使用。
                    topic="",
                    # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
                    query=query,
                    # 将 line 的值保存下来，供后续流程判断或组装响应时使用。
                    line=max_lines,
                    # 将 offset 的值保存下来，供后续流程判断或组装响应时使用。
                    offset=0,
                    # 将 reverse 的值保存下来，供后续流程判断或组装响应时使用。
                    reverse=False,
                )
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = await loop.run_in_executor(
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    None,
                    # 调用 client.get_logs 完成当前步骤需要的业务处理。
                    lambda: client.get_logs(request),
                )

                # 将 logs 的值保存下来，供后续流程判断或组装响应时使用。
                logs = []
                # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
                for log in _iter_sls_logs(result):
                    # 将 contents 的值保存下来，供后续流程判断或组装响应时使用。
                    contents = _log_contents(log)
                    # 将 log_entry 的值保存下来，供后续流程判断或组装响应时使用。
                    log_entry = {
                        # 调用 _log_time 完成当前步骤需要的业务处理。
                        "timestamp": _log_time(log),
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "content": contents,
                    }

                    # 提取常见字段
                    if isinstance(contents, dict):
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        log_entry["level"] = contents.get("level", contents.get("severity", ""))
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        log_entry["service"] = contents.get("service", contents.get("app", ""))
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        log_entry["message"] = _extract_message(contents)
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        log_entry["trace_id"] = contents.get("trace_id", contents.get("traceId", ""))
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        log_entry["span_id"] = contents.get("span_id", contents.get("spanId", ""))
                    # 处理前面条件都不满足时的默认分支。
                    else:
                        # 保存当前计算结果，供后续流程判断或组装响应时使用。
                        log_entry["message"] = str(contents)

                    # 把当前结果追加到集合中，逐步构建最终输出。
                    logs.append(log_entry)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return logs

        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except LogException as e:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.error("SLS query failed: code=%s message=%s", e.get_error_code(), e.get_error_message())
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception as e:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.error("SLS query unexpected error: %s", e)
            # 执行当前业务步骤，推动流程继续向下游推进。
            raise


# 定义 SLSLogsProvider 组件，集中管理这个模块的状态和行为。
class SLSLogsProvider:
    """基于 SLS 的日志查询提供者"""

    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, client: AliyunSLSClient) -> None:
        # 绑定 SLS 客户端，把阿里云日志查询能力包装成项目统一的日志 provider。
        self._client = client

    # 定义 query_logs 相关的处理逻辑，供流程或外部调用复用。
    async def query_logs(self, alert_summary: str, group_id: str | None = None) -> dict[str, Any]:
        """查询与告警相关的日志

        Args:
            alert_summary: 告警摘要文本
            group_id: 飞书群组 ID（可用于映射到不同的 project/logstore）

        Returns:
            包含 matches 列表和统计信息的字典
        """
        # 根据告警摘要生成候选 SLS 查询，返回命中的日志、级别统计和错误摘要。
        logger.info("Querying SLS logs summary_chars=%s group_id=%s", len(alert_summary), group_id)

        # 将 queries 的值保存下来，供后续流程判断或组装响应时使用。
        queries = self._build_queries(alert_summary)
        # 将 logs 的值保存下来，供后续流程判断或组装响应时使用。
        logs: list[dict[str, Any]] = []
        # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
        query = queries[0]
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for candidate in queries:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("SLS query: %s", candidate)
            # 将 logs 的值保存下来，供后续流程判断或组装响应时使用。
            logs = await self._client.query_logs(query=candidate)
            # 将 query 的值保存下来，供后续流程判断或组装响应时使用。
            query = candidate
            # 根据 logs 判断当前流程该进入哪个处理分支。
            if logs:
                # 满足停止条件后退出循环，避免继续执行无效处理。
                break

        # 调用 monitor.record_tool_call 完成当前步骤需要的业务处理。
        monitor.record_tool_call("query_logs", group_id)

        # 统计日志级别分布
        level_stats = {}
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for log in logs:
            # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
            level = log.get("level", "UNKNOWN")
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            level_stats[level] = level_stats.get(level, 0) + 1

        # 提取错误消息
        error_messages = []
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for log in logs:
            # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
            level = log.get("level", "").upper()
            # 根据 level in ["ERROR", "FATAL", "CRITICAL", "EXCEPTION"] 判断当前流程该进入哪个处理分支。
            if level in ["ERROR", "FATAL", "CRITICAL", "EXCEPTION"]:
                # 把当前结果追加到集合中，逐步构建最终输出。
                error_messages.append(log.get("message", str(log.get("content", ""))))

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "matches": [
                {
                    # 调用 log.get 完成当前步骤需要的业务处理。
                    "timestamp": log.get("timestamp"),
                    # 调用 log.get 完成当前步骤需要的业务处理。
                    "level": log.get("level", ""),
                    # 调用 log.get 完成当前步骤需要的业务处理。
                    "service": log.get("service", ""),
                    # 调用 log.get 完成当前步骤需要的业务处理。
                    "message": log.get("message", str(log.get("content", ""))),
                    # 调用 log.get 完成当前步骤需要的业务处理。
                    "trace_id": log.get("trace_id", ""),
                }
                # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
                for log in logs
            ],
            # 调用 len 完成当前步骤需要的业务处理。
            "total": len(logs),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "level_stats": level_stats,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "error_messages": error_messages[:5],  # 最多返回 5 条错误消息
            # 执行当前业务步骤，推动流程继续向下游推进。
            "query": query,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "provider": "aliyun_sls",
        }

    # 定义 _build_queries 相关的处理逻辑，供流程或外部调用复用。
    def _build_queries(self, alert_summary: str) -> list[str]:
        # 根据告警摘要生成多条候选查询，先精确匹配原文，再退回关键词模糊查询。
        queries = []
        # 将 raw_query 的值保存下来，供后续流程判断或组装响应时使用。
        raw_query = self._build_raw_query(alert_summary)
        # 根据 raw_query 判断当前流程该进入哪个处理分支。
        if raw_query:
            # 把当前结果追加到集合中，逐步构建最终输出。
            queries.append(raw_query)

        # 将 fuzzy_query 的值保存下来，供后续流程判断或组装响应时使用。
        fuzzy_query = self._build_query(alert_summary)
        # 根据 fuzzy_query not in queries 判断当前流程该进入哪个处理分支。
        if fuzzy_query not in queries:
            # 把当前结果追加到集合中，逐步构建最终输出。
            queries.append(fuzzy_query)

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return queries or ["*"]

    # 定义 _build_raw_query 相关的处理逻辑，供流程或外部调用复用。
    def _build_raw_query(self, alert_summary: str) -> str | None:
        # 把完整告警摘要转成 SLS 精确短语查询，用于优先命中原始日志。
        text = alert_summary.strip()
        # 根据 not text 判断当前流程该进入哪个处理分支。
        if not text:
            return None
        # 将 escaped 的值保存下来，供后续流程判断或组装响应时使用。
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return f'"{escaped}"'

    # 定义 _build_query 相关的处理逻辑，供流程或外部调用复用。
    def _build_query(self, alert_summary: str) -> str:
        """根据告警摘要构建 SLS 查询语句

        智能分析告警摘要，提取日志级别、服务名和关键词，构建高效的查询语句。

        Args:
            alert_summary: 告警摘要文本

        Returns:
            SLS 查询语句（LOGQL 语法）
        """
        # 从告警摘要中提取日志级别、服务名和关键词，拼成可执行的 SLS 查询语句。
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
            # 把当前结果追加到集合中，逐步构建最终输出。
            conditions.append(f'level: {log_level}')

        # 添加服务名条件
        if service_name:
            # 把当前结果追加到集合中，逐步构建最终输出。
            conditions.append(f'service: "{service_name}"')

        # 添加关键词条件
        if keywords:
            # 将 keyword_query 的值保存下来，供后续流程判断或组装响应时使用。
            keyword_query = " OR ".join([f'"{kw}"' for kw in keywords[:3]])
            # 把当前结果追加到集合中，逐步构建最终输出。
            conditions.append(f'({keyword_query})')

        # 组合查询条件
        if not conditions:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return '* AND (level: ERROR OR level: WARN)'

        # 根据 len(conditions) == 1 判断当前流程该进入哪个处理分支。
        if len(conditions) == 1:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return conditions[0]

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return " AND ".join(conditions)

    # 定义 _extract_log_level 相关的处理逻辑，供流程或外部调用复用。
    def _extract_log_level(self, text: str) -> str | None:
        """从文本中提取日志级别"""
        # 从告警文字中识别 ERROR、WARN 等日志级别，帮助缩小日志查询范围。
        text_upper = text.upper()
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for level in LOG_LEVELS:
            # 根据 level in text_upper 判断当前流程该进入哪个处理分支。
            if level in text_upper:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return level
        return None

    # 定义 _extract_service_name 相关的处理逻辑，供流程或外部调用复用。
    def _extract_service_name(self, text: str) -> str | None:
        """从文本中提取服务名"""
        # 从告警文字中识别常见服务名，例如 xxx-service 或 xxx-api。
        for pattern in SERVICE_PATTERNS:
            # 根据当前值进入模式匹配，选择对应的处理分支。
            match = re.search(pattern, text, re.IGNORECASE)
            # 根据 match 判断当前流程该进入哪个处理分支。
            if match:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return match.group(0)
        return None

    # 定义 _extract_keywords 相关的处理逻辑，供流程或外部调用复用。
    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词

        过滤停用词，提取有意义的关键词。
        """
        # 从告警文字中抽取可用于日志搜索的关键词，并过滤常见无意义词。
        # 停用词列表
        stop_words = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "让", "用", "对",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "等", "但", "而", "如果", "虽然", "因为", "所以", "这个", "那个", "什么", "怎么",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "before", "after", "above", "below", "between", "out", "off", "over",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "under", "again", "further", "then", "once", "and", "but", "or",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "nor", "not", "so", "very", "just", "than", "too", "also",
        }

        # 提取单词（支持中英文）
        words = re.findall(r'[\w\u4e00-\u9fff]+', text)

        # 过滤停用词和短词
        keywords = [
            # 执行当前业务步骤，推动流程继续向下游推进。
            word for word in words
            # 根据 len(word) > 2 and word.lower() not in stop_words 判断当前流程该进入哪个处理分支。
            if len(word) > 2 and word.lower() not in stop_words
        ]

        # 去重并限制数量
        seen = set()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = []
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for kw in keywords:
            # 根据 kw.lower() not in seen 判断当前流程该进入哪个处理分支。
            if kw.lower() not in seen:
                # 调用 seen.add 完成当前步骤需要的业务处理。
                seen.add(kw.lower())
                # 把当前结果追加到集合中，逐步构建最终输出。
                result.append(kw)

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return result[:5]
