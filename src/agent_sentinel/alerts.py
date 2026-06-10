from __future__ import annotations

import base64
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import logging
import threading
import time

import requests

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 AlertEvent 组件，集中管理这个模块的状态和行为。
class AlertEvent:
    # 执行当前业务步骤，推动流程继续向下游推进。
    source: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    level: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    summary: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    details: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    dedupe_key: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    tags: list[str]
    # 执行当前业务步骤，推动流程继续向下游推进。
    created_at: str


# 定义 FeishuWebhookNotifier 组件，集中管理这个模块的状态和行为。
class FeishuWebhookNotifier:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        webhook_url: str | None,
        # 将 secret 的值保存下来，供后续流程判断或组装响应时使用。
        secret: str | None = None,
        # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
        enabled: bool = True,
        # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
        timeout: int = 10,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 保存飞书 webhook 地址、签名密钥和超时时间，后续告警通知统一从这里发出。
        self.webhook_url = webhook_url
        # 将 self.secret 的值保存下来，供后续流程判断或组装响应时使用。
        self.secret = secret
        # 将 self.enabled 的值保存下来，供后续流程判断或组装响应时使用。
        self.enabled = enabled
        # 将 self.timeout 的值保存下来，供后续流程判断或组装响应时使用。
        self.timeout = timeout

    # 定义 is_configured 相关的处理逻辑，供流程或外部调用复用。
    def is_configured(self) -> bool:
        # 判断飞书 webhook 告警通道是否可用；未配置时调用方会跳过发送。
        return self.enabled and bool(self.webhook_url)

    # 定义 _build_sign 相关的处理逻辑，供流程或外部调用复用。
    def _build_sign(self, timestamp: str) -> str:
        # 按飞书 webhook 规则生成签名，防止外部伪造告警消息。
        if not self.secret:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return ""
        # 飞书自定义机器人签名格式为 timestamp + "\n" + secret，再做 HMAC-SHA256。
        string_to_sign = f"{timestamp}\n{self.secret}".encode("utf-8")
        # 将 hmac_code 的值保存下来，供后续流程判断或组装响应时使用。
        hmac_code = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return base64.b64encode(hmac_code).decode("utf-8")

    # 定义 send_text 相关的处理逻辑，供流程或外部调用复用。
    def send_text(self, title: str, message: str) -> bool:
        # 把告警标题和正文发送到飞书 webhook，未配置或飞书返回错误时给出明确结果。
        if not self.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu alert skipped because webhook is not configured.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False

        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload: dict[str, object] = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "msg_type": "post",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "content": {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "post": {
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "zh_cn": {
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "title": title,
                        # 执行当前业务步骤，推动流程继续向下游推进。
                        "content": [[{"tag": "text", "text": message}]],
                    }
                }
            },
        }

        # 根据 self.secret 判断当前流程该进入哪个处理分支。
        if self.secret:
            # 带 secret 的 webhook 必须附带 timestamp 和 sign，防止消息被伪造。
            timestamp = str(int(time.time()))
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            payload["timestamp"] = timestamp
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            payload["sign"] = self._build_sign(timestamp)

        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()

        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu webhook returned error: {result}")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return True


# 定义 RealtimeAlertService 组件，集中管理这个模块的状态和行为。
class RealtimeAlertService:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        notifier: FeishuWebhookNotifier,
        # 执行当前业务步骤，推动流程继续向下游推进。
        app_env: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        title_prefix: str,
        # 将 dedup_window_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        dedup_window_seconds: int = 60,
        # 将 store_limit 的值保存下来，供后续流程判断或组装响应时使用。
        store_limit: int = 100,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 准备实时告警服务，保存通知器、环境信息、去重窗口和最近告警缓存。
        self.notifier = notifier
        # 将 self.app_env 的值保存下来，供后续流程判断或组装响应时使用。
        self.app_env = app_env
        # 将 self.title_prefix 的值保存下来，供后续流程判断或组装响应时使用。
        self.title_prefix = title_prefix
        # 将 self.dedup_window_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        self.dedup_window_seconds = dedup_window_seconds
        # 将 self._recent_events 的值保存下来，供后续流程判断或组装响应时使用。
        self._recent_events: deque[AlertEvent] = deque(maxlen=store_limit)
        # 将 self._dedupe_cache 的值保存下来，供后续流程判断或组装响应时使用。
        self._dedupe_cache: dict[str, float] = {}
        # 将 self._lock 的值保存下来，供后续流程判断或组装响应时使用。
        self._lock = threading.Lock()

    # 定义 report 相关的处理逻辑，供流程或外部调用复用。
    def report(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        source: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        level: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        summary: str,
        # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
        details: str | None = None,
        # 将 dedupe_key 的值保存下来，供后续流程判断或组装响应时使用。
        dedupe_key: str | None = None,
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags: list[str] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> tuple[bool, bool]:
        # 接收一条内部告警，先做去重和记录，再决定是否发送到飞书。
        normalized_level = level.upper()
        # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
        event = AlertEvent(
            # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
            source=source,
            # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
            level=normalized_level,
            # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
            summary=summary,
            # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
            details=details,
            # 将 dedupe_key 的值保存下来，供后续流程判断或组装响应时使用。
            dedupe_key=dedupe_key,
            # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
            tags=tags or [],
            # 将 created_at 的值保存下来，供后续流程判断或组装响应时使用。
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # 将 deduplicated 的值保存下来，供后续流程判断或组装响应时使用。
        deduplicated = self._should_deduplicate(event)
        # 即使被去重，也记录到 recent_events，便于接口侧看到最近发生过的重复告警。
        self._remember(event)
        # 根据 deduplicated 判断当前流程该进入哪个处理分支。
        if deduplicated:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Alert deduplicated for key=%s", dedupe_key)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False, True

        # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
        title = f"{self.title_prefix} {normalized_level}"
        # 将 message 的值保存下来，供后续流程判断或组装响应时使用。
        message = self._format_message(event)
        # 将 dispatched 的值保存下来，供后续流程判断或组装响应时使用。
        dispatched = self.notifier.send_text(title, message[:2000])
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return dispatched, False

    # 定义 report_exception 相关的处理逻辑，供流程或外部调用复用。
    def report_exception(self, scene: str, exc: Exception) -> tuple[bool, bool]:
        # 异常告警使用 scene + 异常类型 + 消息做去重键，避免同一错误短时间刷屏。
        # 把 Python 异常包装成标准告警事件，方便统一走去重和飞书通知逻辑。
        details = (
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"scene: {scene}\n"
            # 调用 type 完成当前步骤需要的业务处理。
            f"exception: {type(exc).__name__}\n"
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"message: {exc}"
        )
        # 将 dedupe_key 的值保存下来，供后续流程判断或组装响应时使用。
        dedupe_key = f"{scene}:{type(exc).__name__}:{exc}"
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return self.report(
            # 将 source 的值保存下来，供后续流程判断或组装响应时使用。
            source="agent-sentinel",
            # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
            level="ERROR",
            # 将 summary 的值保存下来，供后续流程判断或组装响应时使用。
            summary="Application exception detected",
            # 将 details 的值保存下来，供后续流程判断或组装响应时使用。
            details=details,
            # 将 dedupe_key 的值保存下来，供后续流程判断或组装响应时使用。
            dedupe_key=dedupe_key,
            # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
            tags=["exception", scene],
        )

    # 定义 recent_alerts 相关的处理逻辑，供流程或外部调用复用。
    def recent_alerts(self, limit: int = 20) -> list[dict[str, object]]:
        # 返回最近保存的告警记录，供接口查看告警历史和去重效果。
        records = list(self._recent_events)[-limit:]
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [asdict(record) for record in reversed(records)]

    # 定义 _should_deduplicate 相关的处理逻辑，供流程或外部调用复用。
    def _should_deduplicate(self, event: AlertEvent) -> bool:
        # 判断同一去重键是否在窗口期内重复出现，重复则不再发送飞书。
        if not event.dedupe_key:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False

        # 将 now 的值保存下来，供后续流程判断或组装响应时使用。
        now = time.time()
        # 进入上下文管理器保护的区域，自动处理资源生命周期。
        with self._lock:
            # 去重缓存和最近事件可能被多个请求线程同时访问，必须在锁内维护。
            self._prune_old_keys(now)
            # 将 last_seen 的值保存下来，供后续流程判断或组装响应时使用。
            last_seen = self._dedupe_cache.get(event.dedupe_key)
            # 根据 last_seen is not None and now - last_seen < self.ded... 判断当前流程该进入哪个处理分支。
            if last_seen is not None and now - last_seen < self.dedup_window_seconds:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return True
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            self._dedupe_cache[event.dedupe_key] = now
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False

    # 定义 _prune_old_keys 相关的处理逻辑，供流程或外部调用复用。
    def _prune_old_keys(self, now: float) -> None:
        # 清理过期去重键，避免去重缓存无限增长。
        expired = [
            # 执行当前业务步骤，推动流程继续向下游推进。
            key
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for key, last_seen in self._dedupe_cache.items()
            # 根据 now - last_seen >= self.dedup_window_seconds 判断当前流程该进入哪个处理分支。
            if now - last_seen >= self.dedup_window_seconds
        ]
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for key in expired:
            # 调用 _dedupe_cache.pop 完成当前步骤需要的业务处理。
            self._dedupe_cache.pop(key, None)

    # 定义 _remember 相关的处理逻辑，供流程或外部调用复用。
    def _remember(self, event: AlertEvent) -> None:
        # 把告警放入最近记录队列，后续查询接口可以看到它。
        with self._lock:
            # 把当前结果追加到集合中，逐步构建最终输出。
            self._recent_events.append(event)

    # 定义 _format_message 相关的处理逻辑，供流程或外部调用复用。
    def _format_message(self, event: AlertEvent) -> str:
        # 把告警字段整理成飞书消息正文，包含环境、来源、级别、摘要和详情。
        lines = [
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"env: {self.app_env}",
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"source: {event.source}",
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"level: {event.level}",
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"summary: {event.summary}",
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"time: {event.created_at}",
        ]
        # 根据 event.tags 判断当前流程该进入哪个处理分支。
        if event.tags:
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(f"tags: {', '.join(event.tags)}")
        # 根据 event.details 判断当前流程该进入哪个处理分支。
        if event.details:
            # 把当前结果追加到集合中，逐步构建最终输出。
            lines.append(f"details:\n{event.details}")
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "\n".join(lines)
