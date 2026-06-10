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

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AlertEvent:
    source: str
    level: str
    summary: str
    details: str | None
    dedupe_key: str | None
    tags: list[str]
    created_at: str


class FeishuWebhookNotifier:
    def __init__(
        self,
        webhook_url: str | None,
        secret: str | None = None,
        enabled: bool = True,
        timeout: int = 10,
    ) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.webhook_url = webhook_url
        self.secret = secret
        self.enabled = enabled
        self.timeout = timeout

    def is_configured(self) -> bool:
        # 方法说明：校验输入或状态是否满足继续处理的条件。
        return self.enabled and bool(self.webhook_url)

    def _build_sign(self, timestamp: str) -> str:
        # 方法说明：构建并返回调用方需要的对象。
        if not self.secret:
            return ""
        # 飞书自定义机器人签名格式为 timestamp + "\n" + secret，再做 HMAC-SHA256。
        string_to_sign = f"{timestamp}\n{self.secret}".encode("utf-8")
        hmac_code = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
        return base64.b64encode(hmac_code).decode("utf-8")

    def send_text(self, title: str, message: str) -> bool:
        # 方法说明：将数据发送到外部通道，并隔离调用细节。
        if not self.is_configured():
            logger.info("Feishu alert skipped because webhook is not configured.")
            return False

        payload: dict[str, object] = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": [[{"tag": "text", "text": message}]],
                    }
                }
            },
        }

        if self.secret:
            # 带 secret 的 webhook 必须附带 timestamp 和 sign，防止消息被伪造。
            timestamp = str(int(time.time()))
            payload["timestamp"] = timestamp
            payload["sign"] = self._build_sign(timestamp)

        response = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        if result.get("code") not in (0, "0", None):
            raise RuntimeError(f"Feishu webhook returned error: {result}")
        return True


class RealtimeAlertService:
    def __init__(
        self,
        notifier: FeishuWebhookNotifier,
        app_env: str,
        title_prefix: str,
        dedup_window_seconds: int = 60,
        store_limit: int = 100,
    ) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.notifier = notifier
        self.app_env = app_env
        self.title_prefix = title_prefix
        self.dedup_window_seconds = dedup_window_seconds
        self._recent_events: deque[AlertEvent] = deque(maxlen=store_limit)
        self._dedupe_cache: dict[str, float] = {}
        self._lock = threading.Lock()

    def report(
        self,
        source: str,
        level: str,
        summary: str,
        details: str | None = None,
        dedupe_key: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[bool, bool]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        normalized_level = level.upper()
        event = AlertEvent(
            source=source,
            level=normalized_level,
            summary=summary,
            details=details,
            dedupe_key=dedupe_key,
            tags=tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        deduplicated = self._should_deduplicate(event)
        # 即使被去重，也记录到 recent_events，便于接口侧看到最近发生过的重复告警。
        self._remember(event)
        if deduplicated:
            logger.info("Alert deduplicated for key=%s", dedupe_key)
            return False, True

        title = f"{self.title_prefix} {normalized_level}"
        message = self._format_message(event)
        dispatched = self.notifier.send_text(title, message[:2000])
        return dispatched, False

    def report_exception(self, scene: str, exc: Exception) -> tuple[bool, bool]:
        # 异常告警使用 scene + 异常类型 + 消息做去重键，避免同一错误短时间刷屏。
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        details = (
            f"scene: {scene}\n"
            f"exception: {type(exc).__name__}\n"
            f"message: {exc}"
        )
        dedupe_key = f"{scene}:{type(exc).__name__}:{exc}"
        return self.report(
            source="agent-sentinel",
            level="ERROR",
            summary="Application exception detected",
            details=details,
            dedupe_key=dedupe_key,
            tags=["exception", scene],
        )

    def recent_alerts(self, limit: int = 20) -> list[dict[str, object]]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        records = list(self._recent_events)[-limit:]
        return [asdict(record) for record in reversed(records)]

    def _should_deduplicate(self, event: AlertEvent) -> bool:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if not event.dedupe_key:
            return False

        now = time.time()
        with self._lock:
            # 去重缓存和最近事件可能被多个请求线程同时访问，必须在锁内维护。
            self._prune_old_keys(now)
            last_seen = self._dedupe_cache.get(event.dedupe_key)
            if last_seen is not None and now - last_seen < self.dedup_window_seconds:
                return True
            self._dedupe_cache[event.dedupe_key] = now
            return False

    def _prune_old_keys(self, now: float) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        expired = [
            key
            for key, last_seen in self._dedupe_cache.items()
            if now - last_seen >= self.dedup_window_seconds
        ]
        for key in expired:
            self._dedupe_cache.pop(key, None)

    def _remember(self, event: AlertEvent) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        with self._lock:
            self._recent_events.append(event)

    def _format_message(self, event: AlertEvent) -> str:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        lines = [
            f"env: {self.app_env}",
            f"source: {event.source}",
            f"level: {event.level}",
            f"summary: {event.summary}",
            f"time: {event.created_at}",
        ]
        if event.tags:
            lines.append(f"tags: {', '.join(event.tags)}")
        if event.details:
            lines.append(f"details:\n{event.details}")
        return "\n".join(lines)
