from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable

from agent_sentinel.config import Settings
from agent_sentinel.feishu_app import FeishuBotClient, extract_text_from_message_content

logger = logging.getLogger(__name__)


AnalyzeCallback = Callable[
    [str, str, str, str, str | None, str | None, str, list[str] | None, str | None, str | None, str | None],
    tuple[str, bool],
]


class FeishuMessagePoller:
    def __init__(
        self,
        settings: Settings,
        feishu_bot_client: FeishuBotClient,
        analyze_callback: AnalyzeCallback,
    ) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        self.settings = settings
        self.feishu_bot_client = feishu_bot_client
        self.analyze_callback = analyze_callback
        self._thread: threading.Thread | None = None
        self._started = False
        self._seen_message_ids: set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        # 方法说明：初始化对象，并保存后续调用需要的状态。
        if self._started:
            return
        if not self.settings.feishu_message_polling_enabled:
            logger.info("Feishu message polling is disabled.")
            return
        if not self.settings.feishu_allowed_chat_ids:
            logger.warning("Feishu message polling skipped because FEISHU_ALLOWED_CHAT_IDS is empty.")
            return
        if not self.feishu_bot_client.is_configured():
            logger.warning("Feishu message polling skipped because app credentials are missing.")
            return

        self._thread = threading.Thread(
            target=self._run_forever,
            name="feishu-message-poller",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def _run_forever(self) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        interval = max(self.settings.feishu_message_polling_interval_seconds, 2)
        logger.info(
            "Starting Feishu message polling loop interval_seconds=%s chats=%s page_size=%s",
            interval,
            len(self.settings.feishu_allowed_chat_ids),
            self.settings.feishu_message_polling_page_size,
        )
        while True:
            try:
                for chat_id in self.settings.feishu_allowed_chat_ids:
                    self._poll_chat(chat_id)
            except Exception:
                logger.exception("Feishu message polling iteration failed")
            time.sleep(interval)

    def _poll_chat(self, chat_id: str) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        started = time.perf_counter()
        items = self.feishu_bot_client.list_chat_messages(
            chat_id,
            page_size=max(self.settings.feishu_message_polling_page_size, 1),
        )
        logger.info(
            "Polled Feishu messages chat_id=%s count=%s elapsed_ms=%s",
            chat_id,
            len(items),
            int((time.perf_counter() - started) * 1000),
        )
        for message in reversed(items):
            self._handle_message(chat_id, message)

    def _handle_message(self, chat_id: str, message: dict[str, object]) -> None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        message_id = str(message.get("message_id") or "")
        if not message_id:
            logger.info("Feishu polled message ignored missing message_id chat_id=%s", chat_id)
            return
        if self._already_seen(message_id):
            logger.info("Feishu polled message ignored duplicate message_id=%s chat_id=%s", message_id, chat_id)
            return

        if str(message.get("msg_type") or "") != "text":
            logger.info("Feishu polled message ignored msg_type=%s message_id=%s", message.get("msg_type"), message_id)
            return

        sender = message.get("sender") or {}
        if isinstance(sender, dict):
            sender_type = str(sender.get("sender_type") or "")
            if sender_type and sender_type != "user":
                logger.info("Feishu polled message ignored sender_type=%s message_id=%s", sender_type, message_id)
                return

        raw_content = message.get("body") or message.get("content") or ""
        content_text = self._extract_text(raw_content)
        if not content_text.strip():
            logger.info("Feishu polled message ignored empty content message_id=%s chat_id=%s", message_id, chat_id)
            return

        mentions = message.get("mentions") or []
        if self.settings.feishu_analyze_mention_only and not self._looks_like_bot_mention(content_text, mentions):
            logger.info(
                "Feishu polled message skipped because it does not mention bot message_id=%s content_chars=%s",
                message_id,
                len(content_text),
            )
            return

        logger.info(
            "Feishu polled message routed to analysis message_id=%s chat_id=%s root_message_id=%s content_chars=%s",
            message_id,
            chat_id,
            str(message.get("root_id") or "") or message_id,
            len(content_text),
        )
        self.analyze_callback(
            chat_id,
            "feishu-user",
            "INFO",
            f"{self.settings.feishu_bot_name} received a polled message",
            content_text,
            content_text,
            "user_message_polling",
            ["feishu", "polling"],
            str(message.get("root_id") or "") or message_id,
            self._extract_sender_open_id(sender),
            self._extract_sender_name(sender),
        )

    def _extract_text(self, raw_content: object) -> str:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if isinstance(raw_content, dict):
            if "content" in raw_content:
                return extract_text_from_message_content(str(raw_content.get("content") or ""))
            return json.dumps(raw_content, ensure_ascii=False)
        return extract_text_from_message_content(str(raw_content))

    def _looks_like_bot_mention(self, content_text: str, mentions: object) -> bool:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        bot_name = self.settings.feishu_bot_name.strip()
        if isinstance(mentions, list):
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                if str(mention.get("name") or "").strip() == bot_name:
                    return True
            if mentions:
                return True
        stripped = content_text.strip()
        return (
            "<at " in content_text
            or stripped.startswith("@_user_")
            or stripped.startswith("@")
            or (bot_name and bot_name in content_text)
        )

    def _extract_sender_open_id(self, sender: object) -> str | None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if not isinstance(sender, dict):
            return None
        sender_id = str(sender.get("id") or "").strip()
        sender_id_type = str(sender.get("id_type") or "").strip()
        if sender_id and sender_id_type == "open_id":
            return sender_id
        return None

    def _extract_sender_name(self, sender: object) -> str | None:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if not isinstance(sender, dict):
            return None
        name = str(sender.get("name") or "").strip()
        return name or None

    def _already_seen(self, message_id: str) -> bool:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        with self._lock:
            if message_id in self._seen_message_ids:
                return True
            self._seen_message_ids.add(message_id)
            if len(self._seen_message_ids) > 1000:
                self._seen_message_ids = set(list(self._seen_message_ids)[-500:])
            return False
