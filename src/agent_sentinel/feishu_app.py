from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import aiohttp
import requests

from agent_sentinel.config import Settings
from agent_sentinel.monitoring import monitor

logger = logging.getLogger(__name__)


class FeishuBotClient:
    def __init__(self, settings: Settings, timeout: int = 15) -> None:
        # 保存飞书应用配置，供后续鉴权和接口请求使用。
        self.settings = settings
        # 记录请求超时时间，避免飞书接口调用长时间阻塞。
        self.timeout = timeout
        # 缓存租户访问令牌，减少重复鉴权请求。
        self._token: str | None = None
        # 记录令牌过期时间，用于判断是否需要刷新。
        self._token_expire_at = 0.0
        # 使用锁保护令牌刷新流程，避免并发请求重复刷新。
        self._lock = threading.Lock()

    def is_configured(self) -> bool:
        # 检查飞书应用 ID 和密钥是否齐全，决定飞书能力能不能真正启用。
        return bool(self.settings.feishu_app_id and self.settings.feishu_app_secret)

    def send_text_to_chat(
        self,
        chat_id: str,
        text: str,
        *,
        thread_root_message_id: str | None = None,
        mention_open_id: str | None = None,
        mention_name: str | None = None,
    ) -> bool:
        # 向飞书发送文本消息；有话题根消息时回复到原话题，否则发普通群消息。
        self._ensure_configured()
        token = self._get_tenant_access_token()
        base_url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages"

        final_text = text
        if mention_open_id:
            # 飞书文本消息需要用 at 标签显式 mention 用户，普通 @ 文本不会触发客户端高亮。
            display_name = mention_name or "user"
            final_text = f'<at user_id="{mention_open_id}">@{display_name}</at> {text}'

        if thread_root_message_id:
            # 有 root message 时回复到原话题，避免诊断进度在群里刷成多条独立消息。
            url = f"{base_url}/{thread_root_message_id}/reply"
            payload = {
                "msg_type": "text",
                "content": json.dumps({"text": final_text}, ensure_ascii=False),
                "reply_in_thread": True,
            }
        else:
            # 无话题上下文时退回普通群消息发送。
            url = f"{base_url}?receive_id_type=chat_id"
            payload = {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": final_text}, ensure_ascii=False),
            }

        started = time.perf_counter()
        logger.info(
            "Feishu send text request chat_id=%s thread_root_message_id=%s text_chars=%s mention=%s",
            chat_id,
            thread_root_message_id,
            len(final_text),
            bool(mention_open_id),
        )
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu send message failed: {result}")
        logger.info(
            "Feishu send text completed chat_id=%s thread_root_message_id=%s elapsed_ms=%s",
            chat_id,
            thread_root_message_id,
            int((time.perf_counter() - started) * 1000),
        )
        return True

    def send_interactive_card_to_chat(
        self,
        chat_id: str,
        card: dict[str, object],
        *,
        thread_root_message_id: str | None = None,
    ) -> bool:
        # 向飞书发送交互卡片，用于承载诊断确认、案例选择等需要点击的动作。
        self._ensure_configured()
        token = self._get_tenant_access_token()
        base_url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages"
        if thread_root_message_id:
            url = f"{base_url}/{thread_root_message_id}/reply"
            payload = {
                "msg_type": "interactive",
                "content": json.dumps(card, ensure_ascii=False),
                "reply_in_thread": True,
            }
        else:
            url = f"{base_url}?receive_id_type=chat_id"
            payload = {
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": json.dumps(card, ensure_ascii=False),
            }

        started = time.perf_counter()
        logger.info("Feishu send card request chat_id=%s thread_root_message_id=%s", chat_id, thread_root_message_id)
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu send card failed: {result}")
        logger.info(
            "Feishu send card completed chat_id=%s thread_root_message_id=%s elapsed_ms=%s",
            chat_id,
            thread_root_message_id,
            int((time.perf_counter() - started) * 1000),
        )
        return True

    def send_topic_text(self, chat_id: str, root_message_id: str, text: str) -> str | None:
        """Reply in the source message thread and return the new Feishu message_id."""
        # 在原始消息话题下回复文本，并返回新消息 ID，方便后续继续更新同一话题。
        self._ensure_configured()
        token = self._get_tenant_access_token()
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{root_message_id}/reply"
        payload = {
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
            "reply_in_thread": True,
        }
        started = time.perf_counter()
        logger.info("Feishu send topic text request chat_id=%s root_message_id=%s text_chars=%s", chat_id, root_message_id, len(text))
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu send topic text failed: {result}")
        data = result.get("data") or {}
        message_id = data.get("message_id") if isinstance(data, dict) else None
        logger.info(
            "Feishu send topic text completed chat_id=%s root_message_id=%s message_id=%s elapsed_ms=%s",
            chat_id,
            root_message_id,
            message_id,
            int((time.perf_counter() - started) * 1000),
        )
        return message_id

    def send_topic_card(self, chat_id: str, root_message_id: str, card: dict[str, object]) -> str | None:
        """Send an interactive card in the source message thread and return message_id."""
        # 在原始消息话题下发送交互卡片，并返回卡片消息 ID 供后续 PATCH 更新。
        self._ensure_configured()
        token = self._get_tenant_access_token()
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{root_message_id}/reply"
        payload = {
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
            "reply_in_thread": True,
        }
        started = time.perf_counter()
        logger.info("Feishu send topic card request chat_id=%s root_message_id=%s", chat_id, root_message_id)
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu send topic card failed: {result}")
        data = result.get("data") or {}
        message_id = data.get("message_id") if isinstance(data, dict) else None
        logger.info(
            "Feishu send topic card completed chat_id=%s root_message_id=%s message_id=%s elapsed_ms=%s",
            chat_id,
            root_message_id,
            message_id,
            int((time.perf_counter() - started) * 1000),
        )
        return message_id

    def update_message_card(self, message_id: str, card: dict[str, object]) -> bool:
        """Best-effort update for a bot-sent interactive card."""
        # 同步更新已经发送过的飞书卡片，让同一张卡片展示最新节点进度。
        self._ensure_configured()
        token = self._get_tenant_access_token()
        # PATCH 原卡片消息是单卡片交互的关键，避免每个节点都发送一张新卡片。
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{message_id}"
        payload = {"content": json.dumps(card, ensure_ascii=False)}
        started = time.perf_counter()
        logger.info("Feishu update message card request message_id=%s", message_id)
        response = requests.patch(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu update message card failed: {result}")
        logger.info("Feishu update message card completed message_id=%s elapsed_ms=%s", message_id, int((time.perf_counter() - started) * 1000))
        return True

    async def update_message_card_async(self, message_id: str, card: dict[str, object]) -> bool:
        """Async PATCH update for a bot-sent interactive card."""
        # 异步更新已经发送过的飞书卡片，避免等待网络请求时阻塞工作流。
        self._ensure_configured()
        token = await asyncio.to_thread(self._get_tenant_access_token)
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{message_id}"
        payload = {"content": json.dumps(card, ensure_ascii=False)}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        started = time.perf_counter()
        logger.info("Feishu async update message card request message_id=%s", message_id)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.patch(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                result = await response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu async update message card failed: {result}")
        logger.info("Feishu async update message card completed message_id=%s elapsed_ms=%s", message_id, int((time.perf_counter() - started) * 1000))
        return True

    def list_chat_messages(
        self,
        chat_id: str,
        *,
        page_size: int = 20,
        sort_type: str = "ByCreateTimeDesc",
    ) -> list[dict[str, object]]:
        # 拉取某个飞书会话的最近消息，供轮询模式发现用户新输入。
        self._ensure_configured()
        token = self._get_tenant_access_token()
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages"
        started = time.perf_counter()
        logger.info("Feishu list messages request chat_id=%s page_size=%s sort_type=%s", chat_id, page_size, sort_type)
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": page_size,
                "sort_type": sort_type,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") not in (0, "0", None):
            monitor.record_error("feishu_api_error")
            raise RuntimeError(f"Feishu list messages failed: {result}")
        data = result.get("data") or {}
        items = data.get("items") or []
        logger.info("Feishu list messages completed chat_id=%s count=%s elapsed_ms=%s", chat_id, len(items), int((time.perf_counter() - started) * 1000))
        return [item for item in items if isinstance(item, dict)]

    def _ensure_configured(self) -> None:
        # 在真正调用飞书接口前做配置保护，缺少凭证时直接抛出清晰错误。
        if not self.is_configured():
            raise RuntimeError("FEISHU_APP_ID or FEISHU_APP_SECRET is not configured.")

    def _get_tenant_access_token(self) -> str:
        # 获取并缓存飞书租户访问令牌，过期前自动刷新，减少每次发消息的鉴权成本。
        now = time.time()
        with self._lock:
            if self._token and now < self._token_expire_at:
                logger.debug("Feishu tenant access token cache hit expires_in_seconds=%s", int(self._token_expire_at - now))
                return self._token

            # tenant_access_token 全局复用，刷新时加锁避免并发请求同时打到飞书鉴权接口。
            url = (
                f"{self.settings.feishu_api_base_url.rstrip('/')}"
                "/open-apis/auth/v3/tenant_access_token/internal"
            )
            started = time.perf_counter()
            logger.info("Feishu tenant access token refresh started")
            response = requests.post(
                url,
                json={
                    "app_id": self.settings.feishu_app_id,
                    "app_secret": self.settings.feishu_app_secret,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("code") not in (0, "0", None):
                monitor.record_error("feishu_api_error")
                raise RuntimeError(f"Feishu tenant_access_token fetch failed: {result}")

            token = result.get("tenant_access_token")
            if not token:
                monitor.record_error("feishu_api_error")
                raise RuntimeError("Feishu tenant_access_token is missing in response.")

            expire = int(result.get("expire", 7200))
            self._token = token
            # 提前 120 秒过期，降低临界点使用已失效 token 的概率。
            self._token_expire_at = now + max(expire - 120, 60)
            logger.info("Feishu tenant access token refresh completed expire_seconds=%s elapsed_ms=%s", expire, int((time.perf_counter() - started) * 1000))
            return token


def extract_text_from_message_content(content: str | None) -> str:
    # 从飞书消息 content 字段里提取可读文本，解析失败时保留原始内容兜底。
    if not content:
        return ""
    try:
        # 飞书 text 消息 content 通常是 JSON 字符串；解析失败时按原始文本兜底。
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
    return content
