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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 定义 FeishuBotClient 组件，集中管理这个模块的状态和行为。
class FeishuBotClient:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(self, settings: Settings, timeout: int = 15) -> None:
        # 保存飞书应用配置，并准备 token 缓存和锁，减少重复鉴权请求。
        self.settings = settings
        # 将 self.timeout 的值保存下来，供后续流程判断或组装响应时使用。
        self.timeout = timeout
        # 将 self._token 的值保存下来，供后续流程判断或组装响应时使用。
        self._token: str | None = None
        # 将 self._token_expire_at 的值保存下来，供后续流程判断或组装响应时使用。
        self._token_expire_at = 0.0
        # 将 self._lock 的值保存下来，供后续流程判断或组装响应时使用。
        self._lock = threading.Lock()

    # 定义 is_configured 相关的处理逻辑，供流程或外部调用复用。
    def is_configured(self) -> bool:
        # 检查飞书应用 ID 和密钥是否齐全，决定飞书能力能不能真正启用。
        return bool(self.settings.feishu_app_id and self.settings.feishu_app_secret)

    # 定义 send_text_to_chat 相关的处理逻辑，供流程或外部调用复用。
    def send_text_to_chat(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        text: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
        # 将 mention_open_id 的值保存下来，供后续流程判断或组装响应时使用。
        mention_open_id: str | None = None,
        # 将 mention_name 的值保存下来，供后续流程判断或组装响应时使用。
        mention_name: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 向飞书发送文本消息；有话题根消息时回复到原话题，否则发普通群消息。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self._get_tenant_access_token()
        # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
        base_url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages"

        # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
        final_text = text
        # 根据 mention_open_id 判断当前流程该进入哪个处理分支。
        if mention_open_id:
            # 飞书文本消息需要用 at 标签显式 mention 用户，普通 @ 文本不会触发客户端高亮。
            display_name = mention_name or "user"
            # 将 final_text 的值保存下来，供后续流程判断或组装响应时使用。
            final_text = f'<at user_id="{mention_open_id}">@{display_name}</at> {text}'

        # 根据 thread_root_message_id 判断当前流程该进入哪个处理分支。
        if thread_root_message_id:
            # 有 root message 时回复到原话题，避免诊断进度在群里刷成多条独立消息。
            url = f"{base_url}/{thread_root_message_id}/reply"
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "msg_type": "text",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "content": json.dumps({"text": final_text}, ensure_ascii=False),
                # 执行当前业务步骤，推动流程继续向下游推进。
                "reply_in_thread": True,
            }
        # 处理前面条件都不满足时的默认分支。
        else:
            # 无话题上下文时退回普通群消息发送。
            url = f"{base_url}?receive_id_type=chat_id"
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "receive_id": chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "msg_type": "text",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "content": json.dumps({"text": final_text}, ensure_ascii=False),
            }

        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Feishu send text request chat_id=%s thread_root_message_id=%s text_chars=%s mention=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            thread_root_message_id,
            # 调用 len 完成当前步骤需要的业务处理。
            len(final_text),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(mention_open_id),
        )
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            url,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Authorization": f"Bearer {token}",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Content-Type": "application/json; charset=utf-8",
            },
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json=payload,
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu send message failed: {result}")
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Feishu send text completed chat_id=%s thread_root_message_id=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            thread_root_message_id,
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return True

    # 定义 send_interactive_card_to_chat 相关的处理逻辑，供流程或外部调用复用。
    def send_interactive_card_to_chat(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        card: dict[str, object],
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 thread_root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
        thread_root_message_id: str | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> bool:
        # 向飞书发送交互卡片，用于承载诊断确认、案例选择等需要点击的动作。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self._get_tenant_access_token()
        # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
        base_url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages"
        # 根据 thread_root_message_id 判断当前流程该进入哪个处理分支。
        if thread_root_message_id:
            # 将 url 的值保存下来，供后续流程判断或组装响应时使用。
            url = f"{base_url}/{thread_root_message_id}/reply"
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "msg_type": "interactive",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "content": json.dumps(card, ensure_ascii=False),
                # 执行当前业务步骤，推动流程继续向下游推进。
                "reply_in_thread": True,
            }
        # 处理前面条件都不满足时的默认分支。
        else:
            # 将 url 的值保存下来，供后续流程判断或组装响应时使用。
            url = f"{base_url}?receive_id_type=chat_id"
            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "receive_id": chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "msg_type": "interactive",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "content": json.dumps(card, ensure_ascii=False),
            }

        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu send card request chat_id=%s thread_root_message_id=%s", chat_id, thread_root_message_id)
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            url,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Authorization": f"Bearer {token}",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Content-Type": "application/json; charset=utf-8",
            },
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json=payload,
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu send card failed: {result}")
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Feishu send card completed chat_id=%s thread_root_message_id=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            thread_root_message_id,
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return True

    # 定义 send_topic_text 相关的处理逻辑，供流程或外部调用复用。
    def send_topic_text(self, chat_id: str, root_message_id: str, text: str) -> str | None:
        """Reply in the source message thread and return the new Feishu message_id."""
        # 在原始消息话题下回复文本，并返回新消息 ID，方便后续继续更新同一话题。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self._get_tenant_access_token()
        # 将 url 的值保存下来，供后续流程判断或组装响应时使用。
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{root_message_id}/reply"
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "msg_type": "text",
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "content": json.dumps({"text": text}, ensure_ascii=False),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "reply_in_thread": True,
        }
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu send topic text request chat_id=%s root_message_id=%s text_chars=%s", chat_id, root_message_id, len(text))
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            url,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Authorization": f"Bearer {token}",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Content-Type": "application/json; charset=utf-8",
            },
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json=payload,
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu send topic text failed: {result}")
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = result.get("data") or {}
        # 将 message_id 的值保存下来，供后续流程判断或组装响应时使用。
        message_id = data.get("message_id") if isinstance(data, dict) else None
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Feishu send topic text completed chat_id=%s root_message_id=%s message_id=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            root_message_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            message_id,
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return message_id

    # 定义 send_topic_card 相关的处理逻辑，供流程或外部调用复用。
    def send_topic_card(self, chat_id: str, root_message_id: str, card: dict[str, object]) -> str | None:
        """Send an interactive card in the source message thread and return message_id."""
        # 在原始消息话题下发送交互卡片，并返回卡片消息 ID 供后续 PATCH 更新。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self._get_tenant_access_token()
        # 将 url 的值保存下来，供后续流程判断或组装响应时使用。
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{root_message_id}/reply"
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "msg_type": "interactive",
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "content": json.dumps(card, ensure_ascii=False),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "reply_in_thread": True,
        }
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu send topic card request chat_id=%s root_message_id=%s", chat_id, root_message_id)
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.post(
            # 执行当前业务步骤，推动流程继续向下游推进。
            url,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Authorization": f"Bearer {token}",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Content-Type": "application/json; charset=utf-8",
            },
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json=payload,
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu send topic card failed: {result}")
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = result.get("data") or {}
        # 将 message_id 的值保存下来，供后续流程判断或组装响应时使用。
        message_id = data.get("message_id") if isinstance(data, dict) else None
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Feishu send topic card completed chat_id=%s root_message_id=%s message_id=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            root_message_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            message_id,
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return message_id

    # 定义 update_message_card 相关的处理逻辑，供流程或外部调用复用。
    def update_message_card(self, message_id: str, card: dict[str, object]) -> bool:
        """Best-effort update for a bot-sent interactive card."""
        # 同步更新已经发送过的飞书卡片，让同一张卡片展示最新节点进度。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self._get_tenant_access_token()
        # PATCH 原卡片消息是单卡片交互的关键，避免每个节点都发送一张新卡片。
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{message_id}"
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = {"content": json.dumps(card, ensure_ascii=False)}
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu update message card request message_id=%s", message_id)
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.patch(
            # 执行当前业务步骤，推动流程继续向下游推进。
            url,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Authorization": f"Bearer {token}",
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Content-Type": "application/json; charset=utf-8",
            },
            # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
            json=payload,
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu update message card failed: {result}")
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu update message card completed message_id=%s elapsed_ms=%s", message_id, int((time.perf_counter() - started) * 1000))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return True

    # 定义 update_message_card_async 相关的处理逻辑，供流程或外部调用复用。
    async def update_message_card_async(self, message_id: str, card: dict[str, object]) -> bool:
        """Async PATCH update for a bot-sent interactive card."""
        # 异步更新已经发送过的飞书卡片，避免等待网络请求时阻塞工作流。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = await asyncio.to_thread(self._get_tenant_access_token)
        # 将 url 的值保存下来，供后续流程判断或组装响应时使用。
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages/{message_id}"
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = {"content": json.dumps(card, ensure_ascii=False)}
        # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu async update message card request message_id=%s", message_id)
        # 进入异步上下文管理区域，确保异步资源按约定释放。
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 进入异步上下文管理区域，确保异步资源按约定释放。
            async with session.patch(
                # 执行当前业务步骤，推动流程继续向下游推进。
                url,
                # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
                headers={
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "Authorization": f"Bearer {token}",
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Content-Type": "application/json; charset=utf-8",
                },
                # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
                json=payload,
            # 执行当前业务步骤，推动流程继续向下游推进。
            ) as response:
                # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
                response.raise_for_status()
                # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
                result = await response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu async update message card failed: {result}")
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu async update message card completed message_id=%s elapsed_ms=%s", message_id, int((time.perf_counter() - started) * 1000))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return True

    # 定义 list_chat_messages 相关的处理逻辑，供流程或外部调用复用。
    def list_chat_messages(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        chat_id: str,
        # 执行当前业务步骤，推动流程继续向下游推进。
        *,
        # 将 page_size 的值保存下来，供后续流程判断或组装响应时使用。
        page_size: int = 20,
        # 将 sort_type 的值保存下来，供后续流程判断或组装响应时使用。
        sort_type: str = "ByCreateTimeDesc",
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> list[dict[str, object]]:
        # 拉取某个飞书会话的最近消息，供轮询模式发现用户新输入。
        self._ensure_configured()
        # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
        token = self._get_tenant_access_token()
        # 将 url 的值保存下来，供后续流程判断或组装响应时使用。
        url = f"{self.settings.feishu_api_base_url.rstrip('/')}/open-apis/im/v1/messages"
        # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
        started = time.perf_counter()
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu list messages request chat_id=%s page_size=%s sort_type=%s", chat_id, page_size, sort_type)
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = requests.get(
            # 执行当前业务步骤，推动流程继续向下游推进。
            url,
            # 将 headers 的值保存下来，供后续流程判断或组装响应时使用。
            headers={"Authorization": f"Bearer {token}"},
            # 将 params 的值保存下来，供后续流程判断或组装响应时使用。
            params={
                # 执行当前业务步骤，推动流程继续向下游推进。
                "container_id_type": "chat",
                # 执行当前业务步骤，推动流程继续向下游推进。
                "container_id": chat_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "page_size": page_size,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "sort_type": sort_type,
            },
            # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
            timeout=self.timeout,
        )
        # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
        response.raise_for_status()
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = response.json()
        # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
        if result.get("code") not in (0, "0", None):
            # 调用 monitor.record_error 完成当前步骤需要的业务处理。
            monitor.record_error("feishu_api_error")
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError(f"Feishu list messages failed: {result}")
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = result.get("data") or {}
        # 将 items 的值保存下来，供后续流程判断或组装响应时使用。
        items = data.get("items") or []
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu list messages completed chat_id=%s count=%s elapsed_ms=%s", chat_id, len(items), int((time.perf_counter() - started) * 1000))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [item for item in items if isinstance(item, dict)]

    # 定义 _ensure_configured 相关的处理逻辑，供流程或外部调用复用。
    def _ensure_configured(self) -> None:
        # 在真正调用飞书接口前做配置保护，缺少凭证时直接抛出清晰错误。
        if not self.is_configured():
            # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
            raise RuntimeError("FEISHU_APP_ID or FEISHU_APP_SECRET is not configured.")

    # 定义 _get_tenant_access_token 相关的处理逻辑，供流程或外部调用复用。
    def _get_tenant_access_token(self) -> str:
        # 获取并缓存飞书租户访问令牌，过期前自动刷新，减少每次发消息的鉴权成本。
        now = time.time()
        # 进入上下文管理器保护的区域，自动处理资源生命周期。
        with self._lock:
            # 根据 self._token and now < self._token_expire_at 判断当前流程该进入哪个处理分支。
            if self._token and now < self._token_expire_at:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.debug("Feishu tenant access token cache hit expires_in_seconds=%s", int(self._token_expire_at - now))
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return self._token

            # tenant_access_token 全局复用，刷新时加锁避免并发请求同时打到飞书鉴权接口。
            url = (
                # 调用 feishu_api_base_url.rstrip 完成当前步骤需要的业务处理。
                f"{self.settings.feishu_api_base_url.rstrip('/')}"
                # 执行当前业务步骤，推动流程继续向下游推进。
                "/open-apis/auth/v3/tenant_access_token/internal"
            )
            # 将 started 的值保存下来，供后续流程判断或组装响应时使用。
            started = time.perf_counter()
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu tenant access token refresh started")
            # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
            response = requests.post(
                # 执行当前业务步骤，推动流程继续向下游推进。
                url,
                # 将 json 的值保存下来，供后续流程判断或组装响应时使用。
                json={
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "app_id": self.settings.feishu_app_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    "app_secret": self.settings.feishu_app_secret,
                },
                # 将 timeout 的值保存下来，供后续流程判断或组装响应时使用。
                timeout=self.timeout,
            )
            # 调用 response.raise_for_status 完成当前步骤需要的业务处理。
            response.raise_for_status()
            # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
            result = response.json()
            # 根据 result.get("code") not in (0, "0", None) 判断当前流程该进入哪个处理分支。
            if result.get("code") not in (0, "0", None):
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("feishu_api_error")
                # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
                raise RuntimeError(f"Feishu tenant_access_token fetch failed: {result}")

            # 将 token 的值保存下来，供后续流程判断或组装响应时使用。
            token = result.get("tenant_access_token")
            # 根据 not token 判断当前流程该进入哪个处理分支。
            if not token:
                # 调用 monitor.record_error 完成当前步骤需要的业务处理。
                monitor.record_error("feishu_api_error")
                # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
                raise RuntimeError("Feishu tenant_access_token is missing in response.")

            # 将 expire 的值保存下来，供后续流程判断或组装响应时使用。
            expire = int(result.get("expire", 7200))
            # 将 self._token 的值保存下来，供后续流程判断或组装响应时使用。
            self._token = token
            # 提前 120 秒过期，降低临界点使用已失效 token 的概率。
            self._token_expire_at = now + max(expire - 120, 60)
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu tenant access token refresh completed expire_seconds=%s elapsed_ms=%s", expire, int((time.perf_counter() - started) * 1000))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return token


# 定义 extract_text_from_message_content 相关的处理逻辑，供流程或外部调用复用。
def extract_text_from_message_content(content: str | None) -> str:
    # 从飞书消息 content 字段里提取可读文本，解析失败时保留原始内容兜底。
    if not content:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return ""
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 飞书 text 消息 content 通常是 JSON 字符串；解析失败时按原始文本兜底。
        payload = json.loads(content)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except json.JSONDecodeError:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return content
    # 根据 isinstance(payload, dict) 判断当前流程该进入哪个处理分支。
    if isinstance(payload, dict):
        # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
        text = payload.get("text")
        # 根据 isinstance(text, str) 判断当前流程该进入哪个处理分支。
        if isinstance(text, str):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return text
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return content
