from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable

from agent_sentinel.config import Settings
from agent_sentinel.feishu_app import FeishuBotClient, extract_text_from_message_content

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 将 AnalyzeCallback 的值保存下来，供后续流程判断或组装响应时使用。
AnalyzeCallback = Callable[
    # 执行当前业务步骤，推动流程继续向下游推进。
    [str, str, str, str, str | None, str | None, str, list[str] | None, str | None, str | None, str | None],
    # 执行当前业务步骤，推动流程继续向下游推进。
    tuple[str, bool],
]


# 定义 FeishuMessagePoller 组件，集中管理这个模块的状态和行为。
class FeishuMessagePoller:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings: Settings,
        # 执行当前业务步骤，推动流程继续向下游推进。
        feishu_bot_client: FeishuBotClient,
        # 执行当前业务步骤，推动流程继续向下游推进。
        analyze_callback: AnalyzeCallback,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 保存轮询所需配置、飞书客户端和分析回调，并准备去重缓存。
        self.settings = settings
        # 将 self.feishu_bot_client 的值保存下来，供后续流程判断或组装响应时使用。
        self.feishu_bot_client = feishu_bot_client
        # 将 self.analyze_callback 的值保存下来，供后续流程判断或组装响应时使用。
        self.analyze_callback = analyze_callback
        # 将 self._thread 的值保存下来，供后续流程判断或组装响应时使用。
        self._thread: threading.Thread | None = None
        # 将 self._started 的值保存下来，供后续流程判断或组装响应时使用。
        self._started = False
        # 将 self._seen_message_ids 的值保存下来，供后续流程判断或组装响应时使用。
        self._seen_message_ids: set[str] = set()
        # 将 self._lock 的值保存下来，供后续流程判断或组装响应时使用。
        self._lock = threading.Lock()

    # 定义 start 相关的处理逻辑，供流程或外部调用复用。
    def start(self) -> None:
        # 检查轮询开关、会话列表和凭证后启动后台轮询线程。
        if self._started:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 not self.settings.feishu_message_polling_enabled 判断当前流程该进入哪个处理分支。
        if not self.settings.feishu_message_polling_enabled:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu message polling is disabled.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 not self.settings.feishu_allowed_chat_ids 判断当前流程该进入哪个处理分支。
        if not self.settings.feishu_allowed_chat_ids:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Feishu message polling skipped because FEISHU_ALLOWED_CHAT_IDS is empty.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 not self.feishu_bot_client.is_configured() 判断当前流程该进入哪个处理分支。
        if not self.feishu_bot_client.is_configured():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning("Feishu message polling skipped because app credentials are missing.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 将 self._thread 的值保存下来，供后续流程判断或组装响应时使用。
        self._thread = threading.Thread(
            # 将 target 的值保存下来，供后续流程判断或组装响应时使用。
            target=self._run_forever,
            # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
            name="feishu-message-poller",
            # 将 daemon 的值保存下来，供后续流程判断或组装响应时使用。
            daemon=True,
        )
        # 调用 _thread.start 完成当前步骤需要的业务处理。
        self._thread.start()
        # 将 self._started 的值保存下来，供后续流程判断或组装响应时使用。
        self._started = True

    # 定义 _run_forever 相关的处理逻辑，供流程或外部调用复用。
    def _run_forever(self) -> None:
        # 按固定间隔循环扫描允许的飞书会话，把新消息交给处理逻辑。
        interval = max(self.settings.feishu_message_polling_interval_seconds, 2)
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Starting Feishu message polling loop interval_seconds=%s chats=%s page_size=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            interval,
            # 调用 len 完成当前步骤需要的业务处理。
            len(self.settings.feishu_allowed_chat_ids),
            # 执行当前业务步骤，推动流程继续向下游推进。
            self.settings.feishu_message_polling_page_size,
        )
        # 在条件仍然成立时持续执行循环逻辑。
        while True:
            # 进入可能失败的处理块，便于后续统一捕获和恢复。
            try:
                # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
                for chat_id in self.settings.feishu_allowed_chat_ids:
                    # 调用 self._poll_chat 完成当前步骤需要的业务处理。
                    self._poll_chat(chat_id)
            # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
            except Exception:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.exception("Feishu message polling iteration failed")
            # 调用 time.sleep 完成当前步骤需要的业务处理。
            time.sleep(interval)

    # 定义 _poll_chat 相关的处理逻辑，供流程或外部调用复用。
    def _poll_chat(self, chat_id: str) -> None:
        # 拉取单个飞书会话的最近消息，并按时间顺序逐条检查是否需要诊断。
        started = time.perf_counter()
        # 将 items 的值保存下来，供后续流程判断或组装响应时使用。
        items = self.feishu_bot_client.list_chat_messages(
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 将 page_size 的值保存下来，供后续流程判断或组装响应时使用。
            page_size=max(self.settings.feishu_message_polling_page_size, 1),
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Polled Feishu messages chat_id=%s count=%s elapsed_ms=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 调用 len 完成当前步骤需要的业务处理。
            len(items),
            # 调用 int 完成当前步骤需要的业务处理。
            int((time.perf_counter() - started) * 1000),
        )
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for message in reversed(items):
            # 调用 self._handle_message 完成当前步骤需要的业务处理。
            self._handle_message(chat_id, message)

    # 定义 _handle_message 相关的处理逻辑，供流程或外部调用复用。
    def _handle_message(self, chat_id: str, message: dict[str, object]) -> None:
        # 过滤重复、非文本、非用户或未提及机器人的消息，只把有效文本送入分析。
        message_id = str(message.get("message_id") or "")
        # 根据 not message_id 判断当前流程该进入哪个处理分支。
        if not message_id:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu polled message ignored missing message_id chat_id=%s", chat_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 self._already_seen(message_id) 判断当前流程该进入哪个处理分支。
        if self._already_seen(message_id):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu polled message ignored duplicate message_id=%s chat_id=%s", message_id, chat_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 根据 str(message.get("msg_type") or "") != "text" 判断当前流程该进入哪个处理分支。
        if str(message.get("msg_type") or "") != "text":
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu polled message ignored msg_type=%s message_id=%s", message.get("msg_type"), message_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
        sender = message.get("sender") or {}
        # 根据 isinstance(sender, dict) 判断当前流程该进入哪个处理分支。
        if isinstance(sender, dict):
            # 将 sender_type 的值保存下来，供后续流程判断或组装响应时使用。
            sender_type = str(sender.get("sender_type") or "")
            # 根据 sender_type and sender_type != "user" 判断当前流程该进入哪个处理分支。
            if sender_type and sender_type != "user":
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Feishu polled message ignored sender_type=%s message_id=%s", sender_type, message_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return

        # 将 raw_content 的值保存下来，供后续流程判断或组装响应时使用。
        raw_content = message.get("body") or message.get("content") or ""
        # 将 content_text 的值保存下来，供后续流程判断或组装响应时使用。
        content_text = self._extract_text(raw_content)
        # 根据 not content_text.strip() 判断当前流程该进入哪个处理分支。
        if not content_text.strip():
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu polled message ignored empty content message_id=%s chat_id=%s", message_id, chat_id)
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 将 mentions 的值保存下来，供后续流程判断或组装响应时使用。
        mentions = message.get("mentions") or []
        # 根据 self.settings.feishu_analyze_mention_only and not se... 判断当前流程该进入哪个处理分支。
        if self.settings.feishu_analyze_mention_only and not self._looks_like_bot_mention(content_text, mentions):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Feishu polled message skipped because it does not mention bot message_id=%s content_chars=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                message_id,
                # 调用 len 完成当前步骤需要的业务处理。
                len(content_text),
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info(
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "Feishu polled message routed to analysis message_id=%s chat_id=%s root_message_id=%s content_chars=%s",
            # 执行当前业务步骤，推动流程继续向下游推进。
            message_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 调用 str 完成当前步骤需要的业务处理。
            str(message.get("root_id") or "") or message_id,
            # 调用 len 完成当前步骤需要的业务处理。
            len(content_text),
        )
        # 调用 self.analyze_callback 完成当前步骤需要的业务处理。
        self.analyze_callback(
            # 执行当前业务步骤，推动流程继续向下游推进。
            chat_id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "feishu-user",
            # 执行当前业务步骤，推动流程继续向下游推进。
            "INFO",
            # 执行当前业务步骤，推动流程继续向下游推进。
            f"{self.settings.feishu_bot_name} received a polled message",
            # 执行当前业务步骤，推动流程继续向下游推进。
            content_text,
            # 执行当前业务步骤，推动流程继续向下游推进。
            content_text,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "user_message_polling",
            # 执行当前业务步骤，推动流程继续向下游推进。
            ["feishu", "polling"],
            # 调用 str 完成当前步骤需要的业务处理。
            str(message.get("root_id") or "") or message_id,
            # 调用 self._extract_sender_open_id 完成当前步骤需要的业务处理。
            self._extract_sender_open_id(sender),
            # 调用 self._extract_sender_name 完成当前步骤需要的业务处理。
            self._extract_sender_name(sender),
        )

    # 定义 _extract_text 相关的处理逻辑，供流程或外部调用复用。
    def _extract_text(self, raw_content: object) -> str:
        # 兼容飞书返回的字符串和字典格式，把消息体统一转换成普通文本。
        if isinstance(raw_content, dict):
            # 根据 "content" in raw_content 判断当前流程该进入哪个处理分支。
            if "content" in raw_content:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return extract_text_from_message_content(str(raw_content.get("content") or ""))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return json.dumps(raw_content, ensure_ascii=False)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return extract_text_from_message_content(str(raw_content))

    # 定义 _looks_like_bot_mention 相关的处理逻辑，供流程或外部调用复用。
    def _looks_like_bot_mention(self, content_text: str, mentions: object) -> bool:
        # 判断一条群消息是否像是在呼叫机器人，避免机器人处理所有群聊闲谈。
        bot_name = self.settings.feishu_bot_name.strip()
        # 根据 isinstance(mentions, list) 判断当前流程该进入哪个处理分支。
        if isinstance(mentions, list):
            # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
            for mention in mentions:
                # 根据 not isinstance(mention, dict) 判断当前流程该进入哪个处理分支。
                if not isinstance(mention, dict):
                    # 跳过当前项剩余逻辑，继续处理下一项数据。
                    continue
                # 根据 str(mention.get("name") or "").strip() == bot_name 判断当前流程该进入哪个处理分支。
                if str(mention.get("name") or "").strip() == bot_name:
                    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                    return True
            # 根据 mentions 判断当前流程该进入哪个处理分支。
            if mentions:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return True
        # 将 stripped 的值保存下来，供后续流程判断或组装响应时使用。
        stripped = content_text.strip()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return (
            # 执行当前业务步骤，推动流程继续向下游推进。
            "<at " in content_text
            # 调用 stripped.startswith 完成当前步骤需要的业务处理。
            or stripped.startswith("@_user_")
            # 调用 stripped.startswith 完成当前步骤需要的业务处理。
            or stripped.startswith("@")
            # 调用 or 完成当前步骤需要的业务处理。
            or (bot_name and bot_name in content_text)
        )

    # 定义 _extract_sender_open_id 相关的处理逻辑，供流程或外部调用复用。
    def _extract_sender_open_id(self, sender: object) -> str | None:
        # 从轮询消息的发送人字段中取出 open_id，用于后续定向回复或提醒。
        if not isinstance(sender, dict):
            return None
        # 将 sender_id 的值保存下来，供后续流程判断或组装响应时使用。
        sender_id = str(sender.get("id") or "").strip()
        # 将 sender_id_type 的值保存下来，供后续流程判断或组装响应时使用。
        sender_id_type = str(sender.get("id_type") or "").strip()
        # 根据 sender_id and sender_id_type == "open_id" 判断当前流程该进入哪个处理分支。
        if sender_id and sender_id_type == "open_id":
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return sender_id
        return None

    # 定义 _extract_sender_name 相关的处理逻辑，供流程或外部调用复用。
    def _extract_sender_name(self, sender: object) -> str | None:
        # 从轮询消息的发送人字段中取出姓名，方便回复内容展示用户身份。
        if not isinstance(sender, dict):
            return None
        # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
        name = str(sender.get("name") or "").strip()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return name or None

    # 定义 _already_seen 相关的处理逻辑，供流程或外部调用复用。
    def _already_seen(self, message_id: str) -> bool:
        # 记录已经处理过的消息编号，防止轮询下一轮把同一条消息重复诊断。
        with self._lock:
            # 根据 message_id in self._seen_message_ids 判断当前流程该进入哪个处理分支。
            if message_id in self._seen_message_ids:
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return True
            # 调用 _seen_message_ids.add 完成当前步骤需要的业务处理。
            self._seen_message_ids.add(message_id)
            # 根据 len(self._seen_message_ids) > 1000 判断当前流程该进入哪个处理分支。
            if len(self._seen_message_ids) > 1000:
                # 将 self._seen_message_ids 的值保存下来，供后续流程判断或组装响应时使用。
                self._seen_message_ids = set(list(self._seen_message_ids)[-500:])
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return False
