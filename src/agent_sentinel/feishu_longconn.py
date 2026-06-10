from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import threading
import time
from typing import Callable

from agent_sentinel.config import Settings
from agent_sentinel.feishu_app import extract_text_from_message_content

logger = logging.getLogger(__name__)


AnalyzeCallback = Callable[
    [str, str, str, str, str | None, str | None, str, list[str] | None, str | None, str | None, str | None],
    tuple[str, bool],
]


class FeishuLongConnectionBot:
    def __init__(
        self,
        settings: Settings,
        analyze_callback: AnalyzeCallback,
        card_action_callback: Callable[[dict[str, object]], dict[str, str]] | None = None,
    ) -> None:
        # 方法说明：保存长连接所需配置和回调，收到飞书事件后会把消息转交给诊断流程。
        self.settings = settings
        self.analyze_callback = analyze_callback
        self.card_action_callback = card_action_callback
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        # 方法说明：检查长连接开关、凭证和 SDK 后启动后台监听线程。
        if self._started:
            logger.info("Feishu long connection listener already started.")
            return
        if not self.settings.feishu_long_connection_enabled:
            logger.info("Feishu long connection listener is disabled.")
            return
        if not (self.settings.feishu_app_id and self.settings.feishu_app_secret):
            logger.info("Feishu long connection listener skipped because app credentials are missing.")
            return
        if importlib.util.find_spec("lark_oapi") is None:
            logger.warning(
                "Feishu long connection listener skipped because lark_oapi is not installed. "
                "Run `pip install lark-oapi` or reinstall project dependencies."
            )
            return

        # 飞书长连接客户端是阻塞式运行，放到 daemon 线程中避免阻塞 FastAPI 主线程。
        self._thread = threading.Thread(
            target=self._run_forever,
            name="feishu-long-connection",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info("Feishu long connection listener thread started.")

    def _run_forever(self) -> None:
        # 方法说明：创建飞书长连接客户端并注册事件处理器，让机器人持续接收消息和卡片点击。
        import lark_oapi as lark

        # EventDispatcherHandler 同时注册消息事件和可选卡片回调事件，统一走长连接通道。
        builder = (
            lark.EventDispatcherHandler.builder(
                self.settings.feishu_event_encrypt_key or "",
                self.settings.feishu_event_verification_token or "",
            )
            .register_p2_im_message_receive_v1(self._handle_message_event)
        )
        if self.card_action_callback is not None:
            builder = builder.register_p2_card_action_trigger(self._handle_card_action_event)
        event_handler = builder.build()

        ws_client = lark.ws.Client(
            app_id=self.settings.feishu_app_id,
            app_secret=self.settings.feishu_app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
            auto_reconnect=True,
        )
        logger.info("Starting Feishu long connection listener.")
        ws_client.start()

    def _handle_message_event(self, data: object) -> None:
        # 方法说明：处理飞书实时消息事件，过滤无效消息后把用户文本投递给分析入口。
        started = time.perf_counter()
        try:
            import lark_oapi as lark

            payload = json.loads(lark.JSON.marshal(data))
            event = payload.get("event") or {}
            sender = event.get("sender") or {}
            if sender.get("sender_type") != "user":
                # 忽略机器人/系统消息，避免机器人回复再次触发自身诊断。
                logger.info("Longconn message ignored sender_type=%s", sender.get("sender_type"))
                return

            message = event.get("message") or {}
            chat_id = str(message.get("chat_id") or "")
            if not chat_id:
                logger.info("Longconn message ignored missing chat_id")
                return

            if self.settings.feishu_allowed_chat_ids and chat_id not in self.settings.feishu_allowed_chat_ids:
                logger.info("Longconn message ignored chat_id not allowed chat_id=%s", chat_id)
                return

            chat_type = str(message.get("chat_type") or "")
            mentions = message.get("mentions") or []
            if self.settings.feishu_analyze_mention_only and chat_type != "p2p" and not mentions:
                # 群聊默认只响应 @ 机器人；私聊不强制 mention，交互体验更自然。
                logger.info(
                    "Longconn message ignored mention required chat_id=%s chat_type=%s message_id=%s",
                    chat_id,
                    chat_type,
                    message.get("message_id"),
                )
                return

            content_text = extract_text_from_message_content(message.get("content"))
            if not content_text.strip():
                logger.info("Longconn message ignored empty content chat_id=%s message_id=%s", chat_id, message.get("message_id"))
                return

            root_message_id = str(message.get("root_id") or "") or str(message.get("message_id") or "") or None
            # root_message_id 用于把后续卡片和文本更新都收敛到同一个消息话题。
            logger.info(
                "Longconn message routed chat_id=%s message_id=%s root_message_id=%s chat_type=%s mentions=%s content_chars=%s elapsed_ms=%s",
                chat_id,
                message.get("message_id"),
                root_message_id,
                chat_type,
                len(mentions),
                len(content_text),
                int((time.perf_counter() - started) * 1000),
            )
            self.analyze_callback(
                chat_id,
                "feishu-user",
                "INFO",
                f"{self.settings.feishu_bot_name} received a message",
                content_text,
                content_text,
                "user_message",
                ["feishu", "long-connection"],
                root_message_id,
                self._extract_sender_open_id(sender),
                self._extract_sender_name(sender),
            )
        except Exception:
            logger.exception("Failed to process Feishu long connection message event")

    def _handle_card_action_event(self, data: object) -> None:
        # 方法说明：处理飞书卡片按钮点击，并把可能耗时的后续动作放到独立线程执行。
        if self.card_action_callback is None:
            return
        try:
            import lark_oapi as lark

            payload = json.loads(lark.JSON.marshal(data))
            value = payload.get("action", {}).get("value", {}) if isinstance(payload.get("action"), dict) else {}
            logger.info(
                "Longconn card action received task_id=%s node=%s action=%s payload_keys=%s",
                value.get("task_id") or value.get("decision_id"),
                value.get("node_name") or value.get("node"),
                value.get("action") or value.get("decision"),
                sorted(payload.keys()),
            )
            # 卡片回调可能继续驱动异步工作流，单独开线程避免阻塞飞书 SDK 的事件分发线程。
            threading.Thread(
                target=self._run_card_action_payload,
                args=(payload,),
                name="feishu-card-action-dispatch",
                daemon=True,
            ).start()
        except Exception:
            logger.exception("Failed to process Feishu long connection card action event")

    def _run_card_action_payload(self, payload: dict[str, object]) -> None:
        # 方法说明：在线程中启动异步分发逻辑，让卡片点击可以继续驱动工作流。
        try:
            asyncio.run(self._dispatch_card_action(payload))
        except Exception:
            logger.exception("Failed to dispatch Feishu long connection card action event")

    async def _dispatch_card_action(self, payload: dict[str, object]) -> None:
        # 方法说明：调用外部注册的卡片处理函数，并兼容同步函数和异步协程两种返回方式。
        if self.card_action_callback is None:
            return
        result = self.card_action_callback(payload)
        if asyncio.iscoroutine(result):
            await result

    def _extract_sender_open_id(self, sender: object) -> str | None:
        # 方法说明：从飞书发送人结构中取出 open_id，用于后续回复时精确提醒用户。
        if not isinstance(sender, dict):
            return None
        sender_id = str(sender.get("sender_id") or sender.get("id") or "").strip()
        return sender_id or None

    def _extract_sender_name(self, sender: object) -> str | None:
        # 方法说明：从飞书发送人结构中取出显示名，用于消息提醒时展示更友好的称呼。
        if not isinstance(sender, dict):
            return None
        name = str(sender.get("name") or "").strip()
        return name or None
