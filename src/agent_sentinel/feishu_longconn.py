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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


# 将 AnalyzeCallback 的值保存下来，供后续流程判断或组装响应时使用。
AnalyzeCallback = Callable[
    # 执行当前业务步骤，推动流程继续向下游推进。
    [str, str, str, str, str | None, str | None, str, list[str] | None, str | None, str | None, str | None],
    # 执行当前业务步骤，推动流程继续向下游推进。
    tuple[str, bool],
]


# 定义 FeishuLongConnectionBot 组件，集中管理这个模块的状态和行为。
class FeishuLongConnectionBot:
    # 定义 __init__ 相关的处理逻辑，供流程或外部调用复用。
    def __init__(
        # 执行当前业务步骤，推动流程继续向下游推进。
        self,
        # 执行当前业务步骤，推动流程继续向下游推进。
        settings: Settings,
        # 执行当前业务步骤，推动流程继续向下游推进。
        analyze_callback: AnalyzeCallback,
        # 将 card_action_callback 的值保存下来，供后续流程判断或组装响应时使用。
        card_action_callback: Callable[[dict[str, object]], dict[str, str]] | None = None,
    # 执行当前业务步骤，推动流程继续向下游推进。
    ) -> None:
        # 保存长连接所需配置和回调，收到飞书事件后会把消息转交给诊断流程。
        self.settings = settings
        # 将 self.analyze_callback 的值保存下来，供后续流程判断或组装响应时使用。
        self.analyze_callback = analyze_callback
        # 将 self.card_action_callback 的值保存下来，供后续流程判断或组装响应时使用。
        self.card_action_callback = card_action_callback
        # 将 self._thread 的值保存下来，供后续流程判断或组装响应时使用。
        self._thread: threading.Thread | None = None
        # 将 self._started 的值保存下来，供后续流程判断或组装响应时使用。
        self._started = False

    # 定义 start 相关的处理逻辑，供流程或外部调用复用。
    def start(self) -> None:
        # 检查长连接开关、凭证和 SDK 后启动后台监听线程。
        if self._started:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu long connection listener already started.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 not self.settings.feishu_long_connection_enabled 判断当前流程该进入哪个处理分支。
        if not self.settings.feishu_long_connection_enabled:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu long connection listener is disabled.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 not (self.settings.feishu_app_id and self.settings.f... 判断当前流程该进入哪个处理分支。
        if not (self.settings.feishu_app_id and self.settings.feishu_app_secret):
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Feishu long connection listener skipped because app credentials are missing.")
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 根据 importlib.util.find_spec("lark_oapi") is None 判断当前流程该进入哪个处理分支。
        if importlib.util.find_spec("lark_oapi") is None:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.warning(
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Feishu long connection listener skipped because lark_oapi is not installed. "
                # 执行当前业务步骤，推动流程继续向下游推进。
                "Run `pip install lark-oapi` or reinstall project dependencies."
            )
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return

        # 飞书长连接客户端是阻塞式运行，放到 daemon 线程中避免阻塞 FastAPI 主线程。
        self._thread = threading.Thread(
            # 将 target 的值保存下来，供后续流程判断或组装响应时使用。
            target=self._run_forever,
            # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
            name="feishu-long-connection",
            # 将 daemon 的值保存下来，供后续流程判断或组装响应时使用。
            daemon=True,
        )
        # 调用 _thread.start 完成当前步骤需要的业务处理。
        self._thread.start()
        # 将 self._started 的值保存下来，供后续流程判断或组装响应时使用。
        self._started = True
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Feishu long connection listener thread started.")

    # 定义 _run_forever 相关的处理逻辑，供流程或外部调用复用。
    def _run_forever(self) -> None:
        # 创建飞书长连接客户端并注册事件处理器，让机器人持续接收消息和卡片点击。
        import lark_oapi as lark

        # EventDispatcherHandler 同时注册消息事件和可选卡片回调事件，统一走长连接通道。
        builder = (
            # 调用 EventDispatcherHandler.builder 完成当前步骤需要的业务处理。
            lark.EventDispatcherHandler.builder(
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.settings.feishu_event_encrypt_key or "",
                # 执行当前业务步骤，推动流程继续向下游推进。
                self.settings.feishu_event_verification_token or "",
            )
            # 调用 register_p2_im_message_receive_v1 完成当前步骤需要的业务处理。
            .register_p2_im_message_receive_v1(self._handle_message_event)
        )
        # 根据 self.card_action_callback is not None 判断当前流程该进入哪个处理分支。
        if self.card_action_callback is not None:
            # 将 builder 的值保存下来，供后续流程判断或组装响应时使用。
            builder = builder.register_p2_card_action_trigger(self._handle_card_action_event)
        # 将 event_handler 的值保存下来，供后续流程判断或组装响应时使用。
        event_handler = builder.build()

        # 将 ws_client 的值保存下来，供后续流程判断或组装响应时使用。
        ws_client = lark.ws.Client(
            # 将 app_id 的值保存下来，供后续流程判断或组装响应时使用。
            app_id=self.settings.feishu_app_id,
            # 将 app_secret 的值保存下来，供后续流程判断或组装响应时使用。
            app_secret=self.settings.feishu_app_secret,
            # 将 event_handler 的值保存下来，供后续流程判断或组装响应时使用。
            event_handler=event_handler,
            # 将 log_level 的值保存下来，供后续流程判断或组装响应时使用。
            log_level=lark.LogLevel.INFO,
            # 将 auto_reconnect 的值保存下来，供后续流程判断或组装响应时使用。
            auto_reconnect=True,
        )
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Starting Feishu long connection listener.")
        # 调用 ws_client.start 完成当前步骤需要的业务处理。
        ws_client.start()

    # 定义 _handle_message_event 相关的处理逻辑，供流程或外部调用复用。
    def _handle_message_event(self, data: object) -> None:
        # 处理飞书实时消息事件，过滤无效消息后把用户文本投递给分析入口。
        started = time.perf_counter()
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            import lark_oapi as lark

            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = json.loads(lark.JSON.marshal(data))
            # 将 event 的值保存下来，供后续流程判断或组装响应时使用。
            event = payload.get("event") or {}
            # 将 sender 的值保存下来，供后续流程判断或组装响应时使用。
            sender = event.get("sender") or {}
            # 根据 sender.get("sender_type") != "user" 判断当前流程该进入哪个处理分支。
            if sender.get("sender_type") != "user":
                # 忽略机器人/系统消息，避免机器人回复再次触发自身诊断。
                logger.info("Longconn message ignored sender_type=%s", sender.get("sender_type"))
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return

            # 将 message 的值保存下来，供后续流程判断或组装响应时使用。
            message = event.get("message") or {}
            # 将 chat_id 的值保存下来，供后续流程判断或组装响应时使用。
            chat_id = str(message.get("chat_id") or "")
            # 根据 not chat_id 判断当前流程该进入哪个处理分支。
            if not chat_id:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Longconn message ignored missing chat_id")
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return

            # 根据 self.settings.feishu_allowed_chat_ids and chat_id no... 判断当前流程该进入哪个处理分支。
            if self.settings.feishu_allowed_chat_ids and chat_id not in self.settings.feishu_allowed_chat_ids:
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Longconn message ignored chat_id not allowed chat_id=%s", chat_id)
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return

            # 将 chat_type 的值保存下来，供后续流程判断或组装响应时使用。
            chat_type = str(message.get("chat_type") or "")
            # 将 mentions 的值保存下来，供后续流程判断或组装响应时使用。
            mentions = message.get("mentions") or []
            # 根据 self.settings.feishu_analyze_mention_only and chat_t... 判断当前流程该进入哪个处理分支。
            if self.settings.feishu_analyze_mention_only and chat_type != "p2p" and not mentions:
                # 群聊默认只响应 @ 机器人；私聊不强制 mention，交互体验更自然。
                logger.info(
                    # 保存当前计算结果，供后续流程判断或组装响应时使用。
                    "Longconn message ignored mention required chat_id=%s chat_type=%s message_id=%s",
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    chat_id,
                    # 执行当前业务步骤，推动流程继续向下游推进。
                    chat_type,
                    # 调用 message.get 完成当前步骤需要的业务处理。
                    message.get("message_id"),
                )
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return

            # 将 content_text 的值保存下来，供后续流程判断或组装响应时使用。
            content_text = extract_text_from_message_content(message.get("content"))
            # 根据 not content_text.strip() 判断当前流程该进入哪个处理分支。
            if not content_text.strip():
                # 记录关键运行信息，方便排查流程进展和异常现场。
                logger.info("Longconn message ignored empty content chat_id=%s message_id=%s", chat_id, message.get("message_id"))
                # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
                return

            # 将 root_message_id 的值保存下来，供后续流程判断或组装响应时使用。
            root_message_id = str(message.get("root_id") or "") or str(message.get("message_id") or "") or None
            # root_message_id 用于把后续卡片和文本更新都收敛到同一个消息话题。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Longconn message routed chat_id=%s message_id=%s root_message_id=%s chat_type=%s mentions=%s content_chars=%s elapsed_ms=%s",
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_id,
                # 调用 message.get 完成当前步骤需要的业务处理。
                message.get("message_id"),
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 执行当前业务步骤，推动流程继续向下游推进。
                chat_type,
                # 调用 len 完成当前步骤需要的业务处理。
                len(mentions),
                # 调用 len 完成当前步骤需要的业务处理。
                len(content_text),
                # 调用 int 完成当前步骤需要的业务处理。
                int((time.perf_counter() - started) * 1000),
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
                f"{self.settings.feishu_bot_name} received a message",
                # 执行当前业务步骤，推动流程继续向下游推进。
                content_text,
                # 执行当前业务步骤，推动流程继续向下游推进。
                content_text,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "user_message",
                # 执行当前业务步骤，推动流程继续向下游推进。
                ["feishu", "long-connection"],
                # 执行当前业务步骤，推动流程继续向下游推进。
                root_message_id,
                # 调用 self._extract_sender_open_id 完成当前步骤需要的业务处理。
                self._extract_sender_open_id(sender),
                # 调用 self._extract_sender_name 完成当前步骤需要的业务处理。
                self._extract_sender_name(sender),
            )
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to process Feishu long connection message event")

    # 定义 _handle_card_action_event 相关的处理逻辑，供流程或外部调用复用。
    def _handle_card_action_event(self, data: object) -> None:
        # 处理飞书卡片按钮点击，并把可能耗时的后续动作放到独立线程执行。
        if self.card_action_callback is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 进入可能失败的处理块，便于后续统一捕获和恢复。
        try:
            import lark_oapi as lark

            # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
            payload = json.loads(lark.JSON.marshal(data))
            # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
            value = payload.get("action", {}).get("value", {}) if isinstance(payload.get("action"), dict) else {}
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info(
                # 保存当前计算结果，供后续流程判断或组装响应时使用。
                "Longconn card action received task_id=%s node=%s action=%s payload_keys=%s",
                # 调用 value.get 完成当前步骤需要的业务处理。
                value.get("task_id") or value.get("decision_id"),
                # 调用 value.get 完成当前步骤需要的业务处理。
                value.get("node_name") or value.get("node"),
                # 调用 value.get 完成当前步骤需要的业务处理。
                value.get("action") or value.get("decision"),
                # 调用 sorted 完成当前步骤需要的业务处理。
                sorted(payload.keys()),
            )
            # 卡片回调可能继续驱动异步工作流，单独开线程避免阻塞飞书 SDK 的事件分发线程。
            threading.Thread(
                # 将 target 的值保存下来，供后续流程判断或组装响应时使用。
                target=self._run_card_action_payload,
                # 将 args 的值保存下来，供后续流程判断或组装响应时使用。
                args=(payload,),
                # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
                name="feishu-card-action-dispatch",
                # 将 daemon 的值保存下来，供后续流程判断或组装响应时使用。
                daemon=True,
            # 调用 start 完成当前步骤需要的业务处理。
            ).start()
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to process Feishu long connection card action event")

    # 定义 _run_card_action_payload 相关的处理逻辑，供流程或外部调用复用。
    def _run_card_action_payload(self, payload: dict[str, object]) -> None:
        # 在线程中启动异步分发逻辑，让卡片点击可以继续驱动工作流。
        try:
            # 调用 asyncio.run 完成当前步骤需要的业务处理。
            asyncio.run(self._dispatch_card_action(payload))
        # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
        except Exception:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.exception("Failed to dispatch Feishu long connection card action event")

    # 定义 _dispatch_card_action 相关的处理逻辑，供流程或外部调用复用。
    async def _dispatch_card_action(self, payload: dict[str, object]) -> None:
        # 调用外部注册的卡片处理函数，并兼容同步函数和异步协程两种返回方式。
        if self.card_action_callback is None:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return
        # 将 result 的值保存下来，供后续流程判断或组装响应时使用。
        result = self.card_action_callback(payload)
        # 根据 asyncio.iscoroutine(result) 判断当前流程该进入哪个处理分支。
        if asyncio.iscoroutine(result):
            # 等待异步操作完成，再继续推进当前业务流程。
            await result

    # 定义 _extract_sender_open_id 相关的处理逻辑，供流程或外部调用复用。
    def _extract_sender_open_id(self, sender: object) -> str | None:
        # 从飞书发送人结构中取出 open_id，用于后续回复时精确提醒用户。
        if not isinstance(sender, dict):
            return None
        # 将 sender_id 的值保存下来，供后续流程判断或组装响应时使用。
        sender_id = str(sender.get("sender_id") or sender.get("id") or "").strip()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return sender_id or None

    # 定义 _extract_sender_name 相关的处理逻辑，供流程或外部调用复用。
    def _extract_sender_name(self, sender: object) -> str | None:
        # 从飞书发送人结构中取出显示名，用于消息提醒时展示更友好的称呼。
        if not isinstance(sender, dict):
            return None
        # 将 name 的值保存下来，供后续流程判断或组装响应时使用。
        name = str(sender.get("name") or "").strip()
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return name or None
