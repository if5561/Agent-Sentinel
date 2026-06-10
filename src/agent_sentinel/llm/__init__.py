from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

from agent_sentinel.config import Settings


# 定义 build_chat_model 相关的处理逻辑，供流程或外部调用复用。
def build_chat_model(settings: Settings, model_name: str | None = None) -> ChatOpenAI:
    # 根据项目配置创建聊天模型客户端，统一注入模型名、温度和 HTTP 代理策略。
    if not settings.openai_api_key:
        # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
        raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

    # 将 kwargs 的值保存下来，供后续流程判断或组装响应时使用。
    kwargs: dict[str, object] = {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "api_key": settings.openai_api_key,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "model": model_name or settings.openai_model,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "temperature": settings.openai_temperature,
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "http_client": httpx.Client(trust_env=settings.openai_http_trust_env),
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "http_async_client": httpx.AsyncClient(trust_env=settings.openai_http_trust_env),
    }
    # 根据 settings.openai_base_url 判断当前流程该进入哪个处理分支。
    if settings.openai_base_url:
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        kwargs["base_url"] = settings.openai_base_url

    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return ChatOpenAI(**kwargs)
