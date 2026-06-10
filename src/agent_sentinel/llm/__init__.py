from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

from agent_sentinel.config import Settings


def build_chat_model(settings: Settings, model_name: str | None = None) -> ChatOpenAI:
    # 方法说明：构建并返回调用方需要的对象。
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

    kwargs: dict[str, object] = {
        "api_key": settings.openai_api_key,
        "model": model_name or settings.openai_model,
        "temperature": settings.openai_temperature,
        "http_client": httpx.Client(trust_env=settings.openai_http_trust_env),
        "http_async_client": httpx.AsyncClient(trust_env=settings.openai_http_trust_env),
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return ChatOpenAI(**kwargs)
