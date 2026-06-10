from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI


def build_chat_model(
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    trust_env: bool = False,
) -> ChatOpenAI:
    # 方法说明：构建并返回调用方需要的对象。
    kwargs: dict[str, object] = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "http_client": httpx.Client(trust_env=trust_env),
        "http_async_client": httpx.AsyncClient(trust_env=trust_env),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
