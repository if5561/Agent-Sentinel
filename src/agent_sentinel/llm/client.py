from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI


# 定义 build_chat_model 相关的处理逻辑，供流程或外部调用复用。
def build_chat_model(
    # 执行当前业务步骤，推动流程继续向下游推进。
    *,
    # 执行当前业务步骤，推动流程继续向下游推进。
    api_key: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    model: str,
    # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
    base_url: str | None = None,
    # 将 temperature 的值保存下来，供后续流程判断或组装响应时使用。
    temperature: float = 0.0,
    # 将 trust_env 的值保存下来，供后续流程判断或组装响应时使用。
    trust_env: bool = False,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> ChatOpenAI:
    # 用传入的 API 参数创建 LangChain ChatOpenAI 客户端，供上层执行器真正调用模型。
    kwargs: dict[str, object] = {
        # 执行当前业务步骤，推动流程继续向下游推进。
        "api_key": api_key,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "model": model,
        # 执行当前业务步骤，推动流程继续向下游推进。
        "temperature": temperature,
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "http_client": httpx.Client(trust_env=trust_env),
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        "http_async_client": httpx.AsyncClient(trust_env=trust_env),
    }
    # 根据 base_url 判断当前流程该进入哪个处理分支。
    if base_url:
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        kwargs["base_url"] = base_url
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return ChatOpenAI(**kwargs)
