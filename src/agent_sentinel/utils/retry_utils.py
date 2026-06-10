from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# 将 F 的值保存下来，供后续流程判断或组装响应时使用。
F = TypeVar("F", bound=Callable[..., Any])


# 定义 async_retry 相关的处理逻辑，供流程或外部调用复用。
def async_retry(max_attempts: int = 2) -> Callable[[F], F]:
    # 生成异步重试装饰器，遇到临时异常时按指数退避重试，最后仍失败就抛出原异常。
    return retry(
        # 将 stop 的值保存下来，供后续流程判断或组装响应时使用。
        stop=stop_after_attempt(max_attempts),
        # 将 wait 的值保存下来，供后续流程判断或组装响应时使用。
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        # 将 retry 的值保存下来，供后续流程判断或组装响应时使用。
        retry=retry_if_exception_type(Exception),
        # 将 reraise 的值保存下来，供后续流程判断或组装响应时使用。
        reraise=True,
    )
