from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

F = TypeVar("F", bound=Callable[..., Any])


def async_retry(max_attempts: int = 2) -> Callable[[F], F]:
    # 生成异步重试装饰器，遇到临时异常时按指数退避重试，最后仍失败就抛出原异常。
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
