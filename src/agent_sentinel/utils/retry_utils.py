from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

F = TypeVar("F", bound=Callable[..., Any])


def async_retry(max_attempts: int = 2) -> Callable[[F], F]:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
