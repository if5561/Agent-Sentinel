from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logger(log_level: str = "INFO") -> None:
    # 方法说明：初始化对象，并保存后续调用需要的状态。
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _configure_plain_json_logger(level)


def _configure_plain_json_logger(level: int) -> None:
    # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
    logger = logging.getLogger("agent_sentinel.node_event_json")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    log_path = Path(os.getenv("NODE_EVENT_JSONL_PATH", "/app/logs/node_events.jsonl"))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    except OSError:
        logging.getLogger(__name__).warning("Failed to configure node event JSONL file handler path=%s", log_path)
