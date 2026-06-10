from __future__ import annotations

import logging
import os
from pathlib import Path


# 定义 configure_logger 相关的处理逻辑，供流程或外部调用复用。
def configure_logger(log_level: str = "INFO") -> None:
    # 初始化全局日志格式和级别，让控制台日志保持统一可读。
    level = getattr(logging, log_level.upper(), logging.INFO)
    # 调用 logging.basicConfig 完成当前步骤需要的业务处理。
    logging.basicConfig(
        # 将 level 的值保存下来，供后续流程判断或组装响应时使用。
        level=level,
        # 将 format 的值保存下来，供后续流程判断或组装响应时使用。
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # 调用 _configure_plain_json_logger 完成当前步骤需要的业务处理。
    _configure_plain_json_logger(level)


# 定义 _configure_plain_json_logger 相关的处理逻辑，供流程或外部调用复用。
def _configure_plain_json_logger(level: int) -> None:
    # 单独配置节点事件 JSON 日志，方便后续被文件采集器或测试用例读取。
    logger = logging.getLogger("agent_sentinel.node_event_json")
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.setLevel(level)
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.propagate = False
    # 根据 logger.handlers 判断当前流程该进入哪个处理分支。
    if logger.handlers:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return
    # 将 stream_handler 的值保存下来，供后续流程判断或组装响应时使用。
    stream_handler = logging.StreamHandler()
    # 调用 stream_handler.setFormatter 完成当前步骤需要的业务处理。
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    # 记录关键运行信息，方便排查流程进展和异常现场。
    logger.addHandler(stream_handler)

    # 将 log_path 的值保存下来，供后续流程判断或组装响应时使用。
    log_path = Path(os.getenv("NODE_EVENT_JSONL_PATH", "/app/logs/node_events.jsonl"))
    # 进入可能失败的处理块，便于后续统一捕获和恢复。
    try:
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 将 file_handler 的值保存下来，供后续流程判断或组装响应时使用。
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        # 调用 file_handler.setFormatter 完成当前步骤需要的业务处理。
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.addHandler(file_handler)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except OSError:
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        logging.getLogger(__name__).warning("Failed to configure node event JSONL file handler path=%s", log_path)
