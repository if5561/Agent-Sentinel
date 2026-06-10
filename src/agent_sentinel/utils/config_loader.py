from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# 将 PROJECT_ROOT 的值保存下来，供后续流程判断或组装响应时使用。
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# 定义 load_yaml 相关的处理逻辑，供流程或外部调用复用。
def load_yaml(path: str | Path) -> dict[str, Any]:
    # 读取 YAML 配置文件，并把相对路径统一解释为项目根目录下的路径。
    resolved = Path(path)
    # 根据 not resolved.is_absolute() 判断当前流程该进入哪个处理分支。
    if not resolved.is_absolute():
        # 将 resolved 的值保存下来，供后续流程判断或组装响应时使用。
        resolved = PROJECT_ROOT / resolved
    # 根据 not resolved.exists() 判断当前流程该进入哪个处理分支。
    if not resolved.exists():
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {}
    # 进入上下文管理器保护的区域，自动处理资源生命周期。
    with resolved.open("r", encoding="utf-8") as file:
        # 将 data 的值保存下来，供后续流程判断或组装响应时使用。
        data = yaml.safe_load(file) or {}
    # 根据 not isinstance(data, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(data, dict):
        # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return data


# 定义 load_prompt 相关的处理逻辑，供流程或外部调用复用。
def load_prompt(name: str, prompts_dir: str | Path = "config/prompts") -> dict[str, str]:
    # 按名称读取提示词模板，返回 system 和 human 两段内容。
    prompt = load_yaml(Path(prompts_dir) / f"{name}.yaml")
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return {
        # 调用 str 完成当前步骤需要的业务处理。
        "system": str(prompt.get("system", "")),
        # 调用 str 完成当前步骤需要的业务处理。
        "human": str(prompt.get("human", "")),
    }


# 定义 format_prompt 相关的处理逻辑，供流程或外部调用复用。
def format_prompt(name: str, variables: dict[str, Any], prompts_dir: str | Path = "config/prompts") -> str:
    # 把变量填入提示词模板，生成最终发给大模型的完整文本。
    prompt = load_prompt(name, prompts_dir)
    # 将 system 的值保存下来，供后续流程判断或组装响应时使用。
    system = prompt["system"].format(**variables)
    # 将 human 的值保存下来，供后续流程判断或组装响应时使用。
    human = prompt["human"].format(**variables)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return f"{system}\n\n{human}".strip()


# 定义 load_env 相关的处理逻辑，供流程或外部调用复用。
def load_env() -> None:
    # 加载项目根目录下的 .env 文件，让本地开发也能使用环境变量配置。
    load_dotenv(PROJECT_ROOT / ".env")
