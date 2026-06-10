from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    # 方法说明：读取 YAML 配置文件，并把相对路径统一解释为项目根目录下的路径。
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return data


def load_prompt(name: str, prompts_dir: str | Path = "config/prompts") -> dict[str, str]:
    # 方法说明：按名称读取提示词模板，返回 system 和 human 两段内容。
    prompt = load_yaml(Path(prompts_dir) / f"{name}.yaml")
    return {
        "system": str(prompt.get("system", "")),
        "human": str(prompt.get("human", "")),
    }


def format_prompt(name: str, variables: dict[str, Any], prompts_dir: str | Path = "config/prompts") -> str:
    # 方法说明：把变量填入提示词模板，生成最终发给大模型的完整文本。
    prompt = load_prompt(name, prompts_dir)
    system = prompt["system"].format(**variables)
    human = prompt["human"].format(**variables)
    return f"{system}\n\n{human}".strip()


def load_env() -> None:
    # 方法说明：加载项目根目录下的 .env 文件，让本地开发也能使用环境变量配置。
    load_dotenv(PROJECT_ROOT / ".env")
