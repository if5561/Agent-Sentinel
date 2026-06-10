from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    # 方法说明：读取并返回当前流程需要的数据。
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
    # 方法说明：读取并返回当前流程需要的数据。
    prompt = load_yaml(Path(prompts_dir) / f"{name}.yaml")
    return {
        "system": str(prompt.get("system", "")),
        "human": str(prompt.get("human", "")),
    }


def format_prompt(name: str, variables: dict[str, Any], prompts_dir: str | Path = "config/prompts") -> str:
    # 方法说明：构建并返回调用方需要的对象。
    prompt = load_prompt(name, prompts_dir)
    system = prompt["system"].format(**variables)
    human = prompt["human"].format(**variables)
    return f"{system}\n\n{human}".strip()


def load_env() -> None:
    # 方法说明：读取并返回当前流程需要的数据。
    load_dotenv(PROJECT_ROOT / ".env")
