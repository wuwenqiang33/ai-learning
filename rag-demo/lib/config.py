"""
统一配置加载工具

从 config.json 中读取 API Key 和路径配置，支持默认值回退。

使用方式:
    from config import load_config
    cfg = load_config()
    api_key = cfg["embedding"]["api_key"]
"""

import os
import json
from pathlib import Path


def load_config(config_filename: str = "config.json") -> dict:
    """加载配置文件，返回字典"""
    config_path = Path(__file__).parent.parent / config_filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"提示: 复制 config.example.json -> {config_filename} 并填入你的 API Key"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env(key: str, default: str = "") -> str:
    """从环境变量读取配置，支持回退到默认值"""
    return os.environ.get(key, default)
