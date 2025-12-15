from copy import deepcopy
from pathlib import Path

import yaml
from loguru import logger

GLOBAL_CONFIG = None

DEFAULT_CONFIG = {
    "capture": {
        "fps": 5.0,
        "frame_step": 2,
        "mode": "fullscreen",
        "output_dir": "data/recordings",
        "fullscreen": {
            "width": 1920,
            "height": 1080,
        },
    },
    "target_resolution": {
        "width": 3840,
        "height": 2160,
    },
    "hotkeys": {
        "start": "f9",
        "stop": "f10",
    },
}


def GetConfig():
    """Load config once and return copy of cached dict."""
    global GLOBAL_CONFIG
    if GLOBAL_CONFIG is None:
        path = GetConfigPaht()
        ensure_config_file(path)
        GLOBAL_CONFIG = merge_with_defaults(DEFAULT_CONFIG, load_yaml(path))
    return deepcopy(GLOBAL_CONFIG)


def SaveConfig(config):
    """Persist config to disk and refresh cache."""
    global GLOBAL_CONFIG
    path = GetConfigPaht()
    ensure_config_file(path)
    merged = merge_with_defaults(DEFAULT_CONFIG, config)
    try:
        with open(path, "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(merged, file_obj, allow_unicode=True, default_flow_style=False)
        GLOBAL_CONFIG = deepcopy(merged)
        return True
    except Exception as exc:
        logger.error(f"保存配置失败: {exc}")
        return False


def GetConfigPaht():
    return Path(__file__).parent.parent / "config" / "client_config.yaml"


def ensure_config_file(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(DEFAULT_CONFIG, file_obj, allow_unicode=True, default_flow_style=False)


def load_yaml(config_path: Path):
    try:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj) or {}
    except FileNotFoundError:
        return deepcopy(DEFAULT_CONFIG)
    except Exception as exc:
        logger.error(f"读取配置失败，使用默认值: {exc}")
        return deepcopy(DEFAULT_CONFIG)


def merge_with_defaults(defaults: dict, config: dict):
    """Recursively merge defaults into config without mutating arguments."""
    if not isinstance(config, dict):
        return deepcopy(defaults)
    merged = {}
    for key, default_value in defaults.items():
        if isinstance(default_value, dict):
            merged[key] = merge_with_defaults(default_value, config.get(key, {}))
        else:
            merged[key] = config.get(key, default_value)
    for key, value in config.items():
        if key not in merged:
            merged[key] = value
    return merged
