"""uicard 工具的部署配置：各外部接口的地址与 key。

配置文件由部署者提供，路径写在 Demo 服务的环境变量 ``UICARD_TOOLS_CONFIG`` 里（文件放仓库外、600 权限）。
**没配就不调用外部接口**：需要 key 的工具如实给模型回一个错误，不回退到任何人的默认 key。

文件按 mtime 缓存，改了 key 不用重启 Demo 服务。格式见 ``demos/uicard/README.md``。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

CONFIG_ENV = "UICARD_TOOLS_CONFIG"

# 只有不涉及账号的字段才有默认值；api_key 一律没有默认值。
DEFAULTS: dict[str, dict[str, Any]] = {
    "serp_google_search": {
        "endpoint": "https://llm-center.modelbest.co/llm/mcp?mcp=serp-search",
        "timeout_sec": 20.0,
        "retry_budget_sec": 9.0,
    },
    "tavily": {
        "endpoint": "https://api.tavily.com/search",
        "search_depth": "basic",
        "timeout_sec": 15.0,
    },
    "amap_weather": {
        "endpoint": "https://restapi.amap.com/v3/weather/weatherInfo",
        "timeout_sec": 10.0,
    },
    "map_search": {
        "base_url": "https://restapi.amap.com",
        "timeout_sec": 10.0,
    },
    # 训练数据里定位是冻结值；Demo 也给一个固定位置（北京海淀中关村）。
    "get_client_location": {
        "latitude": 39.9593,
        "longitude": 116.2979,
        "accuracy": 28,
        "city": "北京市",
    },
}

# 浏览器画地图用的高德 JS API 凭证（与 map_search 的 Web 服务 key 不是同一种）；不配就用 OSM 底图。
MAP_DISPLAY_FIELDS = ("amap_js_api_key", "amap_js_security_code", "amap_js_service_host")

# 需要 key 才能调用的配置段。
KEYED_SECTIONS = ("serp_google_search", "tavily", "amap_weather", "map_search")

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"path": None, "mtime": None, "data": {}}


def config_path() -> Path | None:
    """当前生效的配置文件路径；环境变量未设置时为 None。"""

    raw = os.environ.get(CONFIG_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def _load_file() -> dict[str, Any]:
    path = config_path()
    if path is None:
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _LOCK:
        if _CACHE["path"] == str(path) and _CACHE["mtime"] == mtime:
            return _CACHE["data"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    with _LOCK:
        _CACHE.update({"path": str(path), "mtime": mtime, "data": data})
    return data


def tool_config(section: str) -> dict[str, Any]:
    """合并默认值与配置文件里的一个段。

    参数:
        section: 配置段名，如 ``amap_weather``、``tavily``。

    返回:
        设置字典；配置文件里空值（None / 空串）不覆盖默认值。
    """

    merged = dict(DEFAULTS.get(section, {}))
    override = _load_file().get(section)
    if isinstance(override, dict):
        for key, value in override.items():
            if value not in (None, ""):
                merged[key] = value
    return merged


def configuration_status() -> dict[str, Any]:
    """哪些配置段有可用 key（只报布尔值，不泄露 key 本身）。"""

    path = config_path()
    return {
        "config_env": CONFIG_ENV,
        "config_set": path is not None,
        "config_exists": bool(path and path.is_file()),
        "configured": {
            section: bool(str(tool_config(section).get("api_key") or "").strip())
            for section in KEYED_SECTIONS
        },
    }


def map_display_config() -> dict[str, str]:
    """下发给浏览器的地图凭证；JS key 本来就是给浏览器用的。缺安全码与代理时视为未配置。"""

    section = _load_file().get("map_display")
    if not isinstance(section, dict):
        return {}
    merged = {key: str(section.get(key) or "").strip() for key in MAP_DISPLAY_FIELDS}
    if not merged["amap_js_api_key"]:
        return {}
    if not merged["amap_js_security_code"] and not merged["amap_js_service_host"]:
        return {}
    return {key: value for key, value in merged.items() if value}
