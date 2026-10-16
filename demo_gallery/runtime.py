"""Demo 服务自己的运行时配置（与 API 部署无关，所有 demo 页面共用）。

Demo 与 API 只通过协议耦合：模型是谁、口径如何，由页面在会话前向所选 API 端点的
``/v1/info`` 查询，**不读任何 API 进程的环境变量**。这里只放 Demo 自己的配置：

- ``DEMO_API_ENDPOINTS``：页面下拉里可选的 API 端点，JSON 列表 ``[{"name": "...", "url": "https://host:port/e/<端点>"}]``；
- ``DEMO_DEFAULT_API``：页面首次打开时选中的端点 URL（缺省取列表第一个）；
- ``DEMO_API_DISCOVERY``：要自动发现端点的 API 服务地址（``https://host:port``，逗号或空白分隔）。页面在浏览器里
  读各服务的公开状态 ``<地址>/overview/api/state``，把上面的全部端点（含模型、可用 Worker 数）列进下拉框，
  控制面板上新建 / 删除端点后 Demo 不用改配置。缺省取上面两项里出现过的服务地址；设为空字符串则关闭自动发现；
- ``DEMO_DEFAULT_PRESET``：首次打开时的 preset id（不属于当前 demo 时由前端回退到 demo 自己的默认）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger("demo_gallery")


def api_endpoints() -> list[dict[str, str]]:
    raw = os.environ.get("DEMO_API_ENDPOINTS") or "[]"
    try:
        items = json.loads(raw)
    except ValueError:
        logger.warning("[gallery] DEMO_API_ENDPOINTS 不是合法 JSON，忽略: %s", raw[:200])
        return []
    endpoints = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("url"), str) and item["url"].strip():
            url = item["url"].strip().rstrip("/")
            endpoints.append({"name": str(item.get("name") or url), "url": url})
    return endpoints


def _origin(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme in ("http", "https") and parts.netloc else ""


def api_discovery(endpoints: list[dict[str, str]], default_api: str) -> list[str]:
    """要自动发现端点的 API 服务地址（去重、保持顺序）。"""

    raw = os.environ.get("DEMO_API_DISCOVERY")
    candidates = raw.replace(",", " ").split() if raw is not None else [e["url"] for e in endpoints] + [default_api]
    origins: list[str] = []
    for item in candidates:
        origin = _origin(item)
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def gallery_runtime() -> dict[str, Any]:
    """所有 demo 页面共用的 Demo 侧配置；每个 demo 的 ``/defaults`` 在此之上合并自己的字段。"""

    endpoints = api_endpoints()
    default_api = (os.environ.get("DEMO_DEFAULT_API") or "").strip().rstrip("/")
    return {
        "api_endpoints": endpoints,
        "default_api_endpoint": default_api or (endpoints[0]["url"] if endpoints else ""),
        "api_discovery": api_discovery(endpoints, default_api),
        "default_quick_setting": os.environ.get("DEMO_DEFAULT_PRESET") or None,
    }
