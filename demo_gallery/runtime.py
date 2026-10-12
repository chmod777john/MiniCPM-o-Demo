"""Demo 服务自己的运行时配置（与 API 部署无关，所有 demo 页面共用）。

Demo 与 API 只通过协议耦合：模型是谁、口径如何，由页面在会话前向所选 API 端点的
``/v1/info`` 查询，**不读任何 API 进程的环境变量**。这里只放 Demo 自己的配置：

- ``DEMO_API_ENDPOINTS``：页面下拉里可选的 API 端点，JSON 列表 ``[{"name": "...", "url": "https://host:port/e/<端点>"}]``；
- ``DEMO_DEFAULT_API``：页面首次打开时选中的端点 URL（缺省取列表第一个）；
- ``DEMO_DEFAULT_PRESET``：首次打开时的 preset id（不属于当前 demo 时由前端回退到 demo 自己的默认）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

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


def gallery_runtime() -> dict[str, Any]:
    """所有 demo 页面共用的 Demo 侧配置；每个 demo 的 ``/defaults`` 在此之上合并自己的字段。"""

    endpoints = api_endpoints()
    default_api = (os.environ.get("DEMO_DEFAULT_API") or "").strip().rstrip("/")
    return {
        "api_endpoints": endpoints,
        "default_api_endpoint": default_api or (endpoints[0]["url"] if endpoints else ""),
        "default_quick_setting": os.environ.get("DEMO_DEFAULT_PRESET") or None,
    }
