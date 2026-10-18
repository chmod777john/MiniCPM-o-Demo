"""UI Card Demo：ui_card 训练数据的六个工具真实执行，模型调 show_card 时在主面板上画出卡片。

链路（与 board 同构，执行都在本 Demo 的 router 下）：

    模型 tool_call ──(浏览器 onToolCallDone)──▶ POST /api/demos/uicard/tools/execute
                                               └─ tools/executor.py 真调 SERP / Tavily / 高德
    浏览器拿到结果：content → input.tool_result 回灌给模型；cards → 页面的卡片注册表（模型看不到）

天气与地图的卡片由工具拼好、只把编号给模型（训练口径）；模型调 ``show_card({"card_id":"m1"})`` 时页面
按编号查注册表画出来，其余 8 种卡片由模型自己写 ``card`` 对象。

部署配置：外部接口的 key 由部署者在 Demo 服务的环境变量 ``UICARD_TOOLS_CONFIG`` 指向的文件里提供
（仓库外、600 权限，格式见本目录 README.md）。没配时需要 key 的工具如实回错误，其余照常。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from demo_gallery import demo
from demos.uicard.presets import uicard_presets
from demos.uicard.tools.config import configuration_status, map_display_config
from demos.uicard.tools.executor import TOOL_NAMES, execute_tool

router = APIRouter()


@demo(
    demo_id="uicard",
    title="MiniCPM-o · UI Card",
    summary="ui_card 训练口径：天气 / 地图 / 搜索工具真实执行，模型用 show_card 在对话中展示卡片",
    order=30,
    styles=("uicard.css",),
    router=router,
)
class UiCardDemo:
    """defaults 提供各 preset 的训练原样 system 与工具配置状态；通用运行时字段由 Gallery 合并。"""

    def defaults(self) -> dict[str, Any]:
        return {
            "uicard_presets": uicard_presets(),
            "uicard_tools": list(TOOL_NAMES),
            "uicard_tools_config": configuration_status(),
            "uicard_map_display": map_display_config(),
        }


@router.post("/tools/execute")
async def execute(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """浏览器旁路执行一次工具调用。

    请求体 ``{"name", "arguments", "tool_call_id", "options"}``；返回 :class:`ToolExecution`，
    其中只有 ``content`` 回灌给模型。
    """

    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="tool name is required")
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    options = payload.get("options")
    if not isinstance(options, (dict, type(None))):
        raise HTTPException(status_code=400, detail="options must be an object")
    result = await execute_tool(name, arguments, options)
    body = result.model_dump(mode="json")
    body["tool_call_id"] = payload.get("tool_call_id")
    return body
