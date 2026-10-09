"""MVP 画板 Demo：display_object_on_board 工具 → 搜图 → 画板卡片（1×6 FIFO）。

本文件是这个 Demo 与 Gallery 的唯一接口。增删这个 demo 不需要动任何共享文件。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from demo_gallery import StaticMount, demo
from demos.board.defaults import board_defaults

LIVE_IMAGE_REL = "tools/display_object_on_board/live_image_downloads"

# ★ 工具接口挂在这个 router 上，由 Gallery core 挂到 /api/demos/board/**，
#   绝不触碰 gateway 的主 app。
router = APIRouter()


@demo(
    demo_id="board",
    title="MiniCPM-o · FC Board",
    summary="连续语音中识别具象对象，后台调用 display_object_on_board 放上画板",
    order=10,
    styles=("board.css",),
    router=router,
    extra_mounts=(StaticMount(path=LIVE_IMAGE_REL, create=True),),
)
class BoardDemo:
    """Board 的 defaults 来自训练 case；通用运行时字段由 Gallery core 合并。"""

    def defaults(self) -> dict[str, Any]:
        return board_defaults()


_SERVICE = None


def _service():
    global _SERVICE
    if _SERVICE is None:
        from demos.board.tools.display_object_on_board.service import (
            DisplayObjectOnBoardService,
        )

        _SERVICE = DisplayObjectOnBoardService(
            download_dir=Path(__file__).resolve().parent / LIVE_IMAGE_REL
        )
    return _SERVICE


@router.post("/tools/display_object_on_board")
async def display_object_on_board(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """浏览器旁路执行画板工具。

    session.init 用 ``auto_execute_tools=false``：模型只负责发出 tool_call，
    执行走「浏览器 → 本接口 → demo 自己的 python 服务」，再由浏览器把结果经 WS 回灌。
    """

    query = str(payload.get("name") or payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="display_object_on_board requires name/query")
    tool_call_id = payload.get("tool_call_id")

    from demos.board.tools.display_object_on_board.service import (
        board_image_result_from_tool_result,
    )

    result = await asyncio.to_thread(_service().search, query)
    image = board_image_result_from_tool_result(
        result, tool_call_id=str(tool_call_id or query)
    )
    return {
        "query": result.query,
        "image": image.model_dump(),
        "error": result.error,
        "tool_response_content": result.tool_response_content,
    }
