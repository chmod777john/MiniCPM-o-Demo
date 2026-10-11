"""subagent Demo：sub_agent 训练数据的 system 与工具；模型的工具调用由 Claude Code 执行。

链路（与 board 的工具链路同构，执行都在本 Demo 的 router 下）：

    模型 tool_call ──(浏览器 onToolCallDone)──▶ POST /api/demos/subagent/tasks
                                               └─ claude_runner 起 headless Claude Code
    浏览器轮询 GET /tasks/<id>/events ◀── progress / result / error
      progress → input.standalone_event（delegate_task 的进度，训练形态）
      result   → input.tool_result
      error    → input.tool_result（如实告诉模型执行失败）

每个 Demo 会话一个工作区 ``workspace/<workspace_id>/``（会话开始时为空），Claude Code 只能读写
该目录；用户上传的文件也落在这里，并以 ``[FILE_UPLOADED] 文件名`` 的 standalone 通知模型（训练形态）。
"""

from __future__ import annotations

import base64
import re
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from demo_gallery import demo
from demos.subagent.claude_runner import HANDLED_TOOLS, runner
from demos.subagent.presets import subagent_presets

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SAFE_NAME_RE = re.compile(r"^[^/\\\x00]{1,200}$")


@demo(
    demo_id="subagent",
    title="MiniCPM-o · Subagent（Claude Code 执行工具）",
    summary="sub_agent 训练口径；模型的 delegate_task / web_search / calculate 交给 Claude Code，在每个会话独立的工作区里执行",
    order=40,
    styles=("subagent.css",),
    router=router,
)
class SubagentDemo:
    """defaults 提供各 preset 的训练原样 system；通用运行时字段由 Gallery 合并。"""

    def defaults(self) -> dict[str, Any]:
        return {"subagent_presets": subagent_presets(), "subagent_handled_tools": list(HANDLED_TOOLS)}


def _ws(workspace_id: str):
    try:
        return runner.workspace(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/workspaces/{workspace_id}")
async def open_workspace(workspace_id: str) -> dict[str, Any]:
    """会话开始时由前端调用：建立（空的）工作区目录，返回其路径。"""

    ws = _ws(workspace_id)
    return {"workspace_id": ws.workspace_id, "path": str(ws.path)}


@router.get("/workspaces/{workspace_id}/files")
async def list_files(workspace_id: str) -> dict[str, Any]:
    ws = _ws(workspace_id)
    files = [
        {"name": str(p.relative_to(ws.path)), "size": p.stat().st_size}
        for p in sorted(ws.path.rglob("*")) if p.is_file()
    ]
    return {"path": str(ws.path), "files": files}


@router.post("/workspaces/{workspace_id}/files")
async def upload_file(workspace_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """上传一个文件到工作区根目录（JSON：name + content_base64）。"""

    ws = _ws(workspace_id)
    name = str(payload.get("name") or "").strip()
    if not SAFE_NAME_RE.match(name) or name in (".", "..") or name.startswith("."):
        raise HTTPException(status_code=400, detail="文件名不合法（不能含路径分隔符、不能以点开头）")
    try:
        data = base64.b64decode(str(payload.get("content_base64") or ""), validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="content_base64 不是合法的 base64") from exc
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    target = (ws.path / name).resolve()
    if target.parent != ws.path:
        raise HTTPException(status_code=400, detail="文件必须落在工作区根目录")
    target.write_bytes(data)
    return {"name": name, "size": len(data), "path": str(target)}


@router.post("/tasks")
async def submit_task(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """登记一次工具执行；立即返回 task_id，结果经 /tasks/<id>/events 取。"""

    try:
        task = runner.submit(
            workspace_id=str(payload.get("workspace_id") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            tool_name=str(payload.get("name") or ""),
            arguments=payload.get("arguments") or {},
            language=str(payload.get("language") or "zh"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"task_id": task.task_id}


@router.get("/tasks/{task_id}/events")
async def task_events(task_id: str, after: int = 0) -> dict[str, Any]:
    try:
        events, done = runner.events_after(task_id, after)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task 不存在（服务重启后任务表会清空）") from exc
    return {"events": [e.__dict__ for e in events], "done": done}
