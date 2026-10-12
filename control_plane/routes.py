"""控制面板的 HTTP 接口（``/control``，需要密码）与公开的 API 文档页（``/api-docs``）。

密码来自环境变量 ``CONTROL_PANEL_PASSWORD``；**未设置时面板一律 503**（失败即关闭）。
登录成功后发一个签名 cookie（HttpOnly、Secure、只在 /control 下有效，7 天过期）；
签名密钥每次进程启动随机生成，所以控制面重启后需要重新登录。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from control_plane.manager import ControlPlane, ControlPlaneError
from control_plane.specs import WorkerTemplate

STATIC_DIR = Path(__file__).resolve().parent / "static"
API_DOC = Path(__file__).resolve().parents[1] / "docs" / "fc-duplex" / "semantic-realtime-api-v4.md"
COOKIE = "o5_control"
COOKIE_TTL_S = 7 * 24 * 3600
_SIGNING_KEY = secrets.token_bytes(32)

_plane: Optional[ControlPlane] = None


def set_control_plane(plane: Optional[ControlPlane]) -> None:
    global _plane
    _plane = plane


def _password() -> str:
    return os.environ.get("CONTROL_PANEL_PASSWORD") or ""


def _sign(expiry: int) -> str:
    return hmac.new(_SIGNING_KEY, str(expiry).encode(), hashlib.sha256).hexdigest()


def _token_valid(token: Optional[str]) -> bool:
    if not token or "." not in token:
        return False
    expiry, signature = token.split(".", 1)
    return expiry.isdigit() and int(expiry) > time.time() and hmac.compare_digest(signature, _sign(int(expiry)))


def require_login(request: Request) -> ControlPlane:
    if not _password():
        raise HTTPException(status_code=503, detail="控制面板未配置密码（CONTROL_PANEL_PASSWORD），已禁用")
    if not _token_valid(request.cookies.get(COOKIE)):
        raise HTTPException(status_code=401, detail="请先登录")
    if _plane is None:
        raise HTTPException(status_code=503, detail="控制面未启用（Gateway 没有带 --control-plane-root 启动）")
    return _plane


router = APIRouter(prefix="/control")


def _ok(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


async def _call(coro) -> Any:
    try:
        return _ok(await coro)
    except ControlPlaneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def panel_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/static/{name}", include_in_schema=False)
async def panel_static(name: str) -> FileResponse:
    path = (STATIC_DIR / name).resolve()
    if path.parent != STATIC_DIR or not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, headers={"Cache-Control": "no-store"})


class LoginRequest(BaseModel):
    password: str


@router.post("/api/login")
async def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    expected = _password()
    if not expected:
        raise HTTPException(status_code=503, detail="控制面板未配置密码（CONTROL_PANEL_PASSWORD），已禁用")
    if not hmac.compare_digest(payload.password.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="密码错误")
    expiry = int(time.time()) + COOKIE_TTL_S
    response.set_cookie(COOKIE, f"{expiry}.{_sign(expiry)}", max_age=COOKIE_TTL_S, httponly=True,
                        secure=request.url.scheme == "https", samesite="strict", path="/control")
    return {"ok": True}


@router.post("/api/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE, path="/control")
    return {"ok": True}


@router.get("/api/state")
async def state(request: Request, plane: ControlPlane = Depends(require_login)) -> dict:
    base = os.environ.get("API_PUBLIC_BASE_URL") or f"{request.url.scheme}://{request.url.netloc}"
    return {**plane.snapshot(), "public_base": base.rstrip("/")}


class EndpointRequest(BaseModel):
    name: str
    description: str = ""


@router.post("/api/endpoints")
async def create_endpoint(payload: EndpointRequest, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.create_endpoint(payload.name, payload.description))


class EndpointPatch(BaseModel):
    description: str = ""


@router.patch("/api/endpoints/{name}")
async def update_endpoint(name: str, payload: EndpointPatch, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.update_endpoint(name, payload.description))


@router.delete("/api/endpoints/{name}")
async def delete_endpoint(name: str, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.delete_endpoint(name))


@router.put("/api/templates/{name}")
async def put_template(name: str, payload: dict = Body(...), plane: ControlPlane = Depends(require_login)):
    try:
        template = WorkerTemplate.model_validate({**payload, "name": name})
    except ValidationError as exc:
        detail = [{"loc": e["loc"], "msg": f"{'.'.join(map(str, e['loc']))}: {e['msg']}"} for e in exc.errors()]
        raise HTTPException(status_code=422, detail=detail) from exc
    return await _call(plane.put_template(template))


@router.delete("/api/templates/{name}")
async def delete_template(name: str, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.delete_template(name))


class WorkerRequest(BaseModel):
    endpoint: str
    template: str


@router.post("/api/workers")
async def add_worker(payload: WorkerRequest, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.add_worker(payload.endpoint, payload.template))


@router.post("/api/workers/{worker_id}/scale")
async def scale_up(worker_id: str, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.scale_up(worker_id))


@router.post("/api/workers/{worker_id}/drain")
async def drain(worker_id: str, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.drain(worker_id))


@router.post("/api/workers/{worker_id}/force")
async def force_close(worker_id: str, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.force_close(worker_id))


class MoveRequest(BaseModel):
    endpoint: str = Field(min_length=1)


@router.post("/api/workers/{worker_id}/move")
async def move(worker_id: str, payload: MoveRequest, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.move(worker_id, payload.endpoint))


@router.delete("/api/workers/{worker_id}")
async def forget(worker_id: str, plane: ControlPlane = Depends(require_login)):
    return await _call(plane.forget(worker_id))


# ---------------------------------------------------------------- 公开文档

docs_router = APIRouter()

_DOC_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniCPM-o 5 Realtime API v4</title>
<style>
body{max-width:920px;margin:2em auto;padding:0 1em;font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#1f2328}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto;font-size:13px;line-height:1.5}
code{background:#f6f8fa;padding:1px 4px;border-radius:4px;font-size:90%}pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1em 0}th,td{border:1px solid #d0d7de;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f6f8fa}h1,h2,h3{border-bottom:1px solid #eaecef;padding-bottom:.3em}
</style></head><body>{body}</body></html>"""


@docs_router.get("/api-docs", response_class=HTMLResponse)
async def api_docs() -> HTMLResponse:
    from markdown_it import MarkdownIt

    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(API_DOC.read_text(encoding="utf-8"))
    return HTMLResponse(_DOC_PAGE.replace("{body}", body), headers={"Cache-Control": "no-store"})
