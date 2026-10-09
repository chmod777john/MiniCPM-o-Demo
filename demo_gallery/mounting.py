"""把注册表接到 FastAPI：页面路由 / 静态资源 / per-demo API 前缀。

**Demo 贡献的每一条路径都落在 ``/demos/`` 或 ``/api/demos/`` 下**，绝不挂到主路由上
（有守卫测试 ``test_no_demo_api_escapes_to_the_main_router`` 盯着）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from demo_gallery.deployment import gallery_runtime
from demo_gallery.discovery import discover_demos
from demo_gallery.page import render_index, render_shell
from demo_gallery.registry import DemoSpec, registry

logger = logging.getLogger("demo_gallery")

#: install_gallery 新增的每条路径，(kind, path)。守卫测试用它断言没有路径逃逸到主路由。
INSTALLED_PATHS: list[tuple[str, str]] = []


def install_gallery(app: FastAPI, *, package: str = "demos") -> list[DemoSpec]:
    """发现并挂载全部 Demo。

    参数:
        app: FastAPI 应用。
        package: demo 所在包名（测试传 fixture 包）。

    返回:
        已挂载的 Demo 列表（按 order 排序）。
    """

    discover_demos(package)
    specs = registry.all()

    gallery_api = APIRouter(prefix="/api/demos")

    @gallery_api.get("")
    async def list_demos() -> dict[str, Any]:
        """Demo 列表（索引页与外部集成用）。"""
        return {
            "demos": [
                {
                    "demo_id": spec.demo_id,
                    "title": spec.title,
                    "summary": spec.summary,
                    "page": spec.page_url,
                }
                for spec in specs
            ]
        }

    @gallery_api.get("/runtime")
    async def runtime() -> dict[str, Any]:
        """Gallery 级通用部署信息（与具体 demo 无关）。"""
        return gallery_runtime()

    app.include_router(gallery_api)
    INSTALLED_PATHS.append(("api", "/api/demos"))

    @app.get("/demos", response_class=HTMLResponse)
    async def gallery_index() -> HTMLResponse:
        """Gallery 索引页。"""
        return HTMLResponse(render_index(specs))

    INSTALLED_PATHS.append(("page", "/demos"))

    for spec in specs:
        _install_one(app, spec)

    logger.info("[gallery] 已挂载 %d 个 demo", len(specs))
    return specs


def _page_handler(spec: DemoSpec) -> Callable[[], Any]:
    """用闭包捕获 spec。

    ⚠️ 不能写成 ``async def page(_spec: DemoSpec = spec)`` —— FastAPI 会把带默认值的
    参数当成请求参数去解析，直接报错。
    """

    async def page() -> HTMLResponse:
        return HTMLResponse(render_shell(spec))

    return page


def _defaults_handler(spec: DemoSpec) -> Callable[[], Any]:
    async def defaults() -> dict[str, Any]:
        """Gallery 通用运行时 + 该 demo 自己的 defaults。"""
        payload = gallery_runtime()
        if spec.defaults_provider is not None:
            payload.update(spec.defaults_provider())
        return payload

    return defaults


def _install_one(app: FastAPI, spec: DemoSpec) -> None:
    # 1) 页面
    app.add_api_route(
        spec.page_url,
        _page_handler(spec),
        methods=["GET"],
        response_class=HTMLResponse,
        include_in_schema=False,
        name=f"demo_page_{spec.demo_id}",
    )
    INSTALLED_PATHS.append(("page", spec.page_url))

    # 2) 前端资源：URL 与磁盘同构，demo 的 JS 才能用相对路径 import 共享 core
    web_dir = spec.package_dir / "web"
    if web_dir.is_dir():
        app.mount(
            spec.assets_url,
            StaticFiles(directory=str(web_dir)),
            name=f"demo_web_{spec.demo_id}",
        )
        INSTALLED_PATHS.append(("mount", spec.assets_url))

    # 3) demo 声明的额外静态目录
    for mount in spec.extra_mounts:
        directory = spec.package_dir / mount.path
        if mount.create:
            directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            logger.warning(
                "[gallery] %s 的静态目录不存在，跳过: %s", spec.demo_id, directory
            )
            continue
        url = spec.mount_url(mount.path)
        app.mount(
            url,
            StaticFiles(directory=str(directory)),
            name=f"demo_mount_{spec.demo_id}_{mount.path.replace('/', '_')}",
        )
        INSTALLED_PATHS.append(("mount", url))

    # 4) per-demo API 前缀 —— 工具接口只能在这里
    demo_api = APIRouter(prefix=spec.api_prefix, tags=[f"demo:{spec.demo_id}"])
    demo_api.add_api_route(
        "/defaults", _defaults_handler(spec), methods=["GET"],
        name=f"demo_defaults_{spec.demo_id}",
    )
    if spec.router is not None:
        demo_api.include_router(spec.router)
    app.include_router(demo_api)
    INSTALLED_PATHS.append(("api", spec.api_prefix))
