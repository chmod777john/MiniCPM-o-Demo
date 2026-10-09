"""Demo Gallery 注册中心。

每个 demo 只做一件事：import 本模块的 ``@demo`` 装饰器并装饰自己的 Demo 类。

**本文件里不允许出现任何具体 demo 的名字** —— 这是「加一个 demo 不改任何共享文件」
这条性质的直接编码，有守卫测试（``tests/test_gallery_pluggability.py``）盯着。
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import APIRouter


@dataclass(frozen=True)
class StaticMount:
    """Demo 想额外暴露的静态目录（相对 demo 包目录）。

    URL 与磁盘路径同构：``demos/<id>/<path>`` → ``/demos/<id>/<path>``。
    """

    path: str
    create: bool = False
    """目录不存在时是否 mkdir —— 下载落地目录首次启动时还不存在。"""


@dataclass(frozen=True)
class DemoSpec:
    """一个 Demo 注册进 Gallery 的全部信息。"""

    demo_id: str
    """必须等于 ``demos/<这个名字>/`` 目录名，也等于前端 ``registerDemo`` 的 id。"""

    title: str
    summary: str = ""
    order: int = 100

    # —— 前端 ——
    entry: str = "web/index.js"
    """相对 demo 包目录；shell 用 dynamic import() 加载它，import 的副作用即自注册。"""

    styles: Sequence[str] = ()
    """相对 ``web/``；shell 注入 <link>，排在共享 shell.css 之后。"""

    # —— 后端 ——
    router: APIRouter | None = None
    """Demo 自己的 API（含工具接口）。由 core 挂到 ``/api/demos/<id>``，不碰主路由。"""

    extra_mounts: Sequence[StaticMount] = ()

    # —— 由 core 回填 ——
    package: str = ""
    package_dir: Path = field(default=Path("."))
    defaults_provider: Callable[[], dict[str, Any]] | None = None

    @property
    def api_prefix(self) -> str:
        return f"/api/demos/{self.demo_id}"

    @property
    def page_url(self) -> str:
        return f"/demos/{self.demo_id}"

    @property
    def assets_url(self) -> str:
        return f"/demos/{self.demo_id}/web"

    def mount_url(self, relative: str) -> str:
        return f"/demos/{self.demo_id}/{relative.strip('/')}"

    @property
    def entry_url(self) -> str:
        return f"{self.page_url}/{self.entry.lstrip('/')}"


class DemoRegistry:
    """线程安全的内存注册表。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._specs: dict[str, DemoSpec] = {}

    def add(self, spec: DemoSpec) -> None:
        with self._lock:
            existing = self._specs.get(spec.demo_id)
            if existing is not None:
                raise ValueError(
                    f"demo id 重复: {spec.demo_id}（已由 {existing.package} 注册）"
                )
            self._specs[spec.demo_id] = spec

    def all(self) -> list[DemoSpec]:
        with self._lock:
            return sorted(self._specs.values(), key=lambda s: (s.order, s.demo_id))

    def get(self, demo_id: str) -> DemoSpec | None:
        with self._lock:
            return self._specs.get(demo_id)

    def reset(self) -> None:
        """仅供测试：清空注册表。"""
        with self._lock:
            self._specs.clear()


registry = DemoRegistry()


def demo(**spec_kwargs: Any) -> Callable[[type], type]:
    """把一个 Demo 类注册进 Gallery。

    参数:
        **spec_kwargs: ``DemoSpec`` 的字段（``demo_id`` / ``title`` / ``router`` …）。

    返回:
        原样返回被装饰的类（装饰器只做注册，不包装行为）。

    异常:
        ValueError: id 与目录名不符、入口/样式文件不存在、或 id 重复。
    """

    def decorate(cls: type) -> type:
        package_dir = Path(inspect.getfile(cls)).resolve().parent
        instance = cls()
        spec = DemoSpec(
            **spec_kwargs,
            package=cls.__module__,
            package_dir=package_dir,
            defaults_provider=getattr(instance, "defaults", None),
        )
        _validate(spec)
        registry.add(spec)
        cls.__gallery_spec__ = spec  # type: ignore[attr-defined]
        return cls

    return decorate


def _validate(spec: DemoSpec) -> None:
    if spec.demo_id != spec.package_dir.name:
        raise ValueError(
            f"demo_id({spec.demo_id}) 必须等于目录名({spec.package_dir.name})"
            " —— URL 与磁盘路径同构，是前端相对 import 能同时在浏览器和 node 里工作的前提"
        )
    if not (spec.package_dir / spec.entry).is_file():
        raise ValueError(f"{spec.demo_id}: 入口 {spec.entry} 不存在")
    for style in spec.styles:
        if not (spec.package_dir / "web" / style).is_file():
            raise ValueError(f"{spec.demo_id}: 样式 web/{style} 不存在")
