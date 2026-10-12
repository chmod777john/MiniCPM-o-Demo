"""API 与 Demo 是两件事：放在一个仓库只是图方便，两侧只通过协议（v4）耦合。

守卫：两侧的 Python 代码互不 import（含函数体里的局部 import）。
- API 侧：Gateway / 控制面 / Worker / 推理后端 / 模型 / 协议客户端。
- Demo 侧：Demo 服务、Gallery、各 Demo。可以用 SDK 的公开类型（minicpm_o5_sdk）与第三方库。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

API_SIDE = ["gateway.py", "worker.py", "worker_state.py", "gateway_modules", "control_plane",
            "py_backend", "core", "modeling", "runtime", "clients"]
DEMO_SIDE = ["demo_server.py", "demo_gallery", "demos"]

API_MODULES = {"gateway", "worker", "worker_state", "gateway_modules", "control_plane", "py_backend",
               "core", "modeling", "runtime", "clients", "config", "scripts"}
DEMO_MODULES = {"demo_server", "demo_gallery", "demos"}


def _python_files(entries: list[str]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        path = ROOT / entry
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in sorted(path.rglob("*.py")) if "/tests/" not in str(p.relative_to(ROOT)) + "/")
    return files


def _imported_top_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", _python_files(API_SIDE), ids=lambda p: str(p.relative_to(ROOT)))
def test_api_side_does_not_import_demo_side(path: Path) -> None:
    leaked = _imported_top_modules(path) & DEMO_MODULES
    assert not leaked, f"{path.relative_to(ROOT)} import 了 Demo 侧模块 {sorted(leaked)}"


@pytest.mark.parametrize("path", _python_files(DEMO_SIDE), ids=lambda p: str(p.relative_to(ROOT)))
def test_demo_side_does_not_import_api_internals(path: Path) -> None:
    leaked = _imported_top_modules(path) & API_MODULES
    assert not leaked, f"{path.relative_to(ROOT)} import 了 API 侧模块 {sorted(leaked)}（Demo 只能通过协议访问 API）"


def test_both_sides_are_scanned() -> None:
    assert len(_python_files(API_SIDE)) > 50 and len(_python_files(DEMO_SIDE)) > 10
