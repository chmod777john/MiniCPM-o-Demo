"""守住「加一个 Demo = 只在一个文件夹里开发」这条性质本身。

这些用例跑绿之前，前面的重构都只是「看着像」解耦了。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demo_gallery.mounting import INSTALLED_PATHS, install_gallery
from demo_gallery.registry import registry

from .conftest import PROJECT_ROOT, needs_node, repo_module_url, run_node

FIXTURE_PACKAGE = "tests.fixtures.gallery_demos"

#: 共享层：这里面**不允许**出现任何 demo 的名字。
SHARED_PATHS = ("demo_server.py", "demo_gallery", "static/demo-gallery")


@pytest.fixture
def fixture_app():
    """只装 fixture demo 的干净 app（不碰生产注册表之外的状态）。"""

    saved = registry.all()
    saved_paths = list(INSTALLED_PATHS)
    registry.reset()
    INSTALLED_PATHS.clear()
    app = FastAPI()
    install_gallery(app, package=FIXTURE_PACKAGE)
    try:
        yield app
    finally:
        registry.reset()
        for spec in saved:
            registry.add(spec)
        INSTALLED_PATHS[:] = saved_paths


def test_fixture_demo_works_end_to_end_without_touching_shared_code(fixture_app) -> None:
    """一个全新的 demo，只靠自己文件夹里的代码就有页面 / 资源 / 独立 API。"""

    client = TestClient(fixture_app)

    page = client.get("/demos/hello_gallery")
    assert page.status_code == 200
    assert 'data-demo="hello_gallery"' in page.text
    assert "/demos/hello_gallery/web/index.js" in page.text          # 入口被注入
    assert "/demos/hello_gallery/web/hello_gallery.css" in page.text  # 样式被注入

    assert client.get("/demos/hello_gallery/web/index.js").status_code == 200
    # demo 自己的 API（含工具）挂在自己的前缀下
    assert client.get("/api/demos/hello_gallery/ping").json() == {"ok": True}
    # defaults = Gallery 通用运行时 + demo 专属
    defaults = client.get("/api/demos/hello_gallery/defaults").json()
    assert defaults["hello"] == "world"
    assert "api_endpoints" in defaults          # Gallery 的 Demo 侧配置（模型信息走 API 的 /v1/info）
    # 索引页自动收录，不需要改任何页面
    assert "/demos/hello_gallery" in client.get("/demos").text


def test_no_demo_path_escapes_to_the_main_router(fixture_app) -> None:
    """demo 贡献的每一条路径都必须落在 Gallery 的两个前缀下。

    用户明确要求：demo 的工具接口不能走主路由。
    """

    assert INSTALLED_PATHS, "install_gallery 应记录自己新增的路径"
    for _kind, path in INSTALLED_PATHS:
        assert path.startswith(("/demos", "/api/demos")), path


def test_shared_code_mentions_no_demo_identifier() -> None:
    """共享层不得出现任何 demo 的名字 —— 这是「加 demo 不改共享文件」的直接编码。"""

    forbidden = [
        "fc_board", "fc-board", "fc_rejection", "hello_gallery",
        "display_object_on_board", "speaker-", "board-scene", "live-image-downloads",
    ]
    hits = subprocess.run(
        ["git", "grep", "-nIF", *sum([["-e", word] for word in forbidden], []), "--", *SHARED_PATHS],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    assert hits == "", f"共享层泄漏了 demo 名:\n{hits}"


def test_every_demo_is_self_contained_on_disk() -> None:
    """每个 demo 的文件夹必须自带注册入口与前端入口。"""

    for spec in registry.all():
        assert (spec.package_dir / "demo.py").is_file(), spec.demo_id
        assert (spec.package_dir / spec.entry).is_file(), spec.demo_id
        assert spec.demo_id == spec.package_dir.name


@needs_node
def test_python_demo_id_matches_js_registered_id() -> None:
    """目录名 / Python demo_id / JS 注册 id 三者必须完全一致。

    它们同时也是 URL 段 —— 三者不一致过（旧路由叫 /fc_rejection、场景 id 叫 speakers）。
    """

    for spec in registry.all():
        got = run_node(f"""
await import({json.dumps((spec.package_dir / spec.entry).as_uri())});
const m = await import({json.dumps(repo_module_url('static/demo-gallery/core/registry.js'))});
console.log(JSON.stringify(m.registeredDemoIds()));
""")
        assert got == [spec.demo_id], f"{spec.demo_id} 的入口注册了 {got}"


@needs_node
def test_every_demo_scene_satisfies_the_contract() -> None:
    """每个 demo 的 scene 都能通过契约校验（必填项齐、可选项可缺）。"""

    for spec in registry.all():
        got = run_node(f"""
await import({json.dumps((spec.package_dir / spec.entry).as_uri())});
const reg = await import({json.dumps(repo_module_url('static/demo-gallery/core/registry.js'))});
const contract = await import({json.dumps(repo_module_url('static/demo-gallery/core/scene-contract.js'))});
const stage = {{ innerHTML: '', classList: {{ add() {{}}, remove() {{}} }} }};
const scene = reg.getRegisteredDemo({json.dumps(spec.demo_id)}).createScene({{
  demoId: {json.dumps(spec.demo_id)}, stageEl: stage, getClient: () => null,
  api: {{ base: '', url: (p) => p }}, assets: {{ base: '', url: (p) => p }}, shell: {{}},
}});
console.log(JSON.stringify({{ ok: contract.validateScene(scene, {json.dumps(spec.demo_id)}) }}));
""")
        assert got["ok"] is True, spec.demo_id


def test_every_demo_css_is_namespaced() -> None:
    """demo CSS 的每条顶层规则必须以 [data-demo="<自己的 id>"] 开头。

    否则会污染 shell 与别的 demo —— 重构前 `.board` 既是通用容器又是画板，
    rejection 只能靠 `.board.speaker-mode` 打补丁。
    """

    for spec in registry.all():
        for style in spec.styles:
            css = (spec.package_dir / "web" / style).read_text(encoding="utf-8")
            css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
            for selector in _top_level_selectors(css):
                if selector.startswith("@keyframes"):
                    name = selector.split(None, 1)[1].strip()
                    assert name.startswith(f"{spec.demo_id}-"), (
                        f"{spec.demo_id}/{style}: @keyframes {name} 必须以 {spec.demo_id}- 开头"
                    )
                    continue
                if selector.startswith("@"):
                    continue  # @media 等，内部规则由下面的递归覆盖
                for one in selector.split(","):
                    one = one.strip()
                    if one:
                        assert one.startswith(f'[data-demo="{spec.demo_id}"]'), (
                            f"{spec.demo_id}/{style}: 选择器 `{one}` 没有命名空间"
                        )


def _top_level_selectors(css: str) -> list[str]:
    """抠出顶层（含 @media 内第一层）的选择器文本。

    `@keyframes` 内部是关键帧选择器（0% / from / to），不是 CSS 选择器，跳过。
    """

    selectors, depth, buffer, in_keyframes = [], 0, "", None
    for char in css:
        if char == "{":
            text = buffer.strip()
            if depth == 0:
                selectors.append(text)
                if text.startswith("@keyframes"):
                    in_keyframes = depth
            elif depth == 1 and text and in_keyframes is None:
                selectors.append(text)
            depth += 1
            buffer = ""
        elif char == "}":
            depth -= 1
            if in_keyframes is not None and depth <= in_keyframes:
                in_keyframes = None
            buffer = ""
        else:
            buffer += char
    return [s for s in selectors if s]


@needs_node
def test_shell_tolerates_a_demo_without_any_preset() -> None:
    """没有任何 quick setting 的 demo 是合法的，shell 不能因此崩。

    脚手架生成的 demo 就没有 preset —— 验收时真的崩过
    （`Unknown quick setting: null`：拿 fallback=null 去查空表）。
    """

    got = run_node(f"""
const contract = await import({json.dumps(repo_module_url('static/demo-gallery/core/scene-contract.js'))});
const scene = contract.normalizeScene({{ render() {{}} }}, {{ demoId: 'x', title: 'X' }});
console.log(JSON.stringify({{
  fallback: scene.fallbackQuickSetting,
  presets: scene.presetOptions,
  built: [...scene.buildQuickSettings({{}}).keys()],
  tools: scene.tools({{}}),
}}));
""")
    assert got["fallback"] is None      # 没有 preset 时 fallback 必须是 null，不是空串
    assert got["presets"] == [] and got["built"] == [] and got["tools"] == []

    # shell 必须显式处理这种情况，而不是无条件 applyQuickSetting。
    app_js = (PROJECT_ROOT / "static/demo-gallery/shell/app.js").read_text(encoding="utf-8")
    assert "quickSettings.has(initialSetting)" in app_js


def test_copying_the_template_yields_a_working_demo(tmp_path, monkeypatch) -> None:
    """「复制 _template、改三处名字」就是一个能用的新 demo —— 规范文档承诺的流程本身。

    脚手架没人日常运行，改坏了不会有人发现；这条用例按 demos/README.md 的步骤
    逐字做一遍，确保那几步永远是对的。
    """

    import shutil
    import sys

    new_id = "scaffold_copy"  # 不能含 "_template" 子串，否则下面的替换会二次命中
    package = "scaffold_probe_pkg"
    target = tmp_path / package / new_id
    shutil.copytree(PROJECT_ROOT / "demos" / "_template", target,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    # README 的三步：改 id、改 import 路径、改 CSS 文件名与命名空间
    for path in [*target.rglob("*.py"), *target.rglob("*.js"), *target.rglob("*.css")]:
        text = path.read_text(encoding="utf-8")
        text = text.replace("demos._template", f"{package}.{new_id}").replace("_template", new_id)
        path.write_text(text, encoding="utf-8")
    (target / "web" / "_template.css").rename(target / "web" / f"{new_id}.css")

    monkeypatch.syspath_prepend(str(tmp_path))
    saved, saved_paths = registry.all(), list(INSTALLED_PATHS)
    registry.reset()
    INSTALLED_PATHS.clear()
    try:
        app = FastAPI()
        specs = install_gallery(app, package=package)
        assert [s.demo_id for s in specs] == [new_id]

        client = TestClient(app)
        page = client.get(f"/demos/{new_id}")
        assert page.status_code == 200
        assert f'data-demo="{new_id}"' in page.text
        assert f"/demos/{new_id}/web/{new_id}.css" in page.text
        for asset in ("index.js", "scene.js", f"{new_id}.css"):
            assert client.get(f"/demos/{new_id}/web/{asset}").status_code == 200, asset
        assert client.get(f"/api/demos/{new_id}/example").json() == {"ok": True}
        assert "api_endpoints" in client.get(f"/api/demos/{new_id}/defaults").json()
        assert f"/demos/{new_id}" in client.get("/demos").text
    finally:
        registry.reset()
        for spec in saved:
            registry.add(spec)
        INSTALLED_PATHS[:] = saved_paths
        for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
            del sys.modules[name]


def test_template_is_hidden_from_the_gallery() -> None:
    """_template 只是脚手架，不能出现在生产索引页里。"""

    assert "_template" not in [spec.demo_id for spec in registry.all()]


@needs_node
def test_template_scene_satisfies_the_contract() -> None:
    """脚手架的前端入口本身必须满足场景契约（复制出来的 demo 才能直接跑）。"""

    entry = PROJECT_ROOT / "demos" / "_template" / "web" / "index.js"
    got = run_node(f"""
await import({json.dumps(entry.as_uri())});
const reg = await import({json.dumps(repo_module_url('static/demo-gallery/core/registry.js'))});
const contract = await import({json.dumps(repo_module_url('static/demo-gallery/core/scene-contract.js'))});
const stage = {{ innerHTML: '', classList: {{ add() {{}}, remove() {{}} }} }};
const scene = reg.getRegisteredDemo('_template').createScene({{
  demoId: '_template', stageEl: stage, getClient: () => null,
  api: {{ base: '', url: (p) => p }}, assets: {{ base: '', url: (p) => p }}, shell: {{}},
}});
scene.render();
const empty = stage.innerHTML;
scene.onThinkEnd();
console.log(JSON.stringify({{ ok: contract.validateScene(scene, '_template'), empty, after: stage.innerHTML }}));
""")
    assert got["ok"] is True
    assert "empty-state" in got["empty"]          # 首帧自己画空状态
    assert "1" in got["after"]                     # 事件钩子真的接上了
