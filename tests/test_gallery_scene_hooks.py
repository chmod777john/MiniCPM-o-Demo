"""Gallery 场景契约的两个通用钩子：tools(defaults, presetId) 与 onUnitStarted(event)。"""

from __future__ import annotations

import json

from .conftest import PROJECT_ROOT, needs_node, repo_module_url, run_node


@needs_node
def test_scene_contract_exposes_unit_started_hook() -> None:
    """onUnitStarted 可选，缺省为空操作；场景提供时原样转交。"""

    got = run_node(f"""
const c = await import({json.dumps(repo_module_url('static/demo-gallery/core/scene-contract.js'))});
const seen = [];
const withHook = c.normalizeScene({{ render() {{}}, onUnitStarted(e) {{ seen.push(e.unit_index); }} }}, {{ demoId: 'x', title: 'X' }});
const without = c.normalizeScene({{ render() {{}} }}, {{ demoId: 'y', title: 'Y' }});
withHook.onUnitStarted({{ unit_index: 7 }});
without.onUnitStarted({{ unit_index: 8 }});
console.log(JSON.stringify({{ seen, type: typeof without.onUnitStarted }}));
""")
    assert got == {"seen": [7], "type": "function"}


def test_shell_passes_current_preset_to_scene_tools() -> None:
    """shell 把当前 preset 交给 scene.tools，转交 unit.started，并在切 preset 时刷新 Tools 显示。"""

    app = (PROJECT_ROOT / "static/demo-gallery/shell/app.js").read_text(encoding="utf-8")
    assert "scene.tools(fcBoardDefaults || {}, selectedQuickSetting())" in app
    assert "scene.onUnitStarted(event)" in app
    assert app.count("renderToolsSummary();") >= 2
