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


@needs_node
def test_scene_contract_spoken_toolcall_and_prepare_system_hooks() -> None:
    """onSpokenDelta / onSpokenEnd / onToolCallBegin / onToolCallDelta 缺省空操作；
    prepareSystem 缺省原样返回，场景提供时可以改写（含 async）。"""

    got = run_node(f"""
const c = await import({json.dumps(repo_module_url('static/demo-gallery/core/scene-contract.js'))});
const seen = [];
const scene = c.normalizeScene({{
  render() {{}},
  onSpokenDelta(e, t) {{ seen.push(['spoken', t]); }},
  onSpokenEnd(e) {{ seen.push(['end', e.reason]); }},
  onToolCallBegin(e) {{ seen.push(['begin', e.tool_call_id]); }},
  onToolCallDelta(e, t) {{ seen.push(['delta', t]); }},
  async prepareSystem(segs) {{ return segs.map((s) => ({{ ...s, text: s.text + '!' }})); }},
}}, {{ demoId: 'x', title: 'X' }});
const bare = c.normalizeScene({{ render() {{}} }}, {{ demoId: 'y', title: 'Y' }});
scene.onSpokenDelta({{}}, '你好'); scene.onSpokenEnd({{ reason: 'turn_eos' }});
scene.onToolCallBegin({{ tool_call_id: 'c1' }}); scene.onToolCallDelta({{}}, '{{"q"');
bare.onSpokenDelta({{}}, 'x'); bare.onSpokenEnd({{}}); bare.onToolCallBegin({{}}); bare.onToolCallDelta({{}}, 'x');
const segs = [{{ kind: 'text', text: 'a' }}];
console.log(JSON.stringify({{
  seen,
  prepared: await scene.prepareSystem(segs),
  bare: await bare.prepareSystem(segs),
}}));
""")
    assert got == {
        "seen": [["spoken", "你好"], ["end", "turn_eos"], ["begin", "c1"], ["delta", '{"q"']],
        "prepared": [{"kind": "text", "text": "a!"}],
        "bare": [{"kind": "text", "text": "a"}],
    }


def test_shell_forwards_new_hooks() -> None:
    app = (PROJECT_ROOT / "static/demo-gallery/shell/app.js").read_text(encoding="utf-8")
    for call in (
        "scene.onSpokenDelta(event, text)",
        "scene.onSpokenEnd(event)",
        "scene.onToolCallBegin(event)",
        "scene.onToolCallDelta(event, textFromSteps(event.steps))",
        "await scene.prepareSystem(cloneSystemSegments(systemSegments))",
    ):
        assert call in app, call


@needs_node
def test_panel_collapse_state_round_trips() -> None:
    """下方面板折叠状态：只认已知面板 id，非法存储内容视为全部展开。"""

    got = run_node(f"""
const m = await import({json.dumps(repo_module_url('static/demo-gallery/core/panel-collapse.js'))});
const store = new Map();
const storage = {{ getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) }};
let s = m.loadCollapsedPanels(storage);
s = m.toggleCollapsedPanel(s, 'speech');
m.saveCollapsedPanels(storage, s);
const afterOne = [...m.loadCollapsedPanels(storage)];
s = m.toggleCollapsedPanel(m.toggleCollapsedPanel(s, 'ns'), 'speech');
m.saveCollapsedPanels(storage, s);
const afterTwo = [...m.loadCollapsedPanels(storage)];
store.set(m.COLLAPSED_PANELS_KEY, '["bogus","speech"]');
const filtered = [...m.loadCollapsedPanels(storage)];
store.set(m.COLLAPSED_PANELS_KEY, '{{not json');
const broken = [...m.loadCollapsedPanels(storage)];
console.log(JSON.stringify({{ afterOne, afterTwo, filtered, broken, none: [...m.loadCollapsedPanels(null)] }}));
""")
    assert got == {"afterOne": ["speech"], "afterTwo": ["ns"], "filtered": ["speech"], "broken": [], "none": []}


def test_shell_bottom_panels_are_collapsible() -> None:
    html = (PROJECT_ROOT / "demo_gallery/templates/shell.html").read_text(encoding="utf-8")
    for panel in ("speech", "ns"):
        assert f'data-collapsible="{panel}"' in html
        assert f'data-collapse-toggle="{panel}"' in html
    app = (PROJECT_ROOT / "static/demo-gallery/shell/app.js").read_text(encoding="utf-8")
    assert "initPanelCollapse(document.getElementById('colRight'))" in app
