"""Standalone Demo：preset 与训练口径一致；共享层新增的两个通用钩子可用。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demos.standalone.demo import PRESET_BUNDLES, SUB_AGENT_ROOT, standalone_presets
from tests.conftest import PROJECT_ROOT, needs_node, repo_module_url, run_node

pytestmark = pytest.mark.skipif(not SUB_AGENT_ROOT.is_dir(), reason="本机没有 sub_agent 训练数据")


def _first_line_system(bundle: str) -> dict:
    with (SUB_AGENT_ROOT / bundle / "traindata.jsonl").open(encoding="utf-8") as handle:
        return json.loads(handle.readline())["system"]


def test_presets_keep_training_segments_and_tool_order() -> None:
    """[参考音频, per-case 正文] 与工具顺序和训练数据逐项一致（工具顺序在每份数据内固定）。"""

    presets = standalone_presets()
    assert set(presets) == set(PRESET_BUNDLES)
    for preset_id, bundle in PRESET_BUNDLES.items():
        preset = presets[preset_id]
        assert "error" not in preset, preset
        source = _first_line_system(bundle)
        ours = preset["system"]["segments"]
        assert [s["kind"] for s in ours] == [s["kind"] for s in source["segments"]]
        for got, want in zip(ours, source["segments"]):
            if want["kind"] == "text":
                assert got["text"] == want["text"]
            else:
                assert got["audio"]["file_path"] == str(
                    (SUB_AGENT_ROOT / bundle / want["audio"]["file_path"]).resolve()
                )
                assert Path(got["audio"]["file_path"]).is_file()
        assert [t["function"]["name"] for t in preset["system"]["tools"]] == [
            t["function"]["name"] for t in source["tools"]
        ]


@needs_node
def test_quick_settings_prepend_matching_prefix_and_follow_preset_tools() -> None:
    """前端 preset = [对应前缀] + 训练原样段；tools(defaults, presetId) 返回该 preset 自己的工具。"""

    defaults = {
        "standalone_presets": standalone_presets(),
        "non_spoken_budget_while_listening": 45,
        "non_spoken_budget_while_speaking": 25,
    }
    got = run_node(f"""
const qs = await import({json.dumps(repo_module_url('demos/standalone/web/quick-settings.js'))});
const px = await import({json.dumps(repo_module_url('static/demo-gallery/core/system-prefixes.js'))});
const defaults = {json.dumps(defaults, ensure_ascii=False)};
const map = qs.buildStandaloneQuickSettings(defaults);
const out = {{}};
for (const [id, s] of map) out[id] = {{ prefix: s.segments[0].text, rest: s.segments.slice(1),
  budgets: [s.runtime.nonSpokenBudgetWhileListening, s.runtime.nonSpokenBudgetWhileSpeaking],
  tools: qs.presetTools(defaults, id).map(t => t.function.name) }};
console.log(JSON.stringify({{ out, px: {{ zhNo: px.SYSTEM_PREFIX_ZH_NO_THINK, zhThink: px.SYSTEM_PREFIX_ZH_THINK, enNo: px.SYSTEM_PREFIX_EN_NO_THINK }} }}));
""")
    expected_prefix = {
        "subagent_zh_nothink": got["px"]["zhNo"],
        "subagent_zh_think": got["px"]["zhThink"],
        "subagent_en_nothink": got["px"]["enNo"],
    }
    presets = defaults["standalone_presets"]
    for preset_id, row in got["out"].items():
        assert row["prefix"] == expected_prefix[preset_id]
        assert row["rest"] == presets[preset_id]["system"]["segments"]
        assert row["budgets"] == [45, 25]
        assert row["tools"] == [t["function"]["name"] for t in presets[preset_id]["system"]["tools"]]
    assert got["out"]["subagent_zh_think"]["tools"] != got["out"]["subagent_zh_nothink"]["tools"]


@needs_node
def test_scene_contract_exposes_unit_started_hook() -> None:
    """共享层通用钩子：onUnitStarted 可选，缺省为空操作；场景提供时原样转交。"""

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
    """shell 必须把当前 preset 交给 scene.tools，且切 preset 时刷新 Tools 显示。"""

    app = (PROJECT_ROOT / "static/demo-gallery/shell/app.js").read_text(encoding="utf-8")
    assert "scene.tools(fcBoardDefaults || {}, selectedQuickSetting())" in app
    assert "scene.onUnitStarted(event)" in app
    assert app.count("renderToolsSummary();") >= 2
