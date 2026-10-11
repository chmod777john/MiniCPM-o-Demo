"""Standalone Demo：只注入 standalone 事件 —— preset 是 [统一前缀, 参考音频]，不带任何工具。"""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import needs_node, repo_module_url, run_node


@needs_node
def test_presets_are_prefix_plus_reference_audio_without_tools() -> None:
    """四个 preset：前缀与语言 / 思考模式对应，第二段是存在的参考音频；工具恒为空。"""

    got = run_node(f"""
const qs = await import({json.dumps(repo_module_url('demos/standalone/web/quick-settings.js'))});
const px = await import({json.dumps(repo_module_url('static/demo-gallery/core/system-prefixes.js'))});
const reg = await import({json.dumps(repo_module_url('static/demo-gallery/core/registry.js'))});
await import({json.dumps(repo_module_url('demos/standalone/web/index.js'))});
const map = qs.buildStandaloneQuickSettings({{ non_spoken_budget_while_listening: 45, non_spoken_budget_while_speaking: 25 }});
const stage = {{ innerHTML: '', classList: {{ add() {{}}, remove() {{}} }} }};
const scene = reg.getRegisteredDemo('standalone').createScene({{ stageEl: stage, getClient: () => null }});
const out = {{}};
for (const [id, s] of map) out[id] = {{ kinds: s.segments.map(x => x.kind), prefix: s.segments[0].text,
  ref: s.segments[1].audio.file_path, tts: s.ttsPromptAudioPath, budgets: [s.runtime.nonSpokenBudgetWhileListening, s.runtime.nonSpokenBudgetWhileSpeaking],
  tools: scene.tools({{}}, id) }};
console.log(JSON.stringify({{ out, px: {{ zh_think: px.SYSTEM_PREFIX_ZH_THINK, zh_nothink: px.SYSTEM_PREFIX_ZH_NO_THINK,
  en_general: px.SYSTEM_PREFIX_EN_GENERAL, en_nothink: px.SYSTEM_PREFIX_EN_NO_THINK }} }}));
""")
    assert sorted(got["out"]) == sorted(got["px"])
    for preset_id, row in got["out"].items():
        assert row["kinds"] == ["text", "audio"]
        assert row["prefix"] == got["px"][preset_id]
        assert Path(row["ref"]).is_file(), row["ref"]
        assert row["tts"] == row["ref"]
        assert row["budgets"] == [45, 25]
        assert row["tools"] == []
        assert ("F409" in row["ref"]) == preset_id.startswith("en_")


def test_standalone_demo_does_not_touch_tools_or_sub_agent() -> None:
    """边界：本 Demo 与工具 / 子任务无关，源码里不应出现这些概念。"""

    root = Path(__file__).resolve().parents[1]
    for path in [root / "demo.py", *sorted((root / "web").glob("*.js"))]:
        text = path.read_text(encoding="utf-8")
        for word in ("sub_agent", "subagent", "delegate_task", "sendToolResult"):
            assert word not in text, f"{path.name} 出现了 {word}"
