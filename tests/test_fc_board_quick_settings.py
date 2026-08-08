"""FC Board quick settings 的纯前端模板测试。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_quick_settings_switch_between_0730_and_0807_sysalign() -> None:
    """0730 保持原模板；0807 应变为 prefix→audio→原任务文本。"""

    project_root = Path(__file__).resolve().parents[1]
    module_path = (
        project_root
        / "static"
        / "fc-board"
        / "fc-board-quick-settings.js"
    )
    script = f"""
import fs from 'node:fs';
const source = fs.readFileSync({json.dumps(str(module_path))}, 'utf8');
const url = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const module = await import(url);
const defaults = {{
  segments: [
    {{kind: 'text', text: 'board-task'}},
    {{kind: 'audio', audio: {{source: 'path', file_path: '/ref.wav'}}}},
  ],
  tools: [],
}};
const presets = module.buildBoardQuickSettings(defaults, '/tts.wav');
console.log(JSON.stringify({{
  old: presets.get(module.QUICK_SETTING_0730),
  aligned: presets.get(module.QUICK_SETTING_0807_SYSALIGN),
  prefix: module.SYSALIGN_0807_PREFIX_TEXT,
}}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["old"]["segments"] == defaults_for_assertion()
    assert result["old"]["ttsPromptAudioPath"] == "/tts.wav"
    assert [item["kind"] for item in result["aligned"]["segments"]] == [
        "text",
        "audio",
        "text",
    ]
    assert result["aligned"]["segments"][0]["text"] == result["prefix"]
    assert result["prefix"] == (
        "扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。"
        "请用高自然度的方式和用户聊天。"
        "你处于双工 Agent 模式：可以一边听、一边说；"
        "可以在 non_spoken_slot 中并行地思考与工具调用，"
        "并在 input_event_slot 中接收工具返回结果。"
        "你是由面壁智能开发的人工智能助手：面壁小钢炮。\n"
    )
    assert result["prefix"].endswith("\n")
    assert result["aligned"]["segments"][1]["audio"]["file_path"] == "/ref.wav"
    assert result["aligned"]["segments"][2]["text"] == "board-task"
    assert result["aligned"]["ttsPromptAudioPath"] == "/tts.wav"


def defaults_for_assertion() -> list[dict[str, str | dict[str, str]]]:
    """返回测试输入 segments 的独立期望值。"""

    return [
        {"kind": "text", "text": "board-task"},
        {
            "kind": "audio",
            "audio": {"source": "path", "file_path": "/ref.wav"},
        },
    ]
