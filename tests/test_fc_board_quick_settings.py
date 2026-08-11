"""FC Board quick settings 的纯前端模板测试。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_quick_settings_cover_family3_and_family4_system_prefixes() -> None:
    """各预设应使用对应 prefix，并保持 Family4 原始 system segment 顺序。"""

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
    {{kind: 'audio', audio: {{source: 'path', file_path: '/ref.wav'}}}},
    {{kind: 'text', text: 'board-task'}},
  ],
  tools: [],
}};
const presets = module.buildBoardQuickSettings(defaults, '/tts.wav', {{
  nonSpokenScheduling: 'quality',
  nonSpokenBudgetWhileListening: 30,
  nonSpokenBudgetWhileSpeaking: 15,
}});
console.log(JSON.stringify({{
  old: presets.get(module.QUICK_SETTING_0730),
  aligned: presets.get(module.QUICK_SETTING_0807_SYSALIGN),
  family4Full: presets.get(module.QUICK_SETTING_0808_FAMILY4_FULL),
  family4Short: presets.get(module.QUICK_SETTING_0808_FAMILY4_SHORT),
  family3Prefix: module.SYSALIGN_0807_PREFIX_TEXT,
  family4FullPrefix: module.FAMILY4_0808_FULL_PREFIX_TEXT,
  family4ShortPrefix: module.FAMILY4_0808_SHORT_PREFIX_TEXT,
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
    assert result["old"]["runtime"] == {
        "nonSpokenScheduling": "quality",
        "nonSpokenBudgetWhileListening": 30,
        "nonSpokenBudgetWhileSpeaking": 15,
    }
    assert [item["kind"] for item in result["aligned"]["segments"]] == [
        "text",
        "audio",
        "text",
    ]
    assert result["aligned"]["segments"][0]["text"] == result["family3Prefix"]
    assert result["family3Prefix"] == (
        "扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。"
        "请用高自然度的方式和用户聊天。"
        "你处于双工 Agent 模式：可以一边听、一边说；"
        "可以在 non_spoken_slot 中并行地思考与工具调用，"
        "并在 input_event_slot 中接收工具返回结果。"
        "你是由面壁智能开发的人工智能助手：面壁小钢炮。\n"
    )
    assert result["family3Prefix"].endswith("\n")
    assert result["aligned"]["segments"][1]["audio"]["file_path"] == "/ref.wav"
    assert result["aligned"]["segments"][2]["text"] == "board-task"
    assert result["aligned"]["ttsPromptAudioPath"] == "/tts.wav"
    assert result["aligned"]["runtime"] == {
        "nonSpokenScheduling": "quality",
        "nonSpokenBudgetWhileListening": 30,
        "nonSpokenBudgetWhileSpeaking": 15,
    }

    expected_full_prefix = (
        "你处于双工 Agent 模式，时间轴按约一秒一个的 unit 组织。"
        "你在 user_audio_slot 中接收当前 unit 的用户语音。"
        "若需要说话，则在 ai_spoken_slot 中作出 speak 决策并生成当前 unit 要说的文本；"
        "否则作出 listen 决策。"
        "你可以在 ai_non_spoken_slot 中生成 think 或 tool_call 内容；"
        "若无额外动作，则作出 no_action 决策。"
        "你在 input_event_slot 中接收工具返回结果或其他独立事件信息。"
        "使用以下音频中的声音说话。\n"
    )
    assert result["family4FullPrefix"] == expected_full_prefix
    assert result["family4ShortPrefix"] == "使用以下音频中的声音说话。\n"

    for preset_name, expected_prefix in (
        ("family4Full", expected_full_prefix),
        ("family4Short", "使用以下音频中的声音说话。\n"),
    ):
        preset = result[preset_name]
        assert preset["segments"][0] == {"kind": "text", "text": expected_prefix}
        assert preset["segments"][1:] == defaults_for_assertion()
        assert preset["ttsPromptAudioPath"] == "/tts.wav"
        assert preset["runtime"] == {
            "nonSpokenScheduling": "quality",
            "nonSpokenBudgetWhileListening": 45,
            "nonSpokenBudgetWhileSpeaking": 25,
        }


def defaults_for_assertion() -> list[dict[str, str | dict[str, str]]]:
    """返回测试输入 segments 的独立期望值。"""

    return [
        {
            "kind": "audio",
            "audio": {"source": "path", "file_path": "/ref.wav"},
        },
        {"kind": "text", "text": "board-task"},
    ]
