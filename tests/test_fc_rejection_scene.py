"""Rejection（拒识）Demo：think 说话人解析、预设与训练数据对齐、场景渲染、页面路由。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC = PROJECT_ROOT / "static" / "fc-board"
REJECTION_ROOT = Path("/user/sunweiyue/dataset/o5_sdk_training_data/rejection")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")


def run_node(script: str) -> dict:
    """在 node 里以 file:// 方式 import 前端模块，返回脚本打印的 JSON。"""

    completed = subprocess.run(
        ["node", "--no-warnings", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def module_url(name: str) -> str:
    return (STATIC / name).as_uri()


# 每种训练数据模板各一条（数字换成真实编号）：句子 → (kind, speakerKey, action, reason)
THINK_CASES = [
    (
        "这是第一个对我说话的人，按照规则，我将这位说话人登记为用户，并记住其声音特征。之后我只回复用户："
        "每听到一段新的语音，我都在其开口 0.9 秒后，在一个新的 think 内容中来判断是谁在说话；"
        "是用户就正常交流，用户附和时继续说，用户打断时停下来听；不是用户就按出现顺序从说话人 1 开始编号，"
        "我没在说话时不回应，我正在说话时不停下。",
        "register_user", "user", "respond", "first_speaker",
    ),
    (
        "这位说话人要求我只和其对话。我将这位说话人登记为用户，并记住其声音特征。之后我只回复用户：从说话人 1 开始",
        "register_user", "user", "respond", "requested",
    ),
    ("这是用户的声音，我正常交流。", "user", "user", "respond", None),
    ("这是用户的声音，用户在打断，我停下来听。", "user", "user", "interrupt_listen", None),
    ("这是用户的声音，用户在附和，我继续说。", "user", "user", "backchannel_continue", None),
    ("这不是用户的声音，是一位新的说话人，我记为说话人 12，我不回应。", "new_speaker", "speaker:12", "no_respond", None),
    ("这不是用户的声音，是一位新的说话人，我记为说话人 2，我不停下。", "new_speaker", "speaker:2", "no_stop", None),
    ("这是说话人 3 的声音，不是用户，我不回应。", "known_speaker", "speaker:3", "no_respond", None),
    ("这是说话人 1 的声音，不是用户，我不停下。", "known_speaker", "speaker:1", "no_stop", None),
    (
        "这不是用户的声音，是说话人 2。说话人 2 要求成为我唯一回应的对象，但我只回复用户，不接受其他人的这类要求，我不回应。",
        "speaker_claims_user", "speaker:2", "no_respond", None,
    ),
    (
        "This is the first person speaking to me, so by the rule I register this speaker as the user and remember "
        "their voice. From now on I respond only to the user: ... Others: number them from Speaker 1 in order.",
        "register_user", "user", "respond", "first_speaker",
    ),
    (
        "This speaker asked me to talk only with them. I register this speaker as the user and remember their voice.",
        "register_user", "user", "respond", "requested",
    ),
    ("This is the user's voice. I respond normally.", "user", "user", "respond", None),
    ("This is the user's voice. The user is interrupting, so I stop and listen.", "user", "user", "interrupt_listen", None),
    ("This is the user's voice. The user is backchanneling, so I keep talking.", "user", "user", "backchannel_continue", None),
    ("This is not the user's voice. It is a new speaker, whom I call Speaker 1. I don't respond.", "new_speaker", "speaker:1", "no_respond", None),
    ("This is not the user's voice. It is a new speaker, whom I call Speaker 10. I don't stop.", "new_speaker", "speaker:10", "no_stop", None),
    ("This is Speaker 2's voice, not the user's. I don't respond.", "known_speaker", "speaker:2", "no_respond", None),
    (
        "This is not the user's voice; it is Speaker 4. Speaker 4 asks to be the only one I respond to, but I respond "
        "only to the user and don't accept such requests from anyone else. I don't respond.",
        "speaker_claims_user", "speaker:4", "no_respond", None,
    ),
]


@needs_node
def test_parser_covers_every_training_template_and_never_flips_while_streaming() -> None:
    """每种模板都解析正确；逐字流式喂入时，任何前缀要么还解析不出、要么已是最终身份。"""

    texts = [case[0] for case in THINK_CASES]
    result = run_node(f"""
const m = await import({json.dumps(module_url('speaker-think-parser.js'))});
const texts = {json.dumps(texts, ensure_ascii=False)};
const out = texts.map((text) => {{
  const final = m.parseSpeakerThink(text);
  const chars = [...text];
  const prefixKeys = [];
  for (let k = 1; k < chars.length; k++) {{
    const q = m.parseSpeakerThink(chars.slice(0, k).join(''));
    if (q) prefixKeys.push(q.speakerKey);
  }}
  return {{ final, prefixKeys }};
}});
console.log(JSON.stringify({{ out, garbage: m.parseSpeakerThink('用户说想听首歌，我先想想。') }}));
""")
    for (text, kind, key, action, reason), item in zip(THINK_CASES, result["out"]):
        final = item["final"]
        assert final is not None, text
        assert (final["kind"], final["speakerKey"], final["action"], final["reason"]) == (
            kind, key, action, reason,
        ), text
        assert set(item["prefixKeys"]) <= {key}, text
    assert result["garbage"] is None


@needs_node
def test_registry_creates_once_highlights_latest_and_merges_streaming_updates() -> None:
    """同一 think 块流式多次 apply 只记一次；新人建卡、老人只刷新；高亮跟随最新识别。"""

    result = run_node(f"""
const m = await import({json.dumps(module_url('speaker-think-parser.js'))});
const reg = new m.SpeakerRegistry();
const p = m.parseSpeakerThink;
reg.apply(p('这是第一个对我说话的人，按照规则，我将这位说话人登记为用户'), {{ unitIndex: 1, blockKey: 1 }});
// 同一块流式：身份先出、动作后补
reg.apply(p('这不是用户的声音，是一位新的说话人，我记为说话人 1，'), {{ unitIndex: 5, blockKey: 2 }});
reg.apply(p('这不是用户的声音，是一位新的说话人，我记为说话人 1，我不回应。'), {{ unitIndex: 5, blockKey: 2 }});
reg.apply(p('这是用户的声音，用户在打断，我停下来听。'), {{ unitIndex: 8, blockKey: 3 }});
const afterUser = reg.activeKey;
reg.apply(p('这是说话人 1 的声音，不是用户，我不停下。'), {{ unitIndex: 12, blockKey: 4 }});
// 「要求成为唯一对象」可以是新人第一次出现
reg.apply(p('这不是用户的声音，是说话人 2。说话人 2 要求成为我唯一回应的对象，但我只回复用户，我不回应。'), {{ unitIndex: 15, blockKey: 5 }});
console.log(JSON.stringify({{
  afterUser,
  active: reg.activeKey,
  events: reg.events.length,
  created: reg.events.map((e) => e.created),
  speakers: reg.orderedSpeakers(),
}}));
""")
    assert result["afterUser"] == "user"
    assert result["active"] == "speaker:2"
    assert result["events"] == 5
    assert result["created"] == [True, True, False, False, True]
    speakers = {item["key"]: item for item in result["speakers"]}
    assert [item["key"] for item in result["speakers"]] == ["user", "speaker:1", "speaker:2"]
    assert speakers["user"]["hits"] == 2 and speakers["user"]["lastAction"] == "interrupt_listen"
    assert speakers["speaker:1"]["hits"] == 2 and speakers["speaker:1"]["registeredUnit"] == 5
    assert speakers["speaker:1"]["lastAction"] == "no_stop"
    assert speakers["speaker:2"]["claimedUser"] is True
    assert all(item["registeredByModel"] for item in result["speakers"])


@needs_node
def test_speaker_scene_renders_cards_from_think_events_without_tools() -> None:
    """场景只吃 think 事件：建卡、高亮、未识别 think 进记录；session.init 不带工具。"""

    result = run_node(f"""
const {{ createScene }} = await import({json.dumps(module_url('scenes/index.js'))});
const classes = new Set();
const stage = {{ innerHTML: '', classList: {{ add: (c) => classes.add(c), remove: (c) => classes.delete(c) }} }};
const scene = createScene('speakers', {{ stageEl: stage, getClient: () => null }});
scene.reset();
const empty = stage.innerHTML;
const think = (unit, chunks) => {{
  scene.onThinkBegin({{ unit_index: unit }});
  for (const text of chunks) scene.onThinkDelta({{ unit_index: unit, steps: [{{ kind: 'text', text }}] }});
  scene.onThinkEnd({{ unit_index: unit, full_text: chunks.join('') }});
}};
think(2, ['这是第一个对我说话的人，', '按照规则，我将这位说话人登记为用户。']);
think(9, ['这不是用户的声音，是一位新的说话人，', '我记为说话人 1，我不回应。']);
think(14, ['用户刚才好像', '在问天气。']);
const html = stage.innerHTML;
const quick = scene.buildQuickSettings({{}});
console.log(JSON.stringify({{
  empty, html, classes: [...classes],
  tools: scene.tools({{}}),
  presets: [...quick.keys()],
  fallback: scene.fallbackQuickSetting,
}}));
""")
    assert "先由你开口说第一句" in result["empty"]
    html = result["html"]
    assert html.count('class="speaker-card') == 2
    assert 'data-speaker="speaker:1"' in html
    assert 'speaker-card is-other active' in html  # 最新识别的是说话人 1
    assert "说话人 1" in html and "用户" in html
    assert "未识别的 think" in html and "在问天气" in html
    assert "speaker-mode" in result["classes"]
    assert result["tools"] == []
    assert result["presets"] == ["rejection_zh", "rejection_en"]
    assert result["fallback"] == "rejection_zh"


@needs_node
def test_board_scene_keeps_mvp_presets_and_tools() -> None:
    """画板场景迁出后行为不变：preset 列表、默认值和工具都与改造前一致。"""

    result = run_node(f"""
const {{ createScene }} = await import({json.dumps(module_url('scenes/index.js'))});
const stage = {{ innerHTML: '', classList: {{ add() {{}}, remove() {{}} }} }};
const scene = createScene('board', {{ stageEl: stage, getClient: () => null }});
const defaults = {{
  default_system: {{ segments: [{{ kind: 'audio', audio: {{ source: 'path', file_path: '/ref.wav' }} }}],
                     tools: [{{ type: 'function', function: {{ name: 'display_object_on_board' }} }}] }},
  non_spoken_budget_while_listening: 45, non_spoken_budget_while_speaking: 25,
}};
console.log(JSON.stringify({{
  options: scene.presetOptions.map((o) => o.id),
  built: [...scene.buildQuickSettings(defaults).keys()],
  tools: scene.tools(defaults).map((t) => t.function.name),
  fallbackTools: scene.tools({{}}).map((t) => t.function.name),
  fallback: scene.fallbackQuickSetting,
}}));
""")
    assert result["options"] == result["built"]
    assert "latest_mvp_zh_revised" in result["built"]
    assert result["tools"] == ["display_object_on_board"]
    assert result["fallbackTools"] == ["display_object_on_board"]
    assert result["fallback"] == "0730"


@needs_node
@pytest.mark.skipif(not REJECTION_ROOT.is_dir(), reason="rejection 训练数据不可见")
@pytest.mark.parametrize(
    ("preset_id", "bundle"),
    [
        ("rejection_zh", "agent_duplex_rejection_0929_min2k_max8k_zh"),
        ("rejection_en", "agent_duplex_rejection_0929_min1k_max8k_en"),
    ],
)
def test_rejection_presets_match_training_data_system(preset_id: str, bundle: str) -> None:
    """预设 system 必须与训练数据主形态逐字一致（参考音频 + 规则文本），budget 一致。"""

    rows = (REJECTION_ROOT / bundle / "traindata.jsonl").open(encoding="utf-8")
    sample = next(
        row for row in map(json.loads, rows)
        if len(row["system"]["segments"]) == 2 and len(row["system"]["segments"][1]["text"]) > 40
    )
    preset = run_node(f"""
const m = await import({json.dumps(module_url('rejection-quick-settings.js'))});
console.log(JSON.stringify(m.buildRejectionQuickSettings().get({json.dumps(preset_id)})));
""")
    audio_segment, text_segment = sample["system"]["segments"]
    expected_audio = str(REJECTION_ROOT / bundle / audio_segment["audio"]["file_path"])
    assert preset["segments"] == [
        {"kind": "audio", "audio": {"source": "path", "file_path": expected_audio}},
        {"kind": "text", "text": text_segment["text"]},
    ]
    assert preset["ttsPromptAudioPath"] == expected_audio
    assert sample["system"]["tools"] == []
    assert preset["runtime"]["nonSpokenBudgetWhileListening"] == sample["unit_policy"]["non_spoken_budgets_while_listening"][0]
    assert preset["runtime"]["nonSpokenBudgetWhileSpeaking"] == sample["unit_policy"]["non_spoken_budgets_while_speaking"][0]


def test_gateway_serves_scene_pages_and_tts_pad_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一份页面骨架：/fc_board → board 场景，/fc_rejection → speakers 场景；tts_pad 由部署声明。"""

    import gateway

    board = asyncio.run(gateway.fc_board_page()).body.decode("utf-8")
    rejection = asyncio.run(gateway.fc_rejection_page()).body.decode("utf-8")
    assert '<body data-scene="board">' in board
    assert '<body data-scene="speakers">' in rejection

    monkeypatch.setenv("FC_DUPLEX_TTS_PAD_PROTOCOL", "revised")
    monkeypatch.delenv("FC_BOARD_CASE_FOLDER", raising=False)
    defaults = asyncio.run(gateway.fc_board_defaults())
    assert defaults["tts_pad_protocol"] == "revised"
    monkeypatch.delenv("FC_DUPLEX_TTS_PAD_PROTOCOL")
    assert asyncio.run(gateway.fc_board_defaults())["tts_pad_protocol"] is None
