"""Rejection（拒识）Demo：think 说话人解析、预设与训练数据对齐、场景渲染、页面路由。"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE = PROJECT_ROOT / "static" / "demo-gallery" / "core"
SHELL = PROJECT_ROOT / "static" / "demo-gallery" / "shell"
BOARD_WEB = PROJECT_ROOT / "demos" / "board" / "web"
REJECTION_WEB = PROJECT_ROOT / "demos" / "rejection" / "web"
SHELL_HTML = PROJECT_ROOT / "demo_gallery" / "templates" / "shell.html"
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
    """共享 core 模块的 file:// URL。"""
    return (CORE / name).as_uri()


def demo_module_url(demo: str, name: str) -> str:
    return (PROJECT_ROOT / "demos" / demo / "web" / name).as_uri()


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
    # 真实模型的换说法（0930 mergeAll iter_500，2026-10-09 浏览器端到端实测），与训练模板不逐字一致。
    (
        "这是用户第一次对我说话，按照规则，我将这段内容登记为用户，并记住其语音特征。之后我只回复用户："
        "不是用户就按出现顺序从说话人 1 开始编号。",
        "register_user", "user", "respond", "first_speaker",
    ),
    ("这是用户的语音，我正常交流。", "user", "user", "respond", None),
    (
        "这不是用户的语音，是说话人 1。说话人 1 要求成为我唯一回应的对象，但只有用户能设定这个规则，"
        "所以我拒绝他的要求，仍然只回复用户。",
        "speaker_claims_user", "speaker:1", None, None,
    ),
    ("这不是用户的语音，是一位新的说话人，我记为说话人 2，我不停下。", "new_speaker", "speaker:2", "no_stop", None),
    (
        "This is the first person speaking to me, so by the rule, this user is the one I should respond to. "
        "I need to register them as the user and remember their voice.",
        "register_user", "user", "respond", "first_speaker",
    ),
    # 不是用户但没编号：只在整块结束时判成「其他人（未编号）」，流式前缀一律不判。
    (
        "This is not the user's voice. It is a child's voice, so mark it as [Child's voice] and do not respond.",
        "other_unnumbered", "other", "no_respond", None,
    ),
    ("This is the user's voice. Respond normally.", "user", "user", "respond", None),
]


@needs_node
def test_parser_covers_every_training_template_and_never_flips_while_streaming() -> None:
    """每种模板都解析正确；逐字流式喂入时，任何前缀要么还解析不出、要么已是最终身份。"""

    texts = [case[0] for case in THINK_CASES]
    result = run_node(f"""
const m = await import({json.dumps(demo_module_url('rejection', 'think-parser.js'))});
const texts = {json.dumps(texts, ensure_ascii=False)};
const out = texts.map((text) => {{
  const final = m.parseSpeakerThink(text, {{ final: true }});
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
const m = await import({json.dumps(demo_module_url('rejection', 'think-parser.js'))});
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
    """走真实入口：import demos/rejection/web/index.js 的副作用就是自注册。

    场景只吃 think 事件：建卡、高亮、未识别 think 进记录；session.init 不带工具。
    """

    result = run_node(f"""
await import({json.dumps(demo_module_url('rejection', 'index.js'))});
const reg = await import({json.dumps(module_url('registry.js'))});
const ids = reg.registeredDemoIds();
const classes = new Set();
const stage = {{ innerHTML: '', classList: {{ add: (c) => classes.add(c), remove: (c) => classes.delete(c) }} }};
const scene = reg.getRegisteredDemo('rejection').createScene({{ stageEl: stage, getClient: () => null }});
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
console.log(JSON.stringify({{
  ids, empty, html: stage.innerHTML, classes: [...classes],
  tools: scene.tools({{}}),
  presets: [...scene.buildQuickSettings({{}}).keys()],
  fallback: scene.fallbackQuickSetting,
}}));
""")
    assert result["ids"] == ["rejection"], "入口模块必须且只注册自己"
    assert "先由你开口说第一句" in result["empty"]
    html = result["html"]
    assert html.count('class="speaker-card') == 2
    assert 'data-speaker="speaker:1"' in html
    assert "speaker-card is-other active" in html  # 最新识别的是说话人 1
    assert "说话人 1" in html and "用户" in html
    assert "未识别的 think" in html and "在问天气" in html
    # 场景不再靠改 stage 的 class 解除骨架布局（speaker-mode 补丁已废），只用通用的 is-empty。
    assert "speaker-mode" not in result["classes"]
    assert result["tools"] == []
    assert result["presets"] == ["rejection_zh", "rejection_en"]
    assert result["fallback"] == "rejection_zh"


@needs_node
def test_board_scene_keeps_mvp_presets_and_tools() -> None:
    """画板场景迁进 demo 文件夹后行为不变：preset 列表、默认值和工具都与改造前一致。"""

    result = run_node(f"""
await import({json.dumps(demo_module_url('board', 'index.js'))});
const reg = await import({json.dumps(module_url('registry.js'))});
const stage = {{ innerHTML: '', classList: {{ add() {{}}, remove() {{}} }} }};
const scene = reg.getRegisteredDemo('board').createScene({{
  stageEl: stage, getClient: () => null,
  api: {{ base: '/api/demos/board', url: (p) => `/api/demos/board/${{p}}` }},
}});
const defaults = {{
  default_system: {{ segments: [{{ kind: 'audio', audio: {{ source: 'path', file_path: '/ref.wav' }} }}],
                     tools: [{{ type: 'function', function: {{ name: 'display_object_on_board' }} }}] }},
  non_spoken_budget_while_listening: 45, non_spoken_budget_while_speaking: 25,
}};
console.log(JSON.stringify({{
  ids: reg.registeredDemoIds(),
  options: scene.presetOptions.map((o) => o.id),
  built: [...scene.buildQuickSettings(defaults).keys()],
  tools: scene.tools(defaults).map((t) => t.function.name),
  fallbackTools: scene.tools({{}}).map((t) => t.function.name),
  fallback: scene.fallbackQuickSetting,
}}));
""")
    assert result["ids"] == ["board"]
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
const m = await import({json.dumps(demo_module_url('rejection', 'quick-settings.js'))});
console.log(JSON.stringify(m.buildRejectionQuickSettings().get({json.dumps(preset_id)})));
""")
    audio_segment, text_segment = sample["system"]["segments"]
    expected_audio = str(REJECTION_ROOT / bundle / audio_segment["audio"]["file_path"])
    # 盘上 bundle 只有 [参考音频, per-case 正文]；第一段统一前缀由训练侧注入，推理侧必须自己补。
    export_name = {
        "rejection_zh": "SYSTEM_PREFIX_ZH_THINK",
        "rejection_en": "SYSTEM_PREFIX_EN_GENERAL",
    }[preset_id]
    prefix = run_node(f"""
const m = await import({json.dumps(module_url('system-prefixes.js'))});
console.log(JSON.stringify(m[{json.dumps(export_name)}]));
""")
    assert preset["segments"] == [
        {"kind": "text", "text": prefix},
        {"kind": "audio", "audio": {"source": "path", "file_path": expected_audio}},
        {"kind": "text", "text": text_segment["text"]},
    ]
    assert preset["ttsPromptAudioPath"] == expected_audio
    assert sample["system"]["tools"] == []
    assert preset["runtime"]["nonSpokenBudgetWhileListening"] == sample["unit_policy"]["non_spoken_budgets_while_listening"][0]
    assert preset["runtime"]["nonSpokenBudgetWhileSpeaking"] == sample["unit_policy"]["non_spoken_budgets_while_speaking"][0]


def test_gallery_serves_demo_pages_with_injected_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每个 demo 页面都由同一份骨架渲染，注入自己的 data-demo / 入口 URL / 样式。"""

    import gateway
    from fastapi.testclient import TestClient

    client = TestClient(gateway.app)
    for demo_id in ("board", "rejection"):
        page = client.get(f"/demos/{demo_id}").text
        assert f'<body data-demo="{demo_id}">' in page
        assert f'/demos/{demo_id}/web/index.js' in page      # 入口被注入
        assert f'/demos/{demo_id}/web/{demo_id}.css' in page  # 样式被注入
        assert "gallery-bootstrap" in page
    # 索引页自动列出全部 demo，新增 demo 不需要改任何页面。
    index = client.get("/demos").text
    assert "/demos/board" in index and "/demos/rejection" in index

    monkeypatch.setenv("FC_DUPLEX_TTS_PAD_PROTOCOL", "revised")
    from demo_gallery.deployment import gallery_runtime

    assert gallery_runtime()["tts_pad_protocol"] == "revised"
    monkeypatch.delenv("FC_DUPLEX_TTS_PAD_PROTOCOL")
    assert gallery_runtime()["tts_pad_protocol"] is None


@needs_node
def test_playback_buffer_default_and_override() -> None:
    """首包缓冲：默认 600ms（实测 200ms 吸收不住单 Unit 抖动），非法输入一律回退、绝不返回 NaN。

    直接 import `playback-policy.js` 的公开导出；不再从 app.js 正则抠函数体
    —— 那种写法对函数名/缩进/字面量格式有隐式要求，文件一动就误报。
    """

    got = run_node(f"""
const m = await import({json.dumps(module_url('playback-policy.js'))});
console.log(JSON.stringify({{
  dflt: m.DEFAULT_PLAYBACK_DELAY_MS,
  ok: m.resolvePlaybackDelayMs('250'),
  empty: m.resolvePlaybackDelayMs(''),
  bad: m.resolvePlaybackDelayMs('abc'),
  negative: m.resolvePlaybackDelayMs('-5'),
  nullish: m.resolvePlaybackDelayMs(undefined),
  zero: m.resolvePlaybackDelayMs('0'),
  custom: m.resolvePlaybackDelayMs('nope', 123),
}}));
""")
    assert got == {
        "dflt": 600, "ok": 250, "empty": 600, "bad": 600,
        "negative": 600, "nullish": 600, "zero": 0, "custom": 123,
    }

    # HTML 里 input 的默认值必须与模块常量一致（两处各写 600 → 升级成两处必须相等）。
    html = (SHELL_HTML).read_text(encoding="utf-8")
    shown = re.search(r'id="playbackDelayMs"[^>]*value="(\d+)"', html)
    assert shown and int(shown.group(1)) == got["dflt"]


@needs_node
def test_opening_silence_warning_threshold_matches_training_data() -> None:
    """开场静音提示：阈值对齐训练数据（最长 5s 开口），且只在开麦后无人声时出现。

    `openingSilenceWarning` 是纯函数、不碰 DOM，所以这里直接调它，不再抠源码。
    """

    got = run_node(f"""
const m = await import({json.dumps(module_url('mic-policy.js'))});
const call = (micLiveState, speechSeen, waitedMs) =>
  m.openingSilenceWarning({{ micLiveState, speechSeen, waitedMs, peakDb: -48 }});
console.log(JSON.stringify({{
  limit: m.MIC_OPENING_SILENCE_LIMIT_MS,
  speechDb: m.MIC_SPEECH_DB,
  quiet9s: call('live', false, 9000),
  quiet3s: call('live', false, 3000).visible,
  atLimit: call('live', false, 5000).visible,
  spoke9s: call('live', true, 9000).visible,
  idle9s: call('idle', false, 9000).visible,
}}));
""")
    assert got["limit"] == 5000 and got["speechDb"] == -35
    assert got["quiet9s"]["visible"] is True
    assert "9s" in got["quiet9s"]["text"] and "-48" in got["quiet9s"]["text"]
    assert got["quiet3s"] is False      # 还没到 5s
    assert got["atLimit"] is False      # 必须「超过」才提示，等于不提示
    assert got["spoke9s"] is False      # 已经检测到人声
    assert got["idle9s"] is False       # 没开麦
    assert '<div id="micSilenceWarn"' in (SHELL_HTML).read_text(encoding="utf-8")


@needs_node
def test_unified_system_prefixes_match_the_spec_verbatim() -> None:
    """统一前缀必须与飞书《O5 agentduplex 中统一系统提示词前缀》逐字一致（含结尾换行）。

    来源 https://modelbest.feishu.cn/wiki/Bhsow0q70iV4vokfEMtcIG4wnKb
    #1 中文 think 通用 / #3 英文通用（rejection 的 think 比例 1.000，英文取通用版）。
    """

    expected_zh = "你处于双工 Agent 模式，时间轴按一秒一个的 unit 组织。你在 user_audio_slot 中接收当前 unit 的用户语音。若你需要说话，则在 ai_spoken_slot 中作出 speak 决策并生成当前 unit 要说的文本；否则作出 listen 决策。你可以在 ai_non_spoken_slot 中生成 think 或 tool_call 内容；若无额外动作，则作出 no_action 决策。你在 input_event_slot 中接收工具返回结果或其他独立事件信息。使用以下音频中的声音说话。\n"
    expected_en = "You are operating in duplex Agent mode. The timeline is organized into units of approximately one second. In the user_audio_slot, you receive the user's speech for the current unit. When you need to speak, make a speak decision in the ai_spoken_slot and generate the text to be spoken during the current unit; otherwise, make a listen decision. You may generate think or tool_call content in the ai_non_spoken_slot. If no additional action is needed, make a no_action decision. In the input_event_slot, you receive tool results or other independent event information. Speak using the voice in the following audio.\n"
    got = run_node(f"""
const m = await import({json.dumps(module_url('system-prefixes.js'))});
const b = await import({json.dumps(demo_module_url('board', 'quick-settings.js'))});
console.log(JSON.stringify({{
  zh: m.SYSTEM_PREFIX_ZH_THINK, en: m.SYSTEM_PREFIX_EN_GENERAL,
  boardSharesZh: b.LATEST_MVP_ZH_PREFIX_TEXT === m.SYSTEM_PREFIX_ZH_THINK,
}}));
""")
    assert got["zh"] == expected_zh
    assert got["en"] == expected_en
    assert got["zh"].endswith("\n") and got["en"].endswith("\n")
    # 画板场景必须共用同一份常量，不另存副本（两处漂移过一次就再也对不上）。
    assert got["boardSharesZh"] is True


def test_defaults_expose_full_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """部署信息必须原样给出当前加载模型的完整路径，不缩写、不省略。"""

    from demo_gallery.deployment import deployment_model_identity

    profile = tmp_path / "deployment_profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile_id": "demo-profile",
                "model_path": "/user/sunweiyue/models/MiniCPM-o-4_6",
                "pt_path": "/backup/user/someone/ckpt/iter_500.pt",
                "backbone_dir": "/backup/user/sunweiyue/delivery/ckpt/iter_500/backbone",
                "checkpoint_sha256": None,
                "deployment_mode": "tp2",
                "sdk_version": "0.0.6a3",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FC_DEPLOYMENT_PROFILE", str(profile))

    identity = deployment_model_identity()
    assert identity["pt_path"] == "/backup/user/someone/ckpt/iter_500.pt"
    assert identity["backbone_dir"] == "/backup/user/sunweiyue/delivery/ckpt/iter_500/backbone"
    assert identity["base_model_path"] == "/user/sunweiyue/models/MiniCPM-o-4_6"
    assert identity["deployment_mode"] == "tp2"
    assert identity["sdk_version"] == "0.0.6a3"
    assert identity["checkpoint_sha256"] is None
    assert identity["profile_path"] == str(profile)

    # Profile 不可读时不能把整个 defaults 拖垮，各项给 None 即可。
    monkeypatch.setenv("FC_DEPLOYMENT_PROFILE", str(tmp_path / "missing.json"))
    degraded = deployment_model_identity()
    assert degraded["pt_path"] is None and degraded["deployment_mode"] is None


@needs_node
def test_model_identity_rows_cover_every_identity_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前端展示行必须与后端字段逐一对应 —— 后端新增字段而前端漏显示会红。

    前端侧直接 import `model-identity.js`，不再从 app.js 正则抠数组字面量。
    """

    from demo_gallery.deployment import deployment_model_identity

    monkeypatch.setenv("FC_DEPLOYMENT_PROFILE", str(tmp_path / "missing.json"))
    backend_fields = set(deployment_model_identity())

    got = run_node(f"""
const m = await import({json.dumps(module_url('model-identity.js'))});
console.log(JSON.stringify({{
  keys: m.MODEL_IDENTITY_ROWS.map((row) => row[0]),
  labels: m.MODEL_IDENTITY_ROWS.map((row) => row[1]),
  hintKeys: Object.keys(m.MODEL_IDENTITY_HINTS),
}}));
""")
    assert len(got["keys"]) == len(set(got["keys"])), f"有重复行: {got['keys']}"
    assert set(got["keys"]) == backend_fields
    assert all(isinstance(label, str) and label for label in got["labels"])
    # 悬停说明只能挂在真实存在的行上。
    assert set(got["hintKeys"]) <= set(got["keys"])
    assert '<dl id="modelIdentity"' in (SHELL_HTML).read_text(encoding="utf-8")
