"""uicard 前端纯逻辑：会话时间刷新、卡片编号与注册表、preset 拼装（node 直接 import 模块测）。"""

from __future__ import annotations

import json

from tests.conftest import needs_node, repo_module_url, run_node

SESSION_TIME = repo_module_url("demos/uicard/web/session-time.js")
REGISTRY = repo_module_url("demos/uicard/web/card-registry.js")
QUICK = repo_module_url("demos/uicard/web/quick-settings.js")

TEXT_1016 = (
    "〔本次会话〕\n端：desktop_client\n"
    "服务端本地时间：2026-10-13 02:22:04 CST；UTC 时间：2026-10-12T18:22:04Z；Unix 时间戳：1791829324\n"
    "（这就是第④条里说的会话初始时间）"
)
TEXT_1004 = (
    "[当前会话时间]\n服务端本地时间：2026-07-22 13:09:11 CST\nUTC 时间：2026-07-22T05:09:11Z\n"
    "Unix 时间戳：1784696951\n\n[时间相关规则]"
)


@needs_node
def test_restamp_rewrites_both_training_formats():
    now_ms = 1792335315533  # 2026-10-18T14:55:15Z = 北京 22:55:15
    got = run_node(f"""
const m = await import({json.dumps(SESSION_TIME)});
const segs = [{{kind:'text', text:{json.dumps(TEXT_1016)}}}, {{kind:'audio', audio:{{source:'path', file_path:'/a.wav'}}}},
              {{kind:'text', text:{json.dumps(TEXT_1004)}}}, {{kind:'text', text:'无时间块'}}];
console.log(JSON.stringify(m.restampSegments(segs, {now_ms})));
""")
    assert got[0]["text"] == (
        "〔本次会话〕\n端：desktop_client\n"
        "服务端本地时间：2026-10-18 22:55:15 CST；UTC 时间：2026-10-18T14:55:15Z；Unix 时间戳：1792335315\n"
        "（这就是第④条里说的会话初始时间）"
    )
    assert got[2]["text"] == (
        "[当前会话时间]\n服务端本地时间：2026-10-18 22:55:15 CST\nUTC 时间：2026-10-18T14:55:15Z\n"
        "Unix 时间戳：1792335315\n\n[时间相关规则]"
    )
    assert got[1] == {"kind": "audio", "audio": {"source": "path", "file_path": "/a.wav"}}
    assert got[3]["text"] == "无时间块"


@needs_node
def test_card_numbers_continue_across_calls_and_unknown_ids_are_misses():
    got = run_node(f"""
const m = await import({json.dumps(REGISTRY)});
const r = new m.CardRegistry();
const out = {{}};
out.firstMap = r.indexBaseFor('map_search');
r.register('map_search', [{{card_id:'m1', card:{{type:'map'}}}}]);
out.secondMap = r.indexBaseFor('map_search');
r.register('amap_weather', [{{card_id:'w1', card:{{type:'weather'}}}}, {{card_id:'w2', card:{{type:'weather'}}}}]);
out.nextWeather = r.indexBaseFor('amap_weather');
out.search = r.indexBaseFor('serp_google_search');
out.hit = r.get('w2');
out.miss = r.get('m9');
const order = [];
await Promise.all([
  r.runExclusive(async () => {{ await new Promise((ok) => setTimeout(ok, 20)); order.push('a'); }}),
  r.runExclusive(async () => {{ order.push('b'); }}),
]);
out.order = order;
r.reset();
out.afterReset = [r.indexBaseFor('map_search'), r.get('m1')];
out.message = m.unknownCardIdMessage('m9');
console.log(JSON.stringify(out));
""")
    assert got["firstMap"] == 1 and got["secondMap"] == 2 and got["nextWeather"] == 3
    assert got["search"] is None
    assert got["hit"] == {"type": "weather"} and got["miss"] is None
    assert got["order"] == ["a", "b"]
    assert got["afterReset"] == [1, None]
    assert "unknown card_id: m9" in got["message"]


@needs_node
def test_presets_prepend_prefix_2_and_keep_case_tools():
    defaults = {
        "uicard_presets": {
            "uicard_1016_r002": {
                "data_id": "case-a",
                "system": {
                    "segments": [{"kind": "audio", "audio": {"source": "path", "file_path": "/ref.wav"}},
                                 {"kind": "text", "text": "正文"}],
                    "tools": [{"type": "function", "function": {"name": "tavily_search"}}],
                },
                "tts_prompt_audio": {"source": "path", "file_path": "/ref.wav"},
            },
            "uicard_1004_r011": {"source": "/x", "error": "FileNotFoundError: x"},
        }
    }
    prefixes = repo_module_url("static/demo-gallery/core/system-prefixes.js")
    got = run_node(f"""
const q = await import({json.dumps(QUICK)});
const p = await import({json.dumps(prefixes)});
const d = {json.dumps(defaults, ensure_ascii=False)};
const settings = q.buildUiCardQuickSettings(d);
const s = settings.get(q.PRESET_1016);
console.log(JSON.stringify({{
  ids: [...settings.keys()],
  prefixOk: s.segments[0].text === p.SYSTEM_PREFIX_ZH_NO_THINK,
  kinds: s.segments.map((x) => x.kind),
  tts: s.ttsPromptAudioPath,
  tools: q.presetTools(d, q.PRESET_1016).map((t) => t.function.name),
  fallbackTools: q.presetTools(d, q.PRESET_1004).map((t) => t.function.name),
  unavailable: q.unavailablePresets(d).map((x) => x.id),
  dataId: q.presetDataId(d, q.PRESET_1016),
}}));
""")
    assert got == {
        "ids": ["uicard_1016_r002"],
        "prefixOk": True,
        "kinds": ["text", "audio", "text"],
        "tts": "/ref.wav",
        "tools": ["tavily_search"],
        "fallbackTools": ["tavily_search"],
        "unavailable": ["uicard_1004_r011"],
        "dataId": "case-a",
    }
