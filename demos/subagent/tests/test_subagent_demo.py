"""subagent Demo：隔离边界、回灌形态、preset 口径。本文件的测试都不启动 Claude Code。"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import demos.subagent.claude_runner as cr
import demos.subagent.demo as sd
from demos.subagent.presets import PRESET_BUNDLES, SUB_AGENT_ROOT, subagent_presets
from tests.conftest import needs_node, repo_module_url, run_node


# ---------------------------------------------------------------- 隔离边界


def test_permission_settings_only_allow_workspace_and_deny_bash(tmp_path: Path) -> None:
    settings = cr.permission_settings(tmp_path)
    perms = settings["permissions"]
    root = f"/{tmp_path.resolve()}/**"
    assert set(perms["allow"]) == {f"Read({root})", f"Edit({root})", f"Write({root})", "WebSearch", "WebFetch"}
    assert perms["deny"] == ["Bash"]
    assert perms["defaultMode"] == "dontAsk"          # 未放行的调用直接拒绝，不会等人批准


def test_claude_env_refuses_without_explicit_settings(monkeypatch) -> None:
    monkeypatch.delenv("SUBAGENT_CLAUDE_SETTINGS", raising=False)
    with pytest.raises(RuntimeError, match="SUBAGENT_CLAUDE_SETTINGS"):
        cr.claude_env()


def test_no_personal_default_paths_in_demo_code() -> None:
    """用谁的 Claude Code / 网关密钥只能由部署者配置；代码里出现任何人的家目录都会被克隆者悄悄用上。"""

    for path in ("claude_runner.py", "demo.py"):
        text = (Path(cr.__file__).parent / path).read_text(encoding="utf-8")
        assert not re.search(r"/user/\w|/home/\w|\.npm-global|settings\.json", text), path


def test_claude_env_isolates_config_and_drops_parent_claude_vars(tmp_path: Path, monkeypatch) -> None:
    user = tmp_path / "settings.json"
    user.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw", "ANTHROPIC_API_KEY": "k"}}))
    monkeypatch.setenv("SUBAGENT_CLAUDE_SETTINGS", str(user))
    monkeypatch.setattr(cr, "CLAUDE_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    env = cr.claude_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://gw" and env["ANTHROPIC_API_KEY"] == "k"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "cfg") == env["HOME"]
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_SESSION_ID" not in env


@pytest.mark.parametrize("bad", ["", "short", "../../etc", "a/b/c/d/e/f", "x" * 65, "has space1"])
def test_workspace_id_cannot_escape(bad: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "WORKSPACE_ROOT", tmp_path)
    with pytest.raises(ValueError):
        cr.SubagentRunner().workspace(bad)


def test_new_workspace_is_an_empty_directory_under_the_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "WORKSPACE_ROOT", tmp_path)
    ws = cr.SubagentRunner().workspace("ws_abcdef123")
    assert ws.path == (tmp_path / "ws_abcdef123").resolve() and list(ws.path.iterdir()) == []


def _client(tmp_path: Path, monkeypatch, *, unset: tuple[str, ...] = ()) -> TestClient:
    monkeypatch.setattr(cr, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(sd, "runner", cr.SubagentRunner())
    monkeypatch.setenv("SUBAGENT_CLAUDE_BIN", "/nonexistent/claude")
    monkeypatch.setenv("SUBAGENT_CLAUDE_SETTINGS", str(tmp_path / "settings.json"))
    for name in unset:
        monkeypatch.delenv(name, raising=False)
    app = FastAPI()
    app.include_router(sd.router, prefix="/api/demos/subagent")
    return TestClient(app)


# 本 Demo 的全部端点（方法, 路径）；新增端点时这里也要加，确保都在配置检查之后。
ENDPOINTS = [
    ("post", "/api/demos/subagent/workspaces/ws_abcdef123"),
    ("get", "/api/demos/subagent/workspaces/ws_abcdef123/files"),
    ("post", "/api/demos/subagent/workspaces/ws_abcdef123/files"),
    ("post", "/api/demos/subagent/tasks"),
    ("get", "/api/demos/subagent/tasks/abc/events"),
]


def test_every_endpoint_is_listed_here() -> None:
    routes = {(m.lower(), "/api/demos/subagent" + r.path) for r in sd.router.routes for m in r.methods}
    assert routes == {(m, path.replace("ws_abcdef123", "{workspace_id}").replace("/abc/", "/{task_id}/"))
                      for m, path in ENDPOINTS}


@pytest.mark.parametrize("method,path", ENDPOINTS)
@pytest.mark.parametrize("missing", ["SUBAGENT_CLAUDE_BIN", "SUBAGENT_CLAUDE_SETTINGS"])
def test_any_missing_config_disables_everything(missing: str, method: str, path: str, tmp_path: Path,
                                                monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, unset=(missing,))
    r = getattr(client, method)(path, **({"json": {}} if method == "post" else {}))
    assert r.status_code == 503 and missing in r.json()["detail"]
    assert not (tmp_path / "ws_abcdef123").exists()


def test_configured_demo_needs_no_password(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    assert client.post("/api/demos/subagent/workspaces/ws_abcdef123").status_code == 200


def test_upload_stays_in_workspace_root(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    url = "/api/demos/subagent/workspaces/ws_abcdef123/files"
    ok = client.post(url, json={"name": "report.txt", "content_base64": base64.b64encode(b"hi").decode()})
    assert ok.status_code == 200 and (tmp_path / "ws_abcdef123" / "report.txt").read_bytes() == b"hi"
    for name in ("../escape.txt", "a/b.txt", ".hidden", "..", ""):
        bad = client.post(url, json={"name": name, "content_base64": base64.b64encode(b"x").decode()})
        assert bad.status_code == 400, name
    assert not (tmp_path / "escape.txt").exists()
    listing = client.get(url).json()
    assert listing["files"] == [{"name": "report.txt", "size": 2}]


def test_unknown_tool_is_rejected_without_starting_anything(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    r = client.post("/api/demos/subagent/tasks", json={
        "workspace_id": "ws_abcdef123", "tool_call_id": "t1", "name": "rm_rf", "arguments": {}})
    assert r.status_code == 400 and "不处理工具" in r.json()["detail"]


# ---------------------------------------------------------------- 回灌形态（对照训练数据）


def test_extract_and_strip_progress() -> None:
    text = "PROGRESS: 已找到两篇文献，正在核对数据\n中间说明\nPROGRESS：价格查到了"
    assert cr.extract_progress(text) == ["已找到两篇文献，正在核对数据", "价格查到了"]
    assert cr.strip_progress(text) == "中间说明"


def test_calculate_result_matches_training_format() -> None:
    got = cr.format_result("calculate", '计算如下 {"type":"number","value":19727.4,"approximate":false}', {})
    # 训练数据原文：result ok True; result result type number; result result value 19727.4; ...
    assert got == ("result ok True; result result type number; result result value 19727.4; "
                   "result result display 19727.4; result result approximate False")


def test_web_search_result_matches_training_format() -> None:
    got = cr.format_result("web_search", json.dumps({"items": [
        {"title": "迭部天气", "url": "https://a.example", "snippet": "14.8℃"},
        {"title": "预报15天", "url": "https://b.example", "snippet": "18/9℃"},
    ]}, ensure_ascii=False), {"query": "迭部 气温"})
    lines = got.split("\n")
    assert lines[0].startswith("result ok True; result spoken 搜到 2 条（")
    assert lines[1] == "1. 迭部天气（https://a.example）14.8℃"


def test_search_and_calculate_prompts_label_the_argument() -> None:
    assert cr.build_prompt("web_search", {"query": "MiniCPM-o 面壁智能"}, "zh")[0] == "搜索关键词：MiniCPM-o 面壁智能"
    assert cr.build_prompt("web_search", {"query": "MiniCPM-o"}, "en")[0] == "Search query: MiniCPM-o"
    assert cr.build_prompt("calculate", {"expression": "2940 * 6.71"}, "zh")[0] == "表达式：2940 * 6.71"


def test_delegate_result_drops_progress_lines() -> None:
    assert cr.format_result("delegate_task", "PROGRESS: 查到了\n结论是可以。", {}) == "结论是可以。"


# ---------------------------------------------------------------- preset 口径

needs_data = pytest.mark.skipif(not SUB_AGENT_ROOT.is_dir(), reason="本机没有 sub_agent 训练数据")


@needs_data
def test_presets_keep_training_segments_and_tool_order() -> None:
    presets = subagent_presets()
    for preset_id, (bundle, _lang) in PRESET_BUNDLES.items():
        with (SUB_AGENT_ROOT / bundle / "traindata.jsonl").open(encoding="utf-8") as handle:
            source = json.loads(handle.readline())["system"]
        preset = presets[preset_id]
        assert "error" not in preset, preset
        assert [s["kind"] for s in preset["system"]["segments"]] == [s["kind"] for s in source["segments"]]
        assert [t["function"]["name"] for t in preset["system"]["tools"]] == [t["function"]["name"] for t in source["tools"]]


@needs_data
@needs_node
def test_frontend_preset_prepends_matching_prefix() -> None:
    defaults = {"subagent_presets": subagent_presets()}
    got = run_node(f"""
const qs = await import({json.dumps(repo_module_url('demos/subagent/web/quick-settings.js'))});
const px = await import({json.dumps(repo_module_url('static/demo-gallery/core/system-prefixes.js'))});
const map = qs.buildSubagentQuickSettings({json.dumps(defaults, ensure_ascii=False)});
console.log(JSON.stringify({{ zhNo: map.get('subagent_zh_nothink').segments[0].text === px.SYSTEM_PREFIX_ZH_NO_THINK,
  zhThink: map.get('subagent_zh_think').segments[0].text === px.SYSTEM_PREFIX_ZH_THINK,
  enNo: map.get('subagent_en_nothink').segments[0].text === px.SYSTEM_PREFIX_EN_NO_THINK }}));
""")
    assert got == {"zhNo": True, "zhThink": True, "enNo": True}


def test_claude_failure_reason_is_reported_to_model(tmp_path: Path, monkeypatch) -> None:
    """上游拦截时 Claude Code exit=1，但 result 里有原因；回给模型的应是这个原因而不是「无输出」。"""

    import asyncio

    fake = tmp_path / "fake_claude"
    fake.write_text('#!/bin/sh\necho \'{"type":"result","result":"API Error: safeguards flagged this message","is_error":true}\'\nexit 1\n')
    fake.chmod(0o755)
    settings = tmp_path / "settings.json"
    settings.write_text('{"env": {}}')
    monkeypatch.setattr(cr, "WORKSPACE_ROOT", tmp_path / "ws")
    monkeypatch.setattr(cr, "CLAUDE_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setenv("SUBAGENT_CLAUDE_BIN", str(fake))
    monkeypatch.setenv("SUBAGENT_CLAUDE_SETTINGS", str(settings))

    async def go() -> cr.Task:
        runner = cr.SubagentRunner()
        task = runner.submit(workspace_id="ws_abcdef123", tool_call_id="t1", tool_name="delegate_task",
                             arguments={"message": "hi"}, language="zh")
        for _ in range(100):
            if task.done:
                break
            await asyncio.sleep(0.05)
        return task

    task = asyncio.run(go())
    assert [(e.kind, e.text) for e in task.events] == [("error", "API Error: safeguards flagged this message")]
