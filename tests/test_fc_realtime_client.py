"""FC Board Realtime Client 的 Session 握手顺序测试。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_audio_cannot_overtake_session_init() -> None:
    """排队期间禁止发送音频，Session 创建后才允许 input.append。"""

    project_root = Path(__file__).resolve().parents[1]
    module_path = (
        project_root
        / "static"
        / "fc-board"
        / "fc-realtime-client.js"
    )
    script = f"""
import fs from 'node:fs';
globalThis.WebSocket = {{ OPEN: 1 }};
const source = fs.readFileSync({json.dumps(str(module_path))}, 'utf8');
const url = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const module = await import(url);
const sent = [];
const client = new module.FcRealtimeClient();
client.ws = {{
  readyState: WebSocket.OPEN,
  send: (payload) => sent.push(JSON.parse(payload)),
}};
client.initSession({{protocol_version: '3'}});
let earlyError = null;
try {{
  client.appendAudio({{audioBase64: 'AAAA', sampleRate: 16000}});
}} catch (error) {{
  earlyError = error.message;
}}
client._handleMessage({{data: JSON.stringify({{type: 'session.queue_done'}})}});
client._handleMessage({{data: JSON.stringify({{type: 'session.created', session_id: 'sess_test'}})}});
client.appendAudio({{audioBase64: 'BBBB', sampleRate: 16000}});
console.log(JSON.stringify({{earlyError, sent}}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["earlyError"] == "Session is not ready for audio input"
    assert [frame["type"] for frame in result["sent"]] == [
        "session.init",
        "input.append",
    ]

