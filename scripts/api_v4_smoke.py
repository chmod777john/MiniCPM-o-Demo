"""对一个 API 端点跑一次真实 v4 会话，打印 JSON 摘要（部署验收用）。

    python scripts/api_v4_smoke.py --url wss://<host>:<port>/e/<端点>/v1/realtime?mode=audio \\
        --ref <16k 单声道参考音频.wav> --user <16k 单声道用户音频.wav> [--seconds 20] [--hold 0] \
        [--config '{"tts_pad_protocol": "legacy"}']

- 参考音频按 v4 以 base64 WAV 上传（同时作 system 音频段与 tts_prompt_audio）。
- 用户音频逐 Unit 锁步发送（发一个 Unit，等 response.unit.committed 再发下一个）。
- ``--hold N``：音频发完后再发 N 个静音 Unit，用来让会话多占住 Worker 一会儿（测排队 / 驱逐）。
- ``--config JSON``：合并进 ``payload.config``（例如覆盖 tts_pad / 预算 / 调度；不传就用端点默认值）。
- 摘要含：是否排过队、分到的 worker_id、session.created.model 与 session_config（实际生效值）、
  模型说的话、结束原因。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clients.v4_payload import wav_file_audio  # noqa: E402

SYSTEM_TEXT = "你是一个语音助手，请用自然口语简洁地回答用户。"


async def run(args: argparse.Namespace, summary: dict) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    user, sr = sf.read(args.user, dtype="float32")
    if sr != 16000 or user.ndim != 1:
        raise SystemExit("--user 必须是 16 kHz 单声道")
    user = user[: int(args.seconds * sr)]
    ref = wav_file_audio(args.ref)
    async with websockets.connect(args.url, ssl=ctx if args.url.startswith("wss") else None,
                                  max_size=128 << 20, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {
            "mode": "full_duplex", "fc_duplex": True, "protocol_version": "4",
            "system": {"segments": [{"kind": "text", "text": SYSTEM_TEXT}, {"kind": "audio", "audio": ref}],
                       "tools": []},
            "tts_prompt_audio": ref, "generate_audio": True,
            "config": {"decoding_policy": "constrained_decoding", **args.config},
        }}))
        while True:
            event = json.loads(await ws.recv())
            kind = event.get("type")
            if kind == "session.queued":
                summary["queued"] = True
            elif kind == "session.queue_done":
                summary["worker_id"] = event.get("worker_id")
                summary["dispatched_after_s"] = round(time.time() - summary["started_at"], 1)
            elif kind == "session.created":
                summary["model"] = event.get("model")
                summary["session_config"] = event.get("session_config")
                break
            elif kind in ("error", "session.closed"):
                summary["error"] = event
                return summary
        units = list(range(0, len(user), sr)) + [None] * args.hold
        for index, start in enumerate(units):
            chunk = np.zeros(sr, np.float32)
            if start is not None:
                piece = user[start:start + sr]
                chunk[: len(piece)] = piece
            await ws.send(json.dumps({"type": "input.append", "input": {
                "input_id": f"unit_{index:06d}", "sample_rate": 16000,
                "audio_base64": base64.b64encode(chunk.tobytes()).decode()}}))
            while True:
                event = json.loads(await ws.recv())
                kind = event.get("type")
                if kind == "response.spoken.end" and event.get("reason") == "turn_eos":
                    summary["spoken"].append(event.get("full_text"))
                if kind == "session.closed":
                    summary["closed_reason"] = event.get("reason")
                    return summary
                if kind == "response.unit.committed":
                    summary["units"] += 1
                    break
        await ws.send(json.dumps({"type": "session.close"}))
        while True:
            event = json.loads(await ws.recv())
            if event.get("type") == "session.closed":
                summary["closed_reason"] = event.get("reason")
                return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--hold", type=int, default=0)
    parser.add_argument("--config", type=json.loads, default={}, help="合并进 payload.config 的 JSON 对象")
    args = parser.parse_args()
    summary: dict = {"queued": False, "worker_id": None, "model": None, "spoken": [], "units": 0,
                     "closed_reason": None, "error": None, "started_at": time.time()}
    try:
        asyncio.run(run(args, summary))
    except Exception as exc:  # noqa: BLE001 验收脚本：任何失败都如实打印（保留已经拿到的部分摘要）
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["failed_after_s"] = round(time.time() - summary["started_at"], 1)
    summary.pop("started_at", None)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
