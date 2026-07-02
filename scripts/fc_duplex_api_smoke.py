"""Minimal FC duplex backend-protocol smoke.

This script is intentionally small: it synthesizes one Chinese user utterance
with edge-tts, streams it as 1s float32 PCM chunks, and prints protocol events.
It does not tune prompts or retry until a tool call appears.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import tempfile
import wave
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import edge_tts
import librosa
import numpy as np
import websockets


USER_TEXT = """接下来我给小朋友会讲一个故事，故事中出现的动物你给我放到画板上
清晨的森林刚刚醒来，一只小松鼠从老橡树上探出头，蹦蹦跳跳地跑下来，草地上一只梅花鹿正在吃草，不远处的小河边一只灰兔子蹲在河边喝水"""


SYSTEM_PROMPT = "你是一个可以一边听用户说话、一边思考并调用工具的语音助手。用户要求把故事里出现的动物放到画板上时，使用 display_object_on_board 工具。"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "display_object_on_board",
            "description": "Display a named concrete object on the visual board so the user can see it.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
]


def ws_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/") + "/backend")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


async def synthesize(text: str, output: Path) -> None:
    await edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural").save(str(output))


def load_float32(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def audio_to_b64(audio: np.ndarray) -> str:
    return base64.b64encode(np.asarray(audio, dtype=np.float32).reshape(-1).tobytes()).decode("utf-8")


def save_audio_delta(event: dict, output_dir: Path, index: int) -> None:
    audio = event.get("audio")
    if not audio:
        return
    sample_rate = int(event.get("sample_rate") or 24000)
    pcm = np.frombuffer(base64.b64decode(audio), dtype=np.float32)
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    path = output_dir / f"fc_smoke_audio_{index:03d}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    print(f"saved_audio={path}", flush=True)


async def recv_until_idle(ws: websockets.WebSocketClientProtocol, *, output_dir: Path, audio_index: int) -> int:
    idle_rounds = 0
    while idle_rounds < 3:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            idle_rounds += 1
            continue
        event = json.loads(raw)
        event_type = event.get("type")
        if event_type in {
            "response.output.delta",
            "response.think.delta",
            "response.tool_call.args.raw",
            "response.tool_result",
            "response.output.sp_tokens",
            "session.closed",
        }:
            print(json.dumps(event, ensure_ascii=False)[:2000], flush=True)
        if event_type == "response.output.delta" and event.get("kind") == "audio":
            audio_index += 1
            save_audio_delta(event, output_dir, audio_index)
        if event_type == "session.closed":
            break
    return audio_index


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run one FC duplex API smoke")
    parser.add_argument("--backend-url", default="http://127.0.0.1:22500")
    parser.add_argument("--text", default=USER_TEXT)
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--output-dir", default="/tmp/fc_duplex_api_smoke")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.audio_path:
        audio_path = Path(args.audio_path)
    else:
        fd, tmp_name = tempfile.mkstemp(prefix="fc_user_", suffix=".mp3", dir=str(output_dir))
        Path(tmp_name).unlink(missing_ok=True)
        audio_path = Path(tmp_name)
        await synthesize(args.text, audio_path)
        print(f"synth_audio={audio_path}", flush=True)

    audio = load_float32(audio_path, args.sample_rate)
    chunk_size = max(1, int(args.sample_rate * args.chunk_sec))
    audio_index = 0
    async with websockets.connect(ws_url(args.backend_url), max_size=128 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "system_prompt": SYSTEM_PROMPT,
                "tools": TOOLS,
                "generate_audio": True,
                "config": {
                    "runtime": "fc_duplex",
                    "sample_rate": args.sample_rate,
                    "max_spoken_tokens": 24,
                    "non_spoken_budget_per_unit": 12,
                    "decode_mode": "greedy",
                    "auto_execute_tools": True,
                },
            },
        }, ensure_ascii=False))
        print(await ws.recv(), flush=True)
        for offset in range(0, len(audio), chunk_size):
            chunk = audio[offset : offset + chunk_size]
            await ws.send(json.dumps({
                "type": "input.append",
                "input": {
                    "input_id": f"in_{offset // chunk_size:04d}",
                    "audio_base64": audio_to_b64(chunk),
                    "sample_rate": args.sample_rate,
                },
            }))
            audio_index = await recv_until_idle(ws, output_dir=output_dir, audio_index=audio_index)
        audio_index = await recv_until_idle(ws, output_dir=output_dir, audio_index=audio_index)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
