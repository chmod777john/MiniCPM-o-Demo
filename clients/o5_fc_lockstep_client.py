"""FC Semantic Realtime API v3 的**锁步**参考客户端（离线推理 / 评测用）。

锁步：每个 Unit 先发该 Unit 之前到达的 ``input.tool_result``，再发一帧
``input.append``（1 个 Unit 的 16 kHz float32 PCM），然后一直读到该 Unit 的
``response.unit.committed``。推理进度只由调用方驱动，与墙钟无关。

本文件**只依赖** ``websockets``（>=12，sync API）与 ``numpy``，不 import 任何 Demo 代码，
可以原样拷进评测仓库使用（评测仓库里的副本必须与本文件逐字节一致）。

System 的拼装由服务端按 SDK 完成：``<|im_start|>system\\n`` + segments 原序 + tools +
``<|im_end|>``。调用方只提供语义内容（prefix 正文、system 参考音频、任务 prompt、工具）。
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

PROTOCOL_VERSION = "3"
INPUT_SAMPLE_RATE = 16000


class O5FcSessionError(RuntimeError):
    """Session 被服务端提前结束、报错或事件顺序不符合协议。"""


def build_system(
    *,
    system_audio_path: Optional[str],
    task_prompt: str,
    tools: Optional[list[dict[str, Any]]] = None,
    prefix_text: Optional[str] = None,
) -> dict[str, Any]:
    """按 ``prefix + 参考音频 + 任务 prompt + 工具`` 构造 v3 ``system``。

    参数:
        system_audio_path: 服务端可读的 system 参考音频绝对路径（16 kHz 单声道）；None 表示不放音频。
        task_prompt: 评测自带的任务 system prompt（不变量）。
        tools: OpenAI 风格工具定义；None/空表示无工具。
        prefix_text: 可选 prefix 正文（不含 ``system\\n``，由服务端统一加）；None 表示不放 prefix。

    返回:
        可直接放进 ``session.init.payload.system`` 的 dict。
    """

    segments: list[dict[str, Any]] = []
    if prefix_text:
        segments.append({"kind": "text", "text": prefix_text})
    if system_audio_path:
        segments.append(
            {"kind": "audio", "audio": {"source": "path", "file_path": system_audio_path}}
        )
    if task_prompt:
        segments.append({"kind": "text", "text": task_prompt})
    return {"segments": segments, "tools": list(tools or [])}


@dataclass(frozen=True)
class O5FcSessionPreset:
    """模型调用口径（per-checkpoint 事实），由最外层调用方用一个 JSON 文件给出。

    JSON 字段与本类字段同名；除 ``prefix_text`` 外全部必填，不提供任何默认值::

        {
          "prefix_text": "You are operating in duplex Agent mode. ...\\n",   // 或 null
          "system_audio_path": "/abs/F409-379_clip_310-324_fix_16khz.wav",  // 或 null
          "tts_prompt_audio_path": "/abs/F409-379_clip_310-324_fix_16khz.wav",  // 或 null=不出音频
          "unit_policy": {"unit_sec": 1.0,
                          "non_spoken_budgets_while_listening": [30],
                          "non_spoken_budgets_while_speaking": [15]},
          "non_spoken_scheduling": "quality",
          "decode_mode": "greedy"
        }
    """

    prefix_text: Optional[str]
    system_audio_path: Optional[str]
    tts_prompt_audio_path: Optional[str]
    unit_policy: dict[str, Any]
    non_spoken_scheduling: str
    decode_mode: str

    @classmethod
    def load(cls, path: str) -> "O5FcSessionPreset":
        """读取并校验 preset JSON（缺字段、多字段、音频不存在都直接报错）。"""

        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        expected = set(cls.__dataclass_fields__)
        missing, unknown = expected - set(data), set(data) - expected
        if missing or unknown:
            raise ValueError(
                f"session preset {path}: 缺字段 {sorted(missing)}，未知字段 {sorted(unknown)}"
            )
        preset = cls(**data)
        for audio_path in (preset.system_audio_path, preset.tts_prompt_audio_path):
            if audio_path is not None and not os.path.isfile(audio_path):
                raise FileNotFoundError(f"session preset {path}: 音频不存在 {audio_path}")
        return preset

    def to_spec(
        self,
        *,
        server_url: str,
        task_prompt: str,
        tools: Optional[list[dict[str, Any]]],
    ) -> "O5FcSessionSpec":
        """把 preset 与评测自带的任务 prompt / 工具组合成一个 Session 规格。"""

        return O5FcSessionSpec(
            server_url=server_url,
            system=build_system(
                system_audio_path=self.system_audio_path,
                task_prompt=task_prompt,
                tools=tools,
                prefix_text=self.prefix_text,
            ),
            unit_policy=self.unit_policy,
            tts_prompt_audio_path=self.tts_prompt_audio_path,
            non_spoken_scheduling=self.non_spoken_scheduling,
            decode_mode=self.decode_mode,
        )


@dataclass(frozen=True)
class O5FcSessionSpec:
    """一个 Session 的全部请求侧语义（per-checkpoint 事实由调用方决定）。

    参数:
        server_url: Gateway 根地址，如 ``http://127.0.0.1:8009``。
        system: ``build_system`` 的结果。
        unit_policy: SDK ``O5UnitPolicy`` 的 JSON（至少含 unit_sec 与两组 budget）。
        tts_prompt_audio_path: TTS 音色 prompt 的服务端可读路径；None 表示不生成音频。
        non_spoken_scheduling: ``quality``（按 budget 完整解码）或 ``latency``。
        decode_mode: ``greedy`` 或 ``sampling``。
        queue_timeout_sec: 等待空闲副本的上限（并发 > 副本数时会排队）。
        event_timeout_sec: 等待单个服务端事件的上限。
    """

    server_url: str
    system: dict[str, Any]
    unit_policy: dict[str, Any]
    tts_prompt_audio_path: Optional[str]
    non_spoken_scheduling: str = "quality"
    decode_mode: str = "greedy"
    queue_timeout_sec: float = 7200.0
    event_timeout_sec: float = 900.0

    def session_init_frame(self) -> dict[str, Any]:
        """构造 ``session.init`` 帧。"""

        payload: dict[str, Any] = {
            "mode": "full_duplex",
            "fc_duplex": True,
            "protocol_version": PROTOCOL_VERSION,
            "tokenizer_target": "o5",
            "system": self.system,
            "generate_audio": self.tts_prompt_audio_path is not None,
            "unit_policy": self.unit_policy,
            "config": {
                "runtime": "fc_duplex",
                "decode_mode": self.decode_mode,
                "sample_rate": INPUT_SAMPLE_RATE,
                "non_spoken_scheduling": self.non_spoken_scheduling,
            },
        }
        if self.tts_prompt_audio_path is not None:
            payload["tts_prompt_audio"] = {
                "source": "path",
                "file_path": self.tts_prompt_audio_path,
            }
        return {"type": "session.init", "payload": payload}


@dataclass
class O5FcToolCall:
    """模型在某个 Unit 完成的一次工具调用。"""

    tool_call_id: str
    name: Optional[str]
    arguments: dict[str, Any]
    error: Optional[str] = None


@dataclass
class O5FcUnitResult:
    """一个 Unit 的全部输出（按协议事件聚合）。"""

    unit_index: int
    spoken_text: str = ""
    think_text: str = ""
    non_spoken_render: str = ""
    audio: Optional[np.ndarray] = None
    audio_sample_rate: Optional[int] = None
    tool_calls: list[O5FcToolCall] = field(default_factory=list)
    consumed_tool_call_ids: list[str] = field(default_factory=list)
    is_speaking: bool = False
    is_listen: bool = False
    end_of_turn: bool = False
    spoken_end_reason: Optional[str] = None
    non_spoken_end_reason: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)


def _ws_url(server_url: str) -> str:
    parsed = urlsplit(server_url.rstrip("/") + "/v1/realtime?mode=audio")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _steps_text(event: dict[str, Any]) -> str:
    return "".join(
        step.get("text", "")
        for step in event.get("steps") or []
        if step.get("kind") == "text"
    )


def _without_audio(event: dict[str, Any]) -> dict[str, Any]:
    if "audio" not in event:
        return event
    slim = dict(event)
    slim["audio"] = f"<{len(event['audio'] or '')} b64 chars>"
    return slim


class O5FcLockstepSession:
    """一个锁步 Session；用 ``with`` 管理生命周期。"""

    def __init__(self, spec: O5FcSessionSpec) -> None:
        self.spec = spec
        self.session_created: dict[str, Any] = {}
        self._ws = None
        self._next_unit = 0
        self._closed = False

    # ── lifecycle ───────────────────────────────────────────────────
    def __enter__(self) -> "O5FcLockstepSession":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def open(self) -> dict[str, Any]:
        """连接 Gateway、等待分配副本并初始化 Session。

        返回:
            ``session.created`` 事件（含 ``system_prefill`` 审计渲染）。
        """

        self._ws = connect(
            _ws_url(self.spec.server_url),
            max_size=256 * 1024 * 1024,
            open_timeout=self.spec.queue_timeout_sec,
            close_timeout=5,
            proxy=None,
        )
        deadline = time.monotonic() + self.spec.queue_timeout_sec
        while True:
            event = self._recv(max(1.0, deadline - time.monotonic()))
            if event.get("type") == "session.queue_done":
                break
            self._raise_if_terminal(event)
        self._send(self.spec.session_init_frame())
        while True:
            event = self._recv(self.spec.event_timeout_sec)
            self._raise_if_terminal(event)
            if event.get("type") == "session.created":
                self.session_created = event
                return event

    def close(self) -> None:
        """发送 ``session.close`` 并尽力等待 ``session.closed``。"""

        if self._ws is None or self._closed:
            return
        self._closed = True
        try:
            self._send({"type": "session.close", "reason": "input_complete"})
            while True:
                if self._recv(10.0).get("type") == "session.closed":
                    break
        except (TimeoutError, ConnectionClosed, O5FcSessionError):
            pass
        finally:
            self._ws.close()

    # ── one unit ────────────────────────────────────────────────────
    def step(
        self,
        audio: np.ndarray,
        tool_results: Optional[list[tuple[str, Any]]] = None,
    ) -> O5FcUnitResult:
        """推进一个 Unit。

        参数:
            audio: 本 Unit 的 16 kHz 单声道 float32 PCM（长度 = unit_sec * 16000）。
            tool_results: 本 Unit 之前到达的 ``(tool_call_id, content)``；content 可为任意 JSON。

        返回:
            本 Unit 的聚合输出。
        """

        if self._ws is None or self._closed:
            raise O5FcSessionError("Session 未打开或已关闭")
        for tool_call_id, content in tool_results or []:
            self._send(
                {"type": "input.tool_result", "tool_call_id": tool_call_id, "content": content}
            )
        unit_index = self._next_unit
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        self._send(
            {
                "type": "input.append",
                "input": {
                    "input_id": f"unit_{unit_index:06d}",
                    "audio_base64": base64.b64encode(pcm.tobytes()).decode("ascii"),
                    "sample_rate": INPUT_SAMPLE_RATE,
                },
            }
        )
        result = O5FcUnitResult(unit_index=unit_index)
        audio_chunks: list[np.ndarray] = []
        while True:
            event = self._recv(self.spec.event_timeout_sec)
            self._raise_if_terminal(event)
            event_type = event.get("type", "")
            if not event_type.startswith("response."):
                continue
            result.events.append(_without_audio(event))
            if event_type == "response.unit.started":
                result.consumed_tool_call_ids = [
                    item.get("tool_call_id")
                    for item in event.get("tool_events") or []
                    if item.get("tool_call_id")
                ]
            elif event_type == "response.spoken.delta":
                result.is_speaking = True
                result.spoken_text += _steps_text(event)
                if event.get("audio"):
                    audio_chunks.append(
                        np.frombuffer(base64.b64decode(event["audio"]), dtype=np.float32)
                    )
                    result.audio_sample_rate = int(event.get("sample_rate") or 24000)
            elif event_type == "response.spoken.end":
                result.spoken_end_reason = event.get("reason")
                result.is_listen = result.spoken_end_reason == "listen"
                result.end_of_turn = result.spoken_end_reason == "turn_eos"
            elif event_type == "response.think.begin":
                result.non_spoken_render += "<think>"
            elif event_type == "response.think.delta":
                text = _steps_text(event)
                result.think_text += text
                result.non_spoken_render += text
            elif event_type == "response.think.end":
                result.non_spoken_render += "</think>"
            elif event_type == "response.tool_call.begin":
                result.non_spoken_render += "<tool_call>"
            elif event_type == "response.tool_call.delta":
                result.non_spoken_render += _steps_text(event)
            elif event_type == "response.tool_call.done":
                result.non_spoken_render += "</tool_call>"
                call = event.get("call") or {}
                arguments = call.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                result.tool_calls.append(
                    O5FcToolCall(
                        tool_call_id=str(event.get("tool_call_id")),
                        name=call.get("name"),
                        arguments=arguments if isinstance(arguments, dict) else {},
                        error=event.get("error"),
                    )
                )
            elif event_type == "response.non_spoken.end":
                result.non_spoken_end_reason = event.get("reason")
            elif event_type == "response.unit.committed":
                committed = int(event.get("unit_index", -1))
                if committed != unit_index:
                    raise O5FcSessionError(
                        f"committed Unit 顺序错误: expected={unit_index}, actual={committed}"
                    )
                break
        if audio_chunks:
            result.audio = np.concatenate(audio_chunks)
        self._next_unit += 1
        return result

    # ── transport ───────────────────────────────────────────────────
    def _send(self, frame: dict[str, Any]) -> None:
        self._ws.send(json.dumps(frame, ensure_ascii=False))

    def _recv(self, timeout_sec: float) -> dict[str, Any]:
        raw = self._ws.recv(timeout=timeout_sec)
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise O5FcSessionError(f"服务端事件必须是 JSON object: {event!r}")
        return event

    @staticmethod
    def _raise_if_terminal(event: dict[str, Any]) -> None:
        if event.get("type") in {"error", "session.closed"}:
            raise O5FcSessionError(f"Session 被服务端结束: {json.dumps(event, ensure_ascii=False)[:2000]}")
