"""使用一条 O5 TrainingData 调用 FC Semantic API 并保存推理产物。

本样例只演示 API 输入、事件读取与结果序列化，不执行 Ground Truth 对拍。输入
TrainingData 的 system、UnitPolicy 和用户媒体用于构造请求；AI 输出轨不会发送给
服务端。脚本始终保存原始双向事件和真实音频，并尽力构造 token-equivalent replay
O5DuplexTrainingData。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
import torchaudio.functional as audio_functional
import websockets
from minicpm_o5_sdk import (
    O5DuplexTrainingData,
    O5SystemTextSegment,
    O5TokenizerID,
    O5UnitPolicy,
)
from minicpm_o5_sdk.protocols.duplex import (
    O5AINonSpokenSegment,
    O5AINonSpokenTrack,
    O5AISpokenSegment,
    O5AISpokenTrack,
    O5Alignment,
    O5DuplexArrangement,
    O5DuplexParseRequest,
    O5DuplexParser,
    O5DuplexTrainingTracks,
    O5GlobalTime,
    O5InputEventTrack,
    O5LazyAudio,
    O5ReplayMediaBundle,
    O5StartTrigger,
    O5SystemContent,
    O5ThinkContent,
    O5TokenProvenance,
    O5ToolCallContent,
    O5UserAudioSegment,
    O5UserAudioTrack,
    O5WordInterval,
    assert_duplex_parse_roundtrip,
    build_user_audio_tensor_from_arrangement,
)
from minicpm_o5_sdk.protocols.duplex.training_data import O5SystemAudioSegment
from pydantic import BaseModel, ConfigDict, Field

from core.fc_duplex.system_input import (
    FcAudioPathInput,
    FcSystemAudioInput,
    FcSystemContentInput,
    FcSystemTextInput,
)
from core.fc_duplex_resume import build_fc_duplex_resume_plan


JsonObject = dict[str, Any]
DecodeTrack = Literal["spoken", "non_spoken"]
PROTOCOL_VERSION = "3"
AUDIO_SAMPLE_RATE = 16_000
OUTPUT_AUDIO_SAMPLE_RATE = 24_000


class ApiHistoryEntry(BaseModel):
    """一条按收发顺序保存的 WebSocket 事件。"""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    direction: Literal["up", "down"]
    event: JsonObject


class ApiDecodeSpan(BaseModel):
    """输入 token scaffold 中一段需要由 API 实际 decode token 替换的区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_index: int = Field(ge=0)
    track: DecodeTrack
    start_token_index: int = Field(ge=0)
    end_token_index_exclusive: int = Field(ge=0)


class ApiUnitInput(BaseModel):
    """一个发送给 API 的 16 kHz 单声道 float32 音频 Unit。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_index: int = Field(ge=0)
    input_id: str
    audio_base64: str
    sample_rate: Literal[16000] = 16000

    def to_frame(self) -> JsonObject:
        """生成可直接发送的 ``input.append`` 帧。"""

        return {
            "type": "input.append",
            "input": {
                "input_id": self.input_id,
                "audio_base64": self.audio_base64,
                "sample_rate": self.sample_rate,
            },
        }


class ApiInferenceScenario(BaseModel):
    """由 TrainingData 投影出的纯 API 推理场景。"""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    data_id: str | None
    unit_policy: O5UnitPolicy
    system: O5SystemContent | None
    session_init: JsonObject
    units: list[ApiUnitInput]
    user_audio_samples: np.ndarray = Field(exclude=True)
    scaffold_token_ids: list[int] = Field(exclude=True)
    decode_spans: list[ApiDecodeSpan] = Field(exclude=True)
    source_arrangement: O5DuplexArrangement = Field(exclude=True)
    tokenizer_fingerprint: dict[str, str]
    source_ai_spoken_segment_count: int = Field(ge=0)
    source_ai_non_spoken_segment_count: int = Field(ge=0)
    source_input_event_count: int = Field(ge=0)


class ApiInferenceSession(BaseModel):
    """一次 API 会话保存的完整双向 history。"""

    model_config = ConfigDict(extra="forbid")

    history: list[ApiHistoryEntry]
    protocol_history: list[JsonObject]
    session_created: JsonObject
    committed_unit_indices: list[int]


class SpokenTurnArtifact(BaseModel):
    """一段 API spoken turn 的文本和音频文件。"""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    start_unit_index: int | None = None
    end_unit_index: int | None = None
    text: str
    sample_rate: int | None = None
    audio_path: str | None = None
    replay_audio_path: str | None = None


class ResponseArtifacts(BaseModel):
    """从原始 API history 提取的人类可读结果与真实音频。"""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    think_texts: list[str] = Field(default_factory=list)
    tool_calls: list[JsonObject] = Field(default_factory=list)
    spoken_turns: list[SpokenTurnArtifact] = Field(default_factory=list)
    warnings: list[JsonObject] = Field(default_factory=list)
    errors: list[JsonObject] = Field(default_factory=list)
    spoken_audio_16k: list[np.ndarray] = Field(
        default_factory=list,
        exclude=True,
    )


class ReplayTrainingDataArtifact(BaseModel):
    """replay TrainingData 的构造状态。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["constructed", "unsupported", "failed"]
    mode: Literal["token_equivalent", "semantic_replay"] | None = None
    path: str | None = None
    roundtrip_exact: bool | None = None
    note: str | None = None
    error: str | None = None


class ApiInferenceResult(BaseModel):
    """样例最终落盘的结构化 API 推理结果。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["o5_fc_api_inference_result.v1"] = (
        "o5_fc_api_inference_result.v1"
    )
    data_id: str | None
    base_url: str
    session_created: JsonObject
    committed_unit_indices: list[int]
    think_texts: list[str]
    tool_calls: list[JsonObject]
    spoken_turns: list[SpokenTurnArtifact]
    warnings: list[JsonObject]
    errors: list[JsonObject]
    source_ai_tracks_sent_to_api: Literal[False] = False
    source_ai_spoken_segment_count: int = Field(ge=0)
    source_ai_non_spoken_segment_count: int = Field(ge=0)
    source_input_event_count: int = Field(ge=0)
    replay_training_data: ReplayTrainingDataArtifact


class ApiInferenceClientError(RuntimeError):
    """带部分 history 的 API 调用错误。"""

    def __init__(
        self,
        message: str,
        *,
        history: list[ApiHistoryEntry],
    ) -> None:
        super().__init__(message)
        self.history = history


class ApiInferenceClient:
    """只负责发送 TrainingData 输入并原样采集 Semantic API 事件。"""

    def __init__(
        self,
        *,
        base_url: str,
        insecure: bool = False,
        queue_timeout_sec: float = 120.0,
        event_timeout_sec: float = 180.0,
        close_timeout_sec: float = 5.0,
    ) -> None:
        """初始化 API 客户端。

        参数:
            base_url: Gateway HTTP(S) 根地址。
            insecure: 是否跳过 WSS 证书校验。
            queue_timeout_sec: 等待 Worker 分配的超时。
            event_timeout_sec: 等待单个 API 事件的超时。
            close_timeout_sec: 关闭 Session 的等待超时。
        """

        self.base_url = base_url
        self.insecure = insecure
        self.queue_timeout_sec = queue_timeout_sec
        self.event_timeout_sec = event_timeout_sec
        self.close_timeout_sec = close_timeout_sec

    async def infer(
        self,
        scenario: ApiInferenceScenario,
    ) -> ApiInferenceSession:
        """执行一条完整的 TrainingData 输入会话。

        参数:
            scenario: 只含请求侧语义和用户音频 Unit 的推理场景。

        返回:
            完整双向 history、parser history 和已提交 Unit。
        """

        history: list[ApiHistoryEntry] = []
        protocol_history: list[JsonObject] = []
        session_created: JsonObject = {}
        committed_units: list[int] = []
        ssl_context = (
            ssl._create_unverified_context() if self.insecure else None
        )
        try:
            async with websockets.connect(
                self._websocket_url(),
                ssl=ssl_context,
                max_size=128 * 1024 * 1024,
                open_timeout=self.queue_timeout_sec,
            ) as websocket:
                await self._wait_queue_done(websocket, history)
                await self._send(
                    websocket,
                    history,
                    protocol_history,
                    scenario.session_init,
                )
                session_created = await self._wait_session_created(
                    websocket,
                    history,
                    protocol_history,
                )
                for unit in scenario.units:
                    await self._send(
                        websocket,
                        history,
                        protocol_history,
                        unit.to_frame(),
                    )
                    await self._consume_until_committed(
                        websocket,
                        history,
                        protocol_history,
                        target_unit_index=unit.unit_index,
                    )
                    committed_units.append(unit.unit_index)
                await self._send(
                    websocket,
                    history,
                    protocol_history,
                    {"type": "session.close", "reason": "input_complete"},
                )
                await self._drain_close(
                    websocket,
                    history,
                    protocol_history,
                )
        except ApiInferenceClientError:
            raise
        except Exception as exc:
            raise ApiInferenceClientError(
                f"Semantic API 调用失败: {type(exc).__name__}: {exc}",
                history=history,
            ) from exc

        return ApiInferenceSession(
            history=history,
            protocol_history=protocol_history,
            session_created=session_created,
            committed_unit_indices=committed_units,
        )

    def _websocket_url(self) -> str:
        """把 HTTP(S) Gateway 地址转换为 audio Realtime WebSocket 地址。"""

        parsed = urlsplit(
            self.base_url.rstrip("/") + "/v1/realtime?mode=audio"
        )
        return urlunsplit(
            (
                "wss" if parsed.scheme == "https" else "ws",
                parsed.netloc,
                parsed.path,
                parsed.query,
                "",
            )
        )

    async def _wait_queue_done(
        self,
        websocket: Any,
        history: list[ApiHistoryEntry],
    ) -> None:
        """等待 Gateway 分配 Worker。"""

        while True:
            event = await self._receive(
                websocket,
                history,
                timeout_sec=self.queue_timeout_sec,
            )
            event_type = str(event.get("type") or "")
            if event_type == "session.queue_done":
                return
            self._raise_on_terminal_event(event, history)

    async def _wait_session_created(
        self,
        websocket: Any,
        history: list[ApiHistoryEntry],
        protocol_history: list[JsonObject],
    ) -> JsonObject:
        """等待后端完成 Session 初始化。"""

        while True:
            event = await self._receive(
                websocket,
                history,
                timeout_sec=self.event_timeout_sec,
            )
            protocol_history.append(dict(event))
            if event.get("type") == "session.created":
                return event
            self._raise_on_terminal_event(event, history)

    async def _consume_until_committed(
        self,
        websocket: Any,
        history: list[ApiHistoryEntry],
        protocol_history: list[JsonObject],
        *,
        target_unit_index: int,
    ) -> None:
        """消费一个 Unit 的全部事件直到 committed。"""

        while True:
            event = await self._receive(
                websocket,
                history,
                timeout_sec=self.event_timeout_sec,
            )
            protocol_history.append(dict(event))
            self._raise_on_terminal_event(event, history)
            if event.get("type") != "response.unit.committed":
                continue
            actual_unit = int(event.get("unit_index", -1))
            if actual_unit != target_unit_index:
                raise ApiInferenceClientError(
                    "committed Unit 顺序错误: "
                    f"expected={target_unit_index}, actual={actual_unit}",
                    history=history,
                )
            return

    async def _drain_close(
        self,
        websocket: Any,
        history: list[ApiHistoryEntry],
        protocol_history: list[JsonObject],
    ) -> None:
        """发送 close 后尽力等待 Session 关闭事件。"""

        while True:
            try:
                event = await self._receive(
                    websocket,
                    history,
                    timeout_sec=self.close_timeout_sec,
                )
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                return
            protocol_history.append(dict(event))
            if event.get("type") == "session.closed":
                return

    async def _send(
        self,
        websocket: Any,
        history: list[ApiHistoryEntry],
        protocol_history: list[JsonObject],
        frame: JsonObject,
    ) -> None:
        """发送并记录一条上行事件。"""

        history.append(
            ApiHistoryEntry(
                sequence=len(history),
                direction="up",
                event=dict(frame),
            )
        )
        protocol_history.append(dict(frame))
        await websocket.send(json.dumps(frame, ensure_ascii=False))

    async def _receive(
        self,
        websocket: Any,
        history: list[ApiHistoryEntry],
        *,
        timeout_sec: float,
    ) -> JsonObject:
        """接收、校验并记录一条下行事件。"""

        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise ApiInferenceClientError(
                f"API 事件必须是 JSON object: {event!r}",
                history=history,
            )
        history.append(
            ApiHistoryEntry(
                sequence=len(history),
                direction="down",
                event=dict(event),
            )
        )
        return event

    @staticmethod
    def _raise_on_terminal_event(
        event: JsonObject,
        history: list[ApiHistoryEntry],
    ) -> None:
        """把 API error 或提前关闭转换为带 history 的异常。"""

        if event.get("type") in {"error", "session.closed"}:
            raise ApiInferenceClientError(
                f"API Session 提前结束: {event}",
                history=history,
            )


def load_training_data_structure(
    path: Path,
    *,
    line_index: int,
) -> JsonObject:
    """读取单 JSON 或 JSONL 中指定的一条 TrainingData 结构。

    参数:
        path: TrainingData JSON/JSONL 文件。
        line_index: JSONL 的零起始行号；单 JSON 只允许 0。

    返回:
        TrainingData JSON object。
    """

    if line_index < 0:
        raise ValueError("line_index 必须大于等于 0")
    if path.suffix.lower() != ".jsonl":
        if line_index != 0:
            raise IndexError("单 JSON 文件只支持 line_index=0")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("TrainingData JSON 必须是 object")
        return value

    with path.open("r", encoding="utf-8") as file:
        for current_index, line in enumerate(file):
            if current_index != line_index:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"TrainingData JSONL line_index={line_index} 必须是 object"
                )
            return value
    raise IndexError(f"TrainingData JSONL 不存在 line_index={line_index}")


def build_session_init_payload(
    *,
    system: FcSystemContentInput,
    unit_policy: O5UnitPolicy,
    tts_prompt_audio_path: str | None,
) -> JsonObject:
    """构造不包含评测信息的 FC Semantic API Session 初始化帧。

    参数:
        system: 保持 TrainingData segment 顺序和嵌套 tools 的 v3 system。
        unit_policy: TrainingData 显式 UnitPolicy。
        tts_prompt_audio_path: 独立 TTS prompt；默认由调用方选择首段 system audio。

    返回:
        可直接发送的 ``session.init`` 帧。
    """

    payload: JsonObject = {
        "mode": "full_duplex",
        "fc_duplex": True,
        "protocol_version": PROTOCOL_VERSION,
        "tokenizer_target": "o5",
        "system": system.model_dump(mode="json"),
        "generate_audio": True,
        "unit_policy": unit_policy.model_dump(mode="json"),
        "config": {
            "runtime": "fc_duplex",
            "decode_mode": "greedy",
            "sample_rate": AUDIO_SAMPLE_RATE,
        },
    }
    if tts_prompt_audio_path is not None:
        payload["tts_prompt_audio"] = {
            "source": "path",
            "file_path": tts_prompt_audio_path,
        }
    return {"type": "session.init", "payload": payload}


def build_api_inference_scenario(
    *,
    training_data: O5DuplexTrainingData,
    data_root: Path,
) -> ApiInferenceScenario:
    """把 TrainingData 投影为不携带 AI Ground Truth 的 API 请求。

    参数:
        training_data: 已绑定媒体 loader 的 TrainingData。
        data_root: 相对媒体路径的解析根目录。

    返回:
        请求帧、用户音频 Unit 与可选 replay scaffold。
    """

    if (
        training_data.tracks.user_video is not None
        and training_data.tracks.user_video.segments
    ):
        raise ValueError("当前样例只支持 user_audio，不支持 user_video")

    tokenize_result = training_data.tokenize(tokenizer_id=O5TokenizerID.O5)
    tokenized = tokenize_result.tokenized_data
    audio_tensor = build_user_audio_tensor_from_arrangement(
        tokenize_result.arrangement
    )
    unit_count = len(tokenized.unit_spans)
    samples_per_unit = round(
        training_data.unit_policy.unit_sec * AUDIO_SAMPLE_RATE
    )
    if audio_tensor is None:
        unit_samples = torch.zeros(
            (unit_count, samples_per_unit),
            dtype=torch.float32,
        )
    else:
        if audio_tensor.num_units != unit_count:
            raise ValueError(
                "用户音频 Unit 数与 tokenized Unit 数不一致: "
                f"audio={audio_tensor.num_units}, tokenized={unit_count}"
            )
        unit_samples = audio_tensor.units

    system_input, tts_prompt_audio_path = _project_system(
        training_data.system,
        data_root=data_root,
    )
    units = [
        ApiUnitInput(
            unit_index=unit_index,
            input_id=f"training_data_unit_{unit_index:06d}",
            audio_base64=base64.b64encode(
                unit_samples[unit_index].contiguous().numpy().tobytes()
            ).decode("ascii"),
        )
        for unit_index in range(unit_count)
    ]
    return ApiInferenceScenario(
        data_id=training_data.data_id,
        unit_policy=training_data.unit_policy,
        system=training_data.system,
        session_init=build_session_init_payload(
            system=system_input,
            unit_policy=training_data.unit_policy,
            tts_prompt_audio_path=tts_prompt_audio_path,
        ),
        units=units,
        user_audio_samples=unit_samples.reshape(-1).numpy().copy(),
        scaffold_token_ids=list(tokenized.input_ids),
        decode_spans=_extract_decode_spans(tokenized.token_provenance),
        source_arrangement=tokenize_result.arrangement,
        tokenizer_fingerprint={
            "vocab_hash": tokenized.tokenizer_fingerprint.vocab_hash,
            "merges_hash": tokenized.tokenizer_fingerprint.merges_hash,
        },
        source_ai_spoken_segment_count=len(
            training_data.tracks.ai_spoken.segments
        ),
        source_ai_non_spoken_segment_count=len(
            training_data.tracks.ai_non_spoken.segments
        ),
        source_input_event_count=len(
            training_data.tracks.input_event.segments
        ),
    )


def _project_system(
    system: O5SystemContent | None,
    *,
    data_root: Path,
) -> tuple[FcSystemContentInput, str | None]:
    """把 SDK system 无损投影为 v3 wire system，并选取首段音频作为 TTS prompt。"""

    if system is None:
        return FcSystemContentInput(), None
    segments: list[FcSystemTextInput | FcSystemAudioInput] = []
    tts_prompt_audio_path: str | None = None
    for segment in system.segments:
        if isinstance(segment, O5SystemTextSegment):
            segments.append(FcSystemTextInput(text=segment.text))
            continue
        if not isinstance(segment, O5SystemAudioSegment):
            raise TypeError(f"不支持的 system segment: {type(segment).__name__}")
        file_path = segment.audio.file_path
        if file_path is None:
            raise ValueError("system reference audio 缺少 file_path")
        resolved_path = str((data_root / file_path).resolve())
        segments.append(
            FcSystemAudioInput(
                audio=FcAudioPathInput(file_path=resolved_path)
            )
        )
        if tts_prompt_audio_path is None:
            tts_prompt_audio_path = resolved_path
    return (
        FcSystemContentInput(
            segments=segments,
            tools=list(system.tools),
        ),
        tts_prompt_audio_path,
    )


def _extract_decode_spans(
    provenance: list[O5TokenProvenance],
) -> list[ApiDecodeSpan]:
    """提取每个 Unit 两条 AI decode 轨在完整 scaffold 中的连续区间。"""

    spans: list[ApiDecodeSpan] = []
    active_key: tuple[int, str] | None = None
    active_start = 0
    for index, item in enumerate([*provenance, None]):
        key: tuple[int, str] | None = None
        if (
            item is not None
            and item.track in {"ai_spoken", "ai_non_spoken"}
            and (
                item.trainable
                or (
                    item.track == "ai_non_spoken"
                    and item.token_text
                    == "<|non_spoken_budget_reached|>"
                )
            )
        ):
            key = (item.unit_index, item.track)
        if key == active_key:
            continue
        if active_key is not None:
            spans.append(
                ApiDecodeSpan(
                    unit_index=active_key[0],
                    track=(
                        "spoken"
                        if active_key[1] == "ai_spoken"
                        else "non_spoken"
                    ),
                    start_token_index=active_start,
                    end_token_index_exclusive=index,
                )
            )
        active_key = key
        active_start = index

    duplicate_keys = [
        key
        for key in {(span.unit_index, span.track) for span in spans}
        if sum(
            (span.unit_index, span.track) == key
            for span in spans
        )
        != 1
    ]
    if duplicate_keys:
        raise ValueError(
            "同一 Unit/track 的 API decode scaffold 不是单一连续区间: "
            f"{duplicate_keys}"
        )
    return spans


def splice_api_decode_tracks(
    *,
    scaffold_token_ids: list[int],
    decode_spans: list[ApiDecodeSpan],
    actual_decode_tokens: Mapping[tuple[int, DecodeTrack], list[int]],
) -> list[int]:
    """用 API 实际 decode token 替换输入 scaffold 的 AI 输出区间。

    参数:
        scaffold_token_ids: 由输入 TrainingData 确定的完整协议 scaffold。
        decode_spans: 每个 Unit/track 的 AI decode 区间。
        actual_decode_tokens: API history 重建出的实际 spoken/non-spoken token。

    返回:
        可交给 SDK parser 的完整预测 token 序列。
    """

    reconstructed: list[int] = []
    cursor = 0
    for span in sorted(
        decode_spans,
        key=lambda item: item.start_token_index,
    ):
        key = (span.unit_index, span.track)
        if key not in actual_decode_tokens:
            raise ValueError(
                f"API history 缺少 decode 轨: unit={key[0]}, track={key[1]}"
            )
        if span.start_token_index < cursor:
            raise ValueError(f"decode span 重叠: {span}")
        reconstructed.extend(
            scaffold_token_ids[cursor : span.start_token_index]
        )
        reconstructed.extend(actual_decode_tokens[key])
        cursor = span.end_token_index_exclusive
    reconstructed.extend(scaffold_token_ids[cursor:])
    return reconstructed


def materialize_response_artifacts(
    *,
    history: list[ApiHistoryEntry],
    output_dir: Path,
) -> ResponseArtifacts:
    """从原始 history 提取语义结果并保存真实 AI 音频。

    参数:
        history: 完整双向 API history。
        output_dir: 本次推理输出目录。

    返回:
        think、tool-call、spoken、warning/error 与 16k replay 音频。
    """

    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    _write_history_jsonl(
        output_dir / "response_history.jsonl",
        [item for item in history if item.direction == "down"],
    )

    think_texts: list[str] = []
    tool_calls: list[JsonObject] = []
    warnings: list[JsonObject] = []
    errors: list[JsonObject] = []
    spoken_turns: list[SpokenTurnArtifact] = []
    spoken_audio_16k: list[np.ndarray] = []
    active_text_parts: list[str] = []
    active_audio_parts: list[np.ndarray] = []
    active_sample_rate: int | None = None
    active_start_unit: int | None = None

    def finish_turn(event: JsonObject | None = None) -> None:
        """把当前 spoken turn 写盘并追加结构化结果。"""

        nonlocal active_text_parts
        nonlocal active_audio_parts
        nonlocal active_sample_rate
        nonlocal active_start_unit

        if not active_text_parts and not active_audio_parts:
            return
        turn_index = len(spoken_turns)
        text = (
            str(event.get("full_text") or "")
            if event is not None
            else ""
        ) or "".join(active_text_parts)
        audio_path: str | None = None
        replay_audio_path: str | None = None
        if active_audio_parts:
            sample_rate = active_sample_rate or OUTPUT_AUDIO_SAMPLE_RATE
            audio = np.concatenate(active_audio_parts).astype(
                np.float32,
                copy=False,
            )
            original_path = (
                media_dir / f"ai_spoken_turn_{turn_index:03d}.wav"
            )
            _write_float_wav(
                original_path,
                samples=audio,
                sample_rate=sample_rate,
            )
            audio_16k = _resample_audio(
                audio,
                source_sample_rate=sample_rate,
                target_sample_rate=AUDIO_SAMPLE_RATE,
            )
            replay_path = (
                media_dir
                / f"ai_spoken_turn_{turn_index:03d}_16k.wav"
            )
            _write_float_wav(
                replay_path,
                samples=audio_16k,
                sample_rate=AUDIO_SAMPLE_RATE,
            )
            audio_path = str(original_path.relative_to(output_dir))
            replay_audio_path = str(replay_path.relative_to(output_dir))
            spoken_audio_16k.append(audio_16k)
        spoken_turns.append(
            SpokenTurnArtifact(
                turn_index=turn_index,
                start_unit_index=active_start_unit,
                end_unit_index=(
                    int(event["unit_index"])
                    if event is not None
                    and event.get("unit_index") is not None
                    else None
                ),
                text=text,
                sample_rate=(
                    active_sample_rate
                    if active_audio_parts
                    else None
                ),
                audio_path=audio_path,
                replay_audio_path=replay_audio_path,
            )
        )
        active_text_parts = []
        active_audio_parts = []
        active_sample_rate = None
        active_start_unit = None

    for entry in history:
        if entry.direction != "down":
            continue
        event = entry.event
        event_type = str(event.get("type") or "")
        if event_type == "response.think.end":
            think_texts.append(str(event.get("full_text") or ""))
        elif event_type == "response.tool_call.done":
            tool_calls.append(dict(event))
        elif event_type == "response.warning":
            warnings.append(dict(event))
        elif event_type == "error":
            errors.append(dict(event))
        elif event_type == "response.spoken.delta":
            if active_start_unit is None and event.get("unit_index") is not None:
                active_start_unit = int(event["unit_index"])
            for step in event.get("steps") or []:
                if isinstance(step, Mapping) and step.get("kind") == "text":
                    active_text_parts.append(str(step.get("text") or ""))
            encoded_audio = event.get("audio")
            if isinstance(encoded_audio, str) and encoded_audio:
                sample_rate = int(
                    event.get("sample_rate") or OUTPUT_AUDIO_SAMPLE_RATE
                )
                if (
                    active_sample_rate is not None
                    and active_sample_rate != sample_rate
                ):
                    raise ValueError(
                        "同一 spoken turn 出现不同 sample_rate: "
                        f"{active_sample_rate} vs {sample_rate}"
                    )
                active_sample_rate = sample_rate
                active_audio_parts.append(
                    np.frombuffer(
                        base64.b64decode(encoded_audio),
                        dtype=np.float32,
                    ).copy()
                )
        elif (
            event_type == "response.spoken.end"
            and event.get("reason") in {"turn_eos", "listen"}
        ):
            finish_turn(event)
    if active_text_parts or active_audio_parts:
        finish_turn()

    return ResponseArtifacts(
        think_texts=think_texts,
        tool_calls=tool_calls,
        spoken_turns=spoken_turns,
        warnings=warnings,
        errors=errors,
        spoken_audio_16k=spoken_audio_16k,
    )


def assemble_replay_training_data(
    *,
    scenario: ApiInferenceScenario,
    session: ApiInferenceSession,
    response_artifacts: ResponseArtifacts,
    output_dir: Path,
) -> ReplayTrainingDataArtifact:
    """尽力把 API history 组装成 replay TrainingData。

    参数:
        scenario: 请求侧 TrainingData 投影和 token scaffold。
        session: API 完整语义 history。
        response_artifacts: 已保存的真实 AI 音频。
        output_dir: 本次推理输出目录。

    返回:
        构造成功、当前不支持或失败的显式状态。
    """

    if not session.committed_unit_indices:
        return ReplayTrainingDataArtifact(
            status="unsupported",
            error="API 没有已提交 Unit，无法构造完整 replay TrainingData",
        )
    token_equivalent_error: str | None = None
    if not scenario.source_input_event_count:
        try:
            return _assemble_token_equivalent_replay(
                scenario=scenario,
                session=session,
                response_artifacts=response_artifacts,
                output_dir=output_dir,
            )
        except Exception as exc:
            token_equivalent_error = f"{type(exc).__name__}: {exc}"
    try:
        replay = _build_semantic_replay_training_data(
            scenario=scenario,
            session=session,
            response_artifacts=response_artifacts,
        )
        replay_path = output_dir / "replay_training_data.json"
        replay_path.write_text(
            json.dumps(
                replay.dump_structure(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return ReplayTrainingDataArtifact(
            status="constructed",
            mode="semantic_replay",
            path=str(replay_path.relative_to(output_dir)),
            note=(
                "该结构按实际 API semantic events 与音频构造，不承诺 token exact，"
                "也不恢复原始 timing constraint 或未发送的 input_event。"
                + (
                    f" token-equivalent 构造失败原因: {token_equivalent_error}"
                    if token_equivalent_error
                    else ""
                )
            ),
        )
    except Exception as exc:
        return ReplayTrainingDataArtifact(
            status="failed",
            error=(
                f"{type(exc).__name__}: {exc}"
                + (
                    f"; token-equivalent 构造失败原因: {token_equivalent_error}"
                    if token_equivalent_error
                    else ""
                )
            ),
        )


def _assemble_token_equivalent_replay(
    *,
    scenario: ApiInferenceScenario,
    session: ApiInferenceSession,
    response_artifacts: ResponseArtifacts,
    output_dir: Path,
) -> ReplayTrainingDataArtifact:
    """通过 resume canonicalizer 和 SDK parser 构造 token-equivalent replay。"""

    identity = session.session_created.get("resume")
    if not isinstance(identity, Mapping):
        identity = session.session_created
    model = str(identity.get("model") or "")
    tokenizer_target = str(identity.get("tokenizer_target") or "o5")
    fingerprint = identity.get("tokenizer_fingerprint")
    if not model or tokenizer_target != "o5" or not isinstance(
        fingerprint,
        Mapping,
    ):
        raise ValueError(
            "session.created 缺少 replay 所需的 model/o5 tokenizer fingerprint"
        )
    plan = build_fc_duplex_resume_plan(
        protocol_version=PROTOCOL_VERSION,
        model=model,
        tokenizer_target="o5",
        tokenizer_fingerprint={
            str(key): str(value)
            for key, value in fingerprint.items()
        },
        through_unit_index=max(session.committed_unit_indices),
        history=session.protocol_history,
    )
    actual_decode_tokens: dict[
        tuple[int, DecodeTrack],
        list[int],
    ] = {}
    for unit in plan.units:
        actual_decode_tokens[(unit.unit_index, "spoken")] = list(
            unit.spoken_token_ids
        )
        actual_decode_tokens[(unit.unit_index, "non_spoken")] = list(
            unit.non_spoken_token_ids
        )
    reconstructed_ids = splice_api_decode_tracks(
        scaffold_token_ids=scenario.scaffold_token_ids,
        decode_spans=scenario.decode_spans,
        actual_decode_tokens=actual_decode_tokens,
    )
    media_bundle = _build_replay_media_bundle(
        scenario=scenario,
        response_artifacts=response_artifacts,
    )
    parser = O5DuplexParser(tokenizer_id=O5TokenizerID.O5)
    request = O5DuplexParseRequest(
        sequence=reconstructed_ids,
        sequence_format="token_ids",
        unit_policy=scenario.unit_policy,
        media_bundle=media_bundle,
        data_id=_inference_data_id(scenario.data_id),
        system=scenario.system,
    )
    replay = parser.parse(request)
    roundtrip = assert_duplex_parse_roundtrip(parser, request)
    replay_path = output_dir / "replay_training_data.json"
    replay_path.write_text(
        json.dumps(
            replay.dump_structure(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReplayTrainingDataArtifact(
        status="constructed",
        mode="token_equivalent",
        path=str(replay_path.relative_to(output_dir)),
        roundtrip_exact=roundtrip.exact_match,
    )


def _build_semantic_replay_training_data(
    *,
    scenario: ApiInferenceScenario,
    session: ApiInferenceSession,
    response_artifacts: ResponseArtifacts,
) -> O5DuplexTrainingData:
    """按实际 semantic events 构造不声称 token exact 的逻辑 replay。"""

    user_audio_track = _build_replay_user_audio_track(scenario)
    spoken_segments: list[O5AISpokenSegment] = []
    for turn, samples_array in zip(
        response_artifacts.spoken_turns,
        response_artifacts.spoken_audio_16k,
        strict=False,
    ):
        if (
            turn.replay_audio_path is None
            or turn.start_unit_index is None
            or samples_array.size == 0
        ):
            continue
        samples = torch.from_numpy(samples_array.copy()).to(
            dtype=torch.float32
        )
        duration_sec = samples.numel() / AUDIO_SAMPLE_RATE
        alignment = O5Alignment(
            word_intervals=(
                [
                    O5WordInterval(
                        text=turn.text,
                        start_sec=0.0,
                        end_sec=duration_sec,
                    )
                ]
                if turn.text
                else []
            )
        )
        spoken_segments.append(
            O5AISpokenSegment(
                audio=O5LazyAudio(
                    duration_sec=duration_sec,
                    get_tensor_fn=lambda value=samples: value.clone(),
                    file_path=turn.replay_audio_path,
                ),
                text=turn.text,
                alignment=alignment,
                start_trigger=_unit_global_trigger(
                    unit_index=turn.start_unit_index,
                    unit_policy=scenario.unit_policy,
                ),
            )
        )

    non_spoken_segments: list[O5AINonSpokenSegment] = []
    for entry in session.history:
        if entry.direction != "down":
            continue
        event = entry.event
        event_type = str(event.get("type") or "")
        if event_type == "response.think.end":
            text = str(event.get("full_text") or "")
            if text:
                non_spoken_segments.append(
                    O5AINonSpokenSegment(
                        content=O5ThinkContent(text=text),
                        start_trigger=_unit_global_trigger(
                            unit_index=int(event.get("unit_index") or 0),
                            unit_policy=scenario.unit_policy,
                        ),
                    )
                )
        elif event_type == "response.tool_call.done":
            call = event.get("call")
            if not isinstance(call, Mapping):
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                arguments = {}
            non_spoken_segments.append(
                O5AINonSpokenSegment(
                    content=O5ToolCallContent(
                        tool_call_id=str(
                            event.get("tool_call_id") or ""
                        ),
                        name=str(call.get("name") or ""),
                        arguments=dict(arguments),
                    ),
                    start_trigger=_unit_global_trigger(
                        unit_index=int(event.get("unit_index") or 0),
                        unit_policy=scenario.unit_policy,
                    ),
                )
            )

    return O5DuplexTrainingData(
        data_id=_inference_data_id(scenario.data_id),
        system=scenario.system,
        unit_policy=scenario.unit_policy,
        tracks=O5DuplexTrainingTracks(
            user_video=None,
            user_audio=user_audio_track,
            input_event=O5InputEventTrack(),
            ai_spoken=O5AISpokenTrack(segments=spoken_segments),
            ai_non_spoken=O5AINonSpokenTrack(
                segments=non_spoken_segments
            ),
        ),
    )


def _build_replay_user_audio_track(
    scenario: ApiInferenceScenario,
) -> O5UserAudioTrack | None:
    """把实际发送的 source user audio 位置改写为全局时间锚点。"""

    arranged_track = scenario.source_arrangement.tracks.user_audio
    if arranged_track is None:
        return None
    return O5UserAudioTrack(
        segments=[
            O5UserAudioSegment(
                audio=segment.audio,
                transcript=segment.transcript,
                alignment=segment.alignment,
                start_trigger=O5StartTrigger(
                    refs=[
                        O5GlobalTime(
                            offset_sec=segment.timeline.timeline_start_sec
                        )
                    ]
                ),
            )
            for segment in arranged_track.segments
        ],
        total_duration_sec=(
            scenario.user_audio_samples.size / AUDIO_SAMPLE_RATE
        ),
    )


def _unit_global_trigger(
    *,
    unit_index: int,
    unit_policy: O5UnitPolicy,
) -> O5StartTrigger:
    """把 API 输出 Unit 转换为不推断因果关系的全局时间锚点。"""

    return O5StartTrigger(
        refs=[
            O5GlobalTime(
                offset_sec=unit_index * unit_policy.unit_sec
            )
        ]
    )


def _inference_data_id(source_data_id: str | None) -> str:
    """生成明确标识 API 推理 replay 的 data_id。"""

    return (
        f"{source_data_id}__api_inference"
        if source_data_id
        else "api_inference"
    )


def _build_replay_media_bundle(
    *,
    scenario: ApiInferenceScenario,
    response_artifacts: ResponseArtifacts,
) -> O5ReplayMediaBundle:
    """用实际发送和接收的音频构造 SDK replay media bundle。"""

    user_samples = torch.from_numpy(
        scenario.user_audio_samples.copy()
    ).to(dtype=torch.float32)
    user_audio = O5LazyAudio(
        duration_sec=user_samples.numel() / AUDIO_SAMPLE_RATE,
        get_tensor_fn=lambda samples=user_samples: samples.clone(),
        file_path="media/user_audio.wav",
    )
    ai_audio: list[O5LazyAudio] = []
    for turn, samples_array in zip(
        response_artifacts.spoken_turns,
        response_artifacts.spoken_audio_16k,
        strict=False,
    ):
        if turn.replay_audio_path is None:
            continue
        samples = torch.from_numpy(samples_array.copy()).to(
            dtype=torch.float32
        )
        ai_audio.append(
            O5LazyAudio(
                duration_sec=samples.numel() / AUDIO_SAMPLE_RATE,
                get_tensor_fn=lambda value=samples: value.clone(),
                file_path=turn.replay_audio_path,
            )
        )
    return O5ReplayMediaBundle(
        user_audio=user_audio,
        ai_spoken_audio_by_turn=ai_audio,
    )


def _write_float_wav(
    path: Path,
    *,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    """把 float32 单声道音频写成标准 PCM16 WAV。"""

    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def _resample_audio(
    samples: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """重采样 float32 单声道音频。"""

    if source_sample_rate == target_sample_rate:
        return samples.astype(np.float32, copy=True)
    tensor = torch.from_numpy(samples.copy()).to(dtype=torch.float32)
    return (
        audio_functional.resample(
            tensor,
            source_sample_rate,
            target_sample_rate,
        )
        .contiguous()
        .numpy()
    )


def _write_history_jsonl(
    path: Path,
    history: list[ApiHistoryEntry],
) -> None:
    """按 sequence 顺序写入可审计 JSONL history。"""

    path.write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n"
            for item in history
        ),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """构造纯 API 推理样例命令行。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_data", type=Path)
    parser.add_argument("--line-index", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成请求预览和 canonical user audio，不连接 API",
    )
    return parser


def _write_request_preview(
    *,
    scenario: ApiInferenceScenario,
    output_dir: Path,
) -> None:
    """写入不包含大体积 base64 音频的请求预览。"""

    preview = {
        "data_id": scenario.data_id,
        "session_init": scenario.session_init,
        "unit_count": len(scenario.units),
        "units": [
            {
                "unit_index": unit.unit_index,
                "input_id": unit.input_id,
                "sample_rate": unit.sample_rate,
                "audio_byte_count": len(
                    base64.b64decode(unit.audio_base64)
                ),
            }
            for unit in scenario.units
        ],
        "source_ai_tracks_sent_to_api": False,
        "source_ai_spoken_segment_count": (
            scenario.source_ai_spoken_segment_count
        ),
        "source_ai_non_spoken_segment_count": (
            scenario.source_ai_non_spoken_segment_count
        ),
        "source_input_event_count": scenario.source_input_event_count,
    }
    (output_dir / "request_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    """执行请求准备、API 推理、结果落盘和可选 replay 构造。

    参数:
        args: CLI 参数。

    返回:
        最终 ``inference_result.json`` 或 prepare-only 预览路径。
    """

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else args.training_data.expanduser().resolve().parent
    )
    structure = load_training_data_structure(
        args.training_data,
        line_index=args.line_index,
    )
    training_data = O5DuplexTrainingData.load_structure(
        structure,
        data_root=data_root,
    )
    scenario = build_api_inference_scenario(
        training_data=training_data,
        data_root=data_root,
    )
    (output_dir / "input_training_data.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_request_preview(scenario=scenario, output_dir=output_dir)
    _write_float_wav(
        output_dir / "media" / "user_audio.wav",
        samples=scenario.user_audio_samples,
        sample_rate=AUDIO_SAMPLE_RATE,
    )
    if args.prepare_only:
        preview_path = output_dir / "request_preview.json"
        print(preview_path)
        return preview_path

    client = ApiInferenceClient(
        base_url=args.base_url,
        insecure=args.insecure,
    )
    try:
        session = asyncio.run(client.infer(scenario))
    except ApiInferenceClientError as exc:
        _write_history_jsonl(
            output_dir / "history.jsonl",
            exc.history,
        )
        error_path = output_dir / "inference_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "schema_version": "o5_fc_api_inference_error.v1",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    _write_history_jsonl(output_dir / "history.jsonl", session.history)
    response_artifacts = materialize_response_artifacts(
        history=session.history,
        output_dir=output_dir,
    )
    replay_artifact = assemble_replay_training_data(
        scenario=scenario,
        session=session,
        response_artifacts=response_artifacts,
        output_dir=output_dir,
    )
    result = ApiInferenceResult(
        data_id=scenario.data_id,
        base_url=args.base_url,
        session_created=session.session_created,
        committed_unit_indices=session.committed_unit_indices,
        think_texts=response_artifacts.think_texts,
        tool_calls=response_artifacts.tool_calls,
        spoken_turns=response_artifacts.spoken_turns,
        warnings=response_artifacts.warnings,
        errors=response_artifacts.errors,
        source_ai_spoken_segment_count=(
            scenario.source_ai_spoken_segment_count
        ),
        source_ai_non_spoken_segment_count=(
            scenario.source_ai_non_spoken_segment_count
        ),
        source_input_event_count=scenario.source_input_event_count,
        replay_training_data=replay_artifact,
    )
    result_path = output_dir / "inference_result.json"
    result_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    print(result_path)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""

    args = build_argument_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(
            f"[fc-api-inference] error: {type(exc).__name__}: {exc}",
            file=__import__("sys").stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
