"""把 SDK 0.0.5 TrainingData 投影为 Semantic API v3 确定性请求场景。

该投影只消费 SDK public API：GT 来自 ``training_data.tokenize``，用户音频来自
``build_user_audio_tensor``；工具调用和 response 调度读取 arrangement 的离散 Unit
字段，不从兼容 timeline 秒值反推。
"""

from __future__ import annotations

import base64
from pathlib import Path

from minicpm_o5_sdk import (
    O5DuplexTrainingData,
    O5SystemTextSegment,
    O5TextContent,
    O5TokenizerID,
    O5ToolCallContent,
    O5ToolResponseEvent,
    OpenAIToolDefinition,
)
from minicpm_o5_sdk.protocols.duplex import (
    O5DuplexArrangement,
    O5TokenProvenance,
    O5UserAudioTensor,
)
from minicpm_o5_sdk.protocols.duplex.training_data import O5SystemAudioSegment

from core.fc_duplex.system_input import (
    FcAudioPathInput,
    FcSystemAudioInput,
    FcSystemContentInput,
    FcSystemTextInput,
)
from .models import (
    FcApiCheckpointProfile,
    FcApiDecodeSpan,
    FcApiExpectedDecodeUnit,
    FcApiExpectedToolCall,
    FcApiToolResponseSchedule,
    FcApiTrainingDataScenario,
    FcApiUnitInput,
    JsonObject,
)


REQUIRED_SDK_VERSION = "0.0.5"


def build_training_data_scenario(
    *,
    training_data: O5DuplexTrainingData,
    profile: FcApiCheckpointProfile,
    data_root: Path,
) -> FcApiTrainingDataScenario:
    """从 TrainingData 构造完整 Semantic API v3 请求与 GT。

    参数:
        training_data: 已加载媒体 loader 的 SDK TrainingData。
        profile: API endpoint 当前部署的 checkpoint/Profile 元信息。
        data_root: 相对媒体路径的解析根目录。

    返回:
        包含 session.init、逐 Unit 音频、工具调度和 provenance GT 的场景。

    异常:
        ValueError: SDK 版本、profile target、UnitPolicy 或输入事件不受支持。
    """

    tokenizer_id = O5TokenizerID(profile.tokenizer_target)
    tokenize_result = training_data.tokenize(
        tokenizer_id=tokenizer_id,
        tool_serializer_name=profile.tool_serializer_name,
    )
    tokenized = tokenize_result.tokenized_data
    if tokenized.minicpm_o5_sdk_version != REQUIRED_SDK_VERSION:
        raise ValueError(
            "sdk_version_mismatch: "
            f"expected={REQUIRED_SDK_VERSION}, "
            f"actual={tokenized.minicpm_o5_sdk_version}"
        )
    if profile.sdk_version != REQUIRED_SDK_VERSION:
        raise ValueError(
            "sdk_version_mismatch: "
            f"profile={profile.sdk_version}, required={REQUIRED_SDK_VERSION}"
        )
    if tokenized.tokenizer_target != profile.tokenizer_target:
        raise ValueError(
            "profile_target_mismatch: "
            f"profile={profile.tokenizer_target}, "
            f"tokenized={tokenized.tokenizer_target}"
        )
    if (
        profile.expected_unit_policy is not None
        and profile.expected_unit_policy != training_data.unit_policy
    ):
        raise ValueError(
            "profile_unit_policy_mismatch: TrainingData UnitPolicy "
            "与 checkpoint Profile 不一致"
        )
    if training_data.tracks.user_video is not None:
        raise ValueError(
            "unsupported_input_modality: 当前 audio Semantic API client "
            "尚未实现 user_video Unit 输入"
        )

    arrangement = tokenize_result.arrangement
    audio_tensor = training_data.build_user_audio_tensor(
        tokenizer_id=tokenizer_id
    )
    unit_count = len(tokenized.unit_spans)
    units = _build_unit_inputs(
        audio_tensor=audio_tensor,
        unit_count=unit_count,
        unit_sec=training_data.unit_policy.unit_sec,
    )
    expected_tool_calls = _extract_expected_tool_calls(arrangement)
    tool_responses = _extract_tool_response_schedule(arrangement)
    _validate_tool_response_support(arrangement)
    expected_decode_units = _extract_expected_decode_units(
        tokenized.token_provenance,
        unit_count=unit_count,
    )
    gt_decode_spans = _extract_gt_decode_spans(tokenized.token_provenance)
    unit_policy_json = training_data.unit_policy.model_dump(mode="json")
    session_init = _build_session_init(
        training_data=training_data,
        profile=profile,
        data_root=data_root,
        unit_policy_json=unit_policy_json,
        expected_tool_calls=expected_tool_calls,
    )
    return FcApiTrainingDataScenario(
        data_id=training_data.data_id,
        sdk_version=tokenized.minicpm_o5_sdk_version,
        tokenizer_target=profile.tokenizer_target,
        tokenizer_fingerprint={
            "vocab_hash": tokenized.tokenizer_fingerprint.vocab_hash,
            "merges_hash": tokenized.tokenizer_fingerprint.merges_hash,
        },
        unit_policy=training_data.unit_policy,
        system=training_data.system,
        session_init=session_init,
        units=units,
        expected_decode_units=expected_decode_units,
        gt_decode_spans=gt_decode_spans,
        expected_tool_calls=expected_tool_calls,
        tool_responses=tool_responses,
        gt_input_ids=list(tokenized.input_ids),
    )


def _build_unit_inputs(
    *,
    audio_tensor: O5UserAudioTensor | None,
    unit_count: int,
    unit_sec: float,
) -> list[FcApiUnitInput]:
    """从 SDK canonical audio tensor 截取逐 Unit float32 PCM。"""

    import torch

    sample_rate = 16_000
    samples_per_unit = round(unit_sec * sample_rate)
    if audio_tensor is None:
        unit_samples = torch.zeros(
            (unit_count, samples_per_unit),
            dtype=torch.float32,
        )
    else:
        unit_samples = audio_tensor.units
        if audio_tensor.num_units != unit_count:
            raise ValueError(
                "build_user_audio_tensor Unit 数与 tokenize 不一致: "
                f"audio={audio_tensor.num_units}, tokenized={unit_count}"
            )
    return [
        FcApiUnitInput(
            unit_index=unit_index,
            input_id=f"training_data_unit_{unit_index:06d}",
            audio_base64=base64.b64encode(
                unit_samples[unit_index].contiguous().numpy().tobytes()
            ).decode("ascii"),
            sample_rate=sample_rate,
        )
        for unit_index in range(unit_count)
    ]


def _extract_expected_decode_units(
    provenance: list[O5TokenProvenance],
    *,
    unit_count: int,
) -> list[FcApiExpectedDecodeUnit]:
    """提取每个 Unit 的两条自由 decode 轨及 framework budget 终止 token。"""

    return [
        FcApiExpectedDecodeUnit(
            unit_index=unit_index,
            spoken_token_ids=[
                int(item.token_id)
                for item in provenance
                if item.unit_index == unit_index
                and item.track == "ai_spoken"
                and _is_api_reconstructable_decode_token(item)
            ],
            non_spoken_token_ids=[
                int(item.token_id)
                for item in provenance
                if item.unit_index == unit_index
                and item.track == "ai_non_spoken"
                and _is_api_reconstructable_decode_token(item)
            ],
        )
        for unit_index in range(unit_count)
    ]


def _extract_gt_decode_spans(
    provenance: list[O5TokenProvenance],
) -> list[FcApiDecodeSpan]:
    """定位每个 Unit/track 在完整 GT 中的连续 trainable 区间。"""

    spans: list[FcApiDecodeSpan] = []
    active_key: tuple[int, str] | None = None
    active_start = 0
    for index, item in enumerate([*provenance, None]):
        key: tuple[int, str] | None = None
        if (
            item is not None
            and item.track in {"ai_spoken", "ai_non_spoken"}
            and _is_api_reconstructable_decode_token(item)
        ):
            key = (item.unit_index, item.track)
        if key == active_key:
            continue
        if active_key is not None:
            spans.append(
                FcApiDecodeSpan(
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
            (span.unit_index, span.track) == key for span in spans
        )
        != 1
    ]
    if duplicate_keys:
        raise ValueError(
            "reconstruction_unsupported: 同一 Unit/track 的 trainable GT "
            f"不是单一连续区间: {duplicate_keys}"
        )
    return spans


def _is_api_reconstructable_decode_token(
    provenance: O5TokenProvenance,
) -> bool:
    """判断 token 是否由 Semantic API 历史显式恢复。

    AI 自由 decode token 均为 trainable；有限 budget 耗尽时由 framework 插入的
    ``<|non_spoken_budget_reached|>`` 虽不参与 loss，也由
    ``response.non_spoken.end`` 明确表达，必须进入 API token exact。
    """

    return provenance.trainable or (
        provenance.track == "ai_non_spoken"
        and provenance.token_text == "<|non_spoken_budget_reached|>"
    )


def _extract_expected_tool_calls(
    arrangement: O5DuplexArrangement,
) -> list[FcApiExpectedToolCall]:
    """按离散 token Unit/decode step 顺序提取固定 GT tool_call_id。"""

    candidates: list[tuple[int, int, O5ToolCallContent]] = []
    for segment in arrangement.tracks.ai_non_spoken.segments:
        content = segment.content
        if not isinstance(content, O5ToolCallContent):
            continue
        if not segment.token_unit_indices:
            raise ValueError(
                f"tool_call {content.tool_call_id!r} 缺少离散 token Unit"
            )
        first_plan_token = min(segment.token_unit_indices)
        candidates.append(
            (
                segment.token_unit_indices[first_plan_token],
                segment.token_decode_step_indices[first_plan_token],
                content,
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [
        FcApiExpectedToolCall(
            ordinal=ordinal,
            tool_call_id=content.tool_call_id,
            name=content.name,
            arguments=dict(content.arguments),
            activation_unit_index=unit_index,
        )
        for ordinal, (unit_index, _, content) in enumerate(candidates)
    ]


def _extract_tool_response_schedule(
    arrangement: O5DuplexArrangement,
) -> list[FcApiToolResponseSchedule]:
    """从 input_event perceived_input_gate 提取原始字符串结果调度。"""

    schedules: list[FcApiToolResponseSchedule] = []
    for segment in arrangement.tracks.input_event.segments:
        if segment.source != "user" or not isinstance(
            segment.event, O5ToolResponseEvent
        ):
            continue
        text_parts = [
            content.text
            for content in segment.event.contents
            if isinstance(content, O5TextContent)
        ]
        if len(text_parts) != len(segment.event.contents):
            raise ValueError(
                "unsupported_input_event: Semantic API 当前只支持 text tool result"
            )
        schedules.append(
            FcApiToolResponseSchedule(
                tool_call_id=segment.event.tool_call_id,
                send_before_unit_index=(
                    segment.perceived_input_gate.unit_index
                ),
                content="".join(text_parts),
            )
        )
    schedules.sort(key=lambda item: item.send_before_unit_index)
    return schedules


def _validate_tool_response_support(
    arrangement: O5DuplexArrangement,
) -> None:
    """拒绝当前 Semantic API client 无法发送的 user standalone 事件。"""

    unsupported = [
        segment
        for segment in arrangement.tracks.input_event.segments
        if segment.source == "user"
        and not isinstance(segment.event, O5ToolResponseEvent)
    ]
    if unsupported:
        raise ValueError(
            "unsupported_input_event: 当前 client 只支持 TrainingData tool response"
        )


def _build_session_init(
    *,
    training_data: O5DuplexTrainingData,
    profile: FcApiCheckpointProfile,
    data_root: Path,
    unit_policy_json: JsonObject,
    expected_tool_calls: list[FcApiExpectedToolCall],
) -> JsonObject:
    """构造保留完整 resolved UnitPolicy 的 session.init。"""

    system_segments: list[FcSystemTextInput | FcSystemAudioInput] = []
    tools: list[OpenAIToolDefinition] = []
    tts_prompt_audio_path: str | None = None
    if training_data.system is not None:
        tools = list(training_data.system.tools)
        for segment in training_data.system.segments:
            if isinstance(segment, O5SystemTextSegment):
                system_segments.append(FcSystemTextInput(text=segment.text))
                continue
            if not isinstance(segment, O5SystemAudioSegment):
                raise TypeError(
                    "request_build_failed: unsupported system segment "
                    f"{type(segment).__name__}"
                )
            file_path = segment.audio.file_path
            if file_path is None:
                raise ValueError(
                    "request_build_failed: system reference audio 缺少 file_path"
                )
            resolved_path = str((data_root / file_path).resolve())
            system_segments.append(
                FcSystemAudioInput(
                    audio=FcAudioPathInput(file_path=resolved_path)
                )
            )
            if tts_prompt_audio_path is None:
                tts_prompt_audio_path = resolved_path

    config: JsonObject = {
        "runtime": "fc_duplex",
        "decode_mode": "greedy",
        "sample_rate": 16_000,
        "non_spoken_scheduling": profile.non_spoken_scheduling,
    }

    payload: JsonObject = {
        "mode": "full_duplex",
        "fc_duplex": True,
        "protocol_version": "3",
        "checkpoint_profile_id": profile.profile_id,
        "model": profile.model,
        "tokenizer_target": profile.tokenizer_target,
        "system": FcSystemContentInput(
            segments=system_segments,
            tools=tools,
        ).model_dump(mode="json"),
        "generate_audio": False,
        "evaluation": {
            "fixed_tool_call_ids": [
                call.tool_call_id for call in expected_tool_calls
            ]
        } if expected_tool_calls else {},
        "unit_policy": unit_policy_json,
        "config": config,
    }
    if tts_prompt_audio_path is not None:
        payload["tts_prompt_audio"] = {
            "source": "path",
            "file_path": tts_prompt_audio_path,
        }
    return {"type": "session.init", "payload": payload}
