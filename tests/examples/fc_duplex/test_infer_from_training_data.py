"""TrainingData 驱动的纯 API 推理样例测试。"""

from __future__ import annotations

import base64
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from minicpm_o5_sdk import O5UnitPolicy

from examples.fc_duplex.infer_from_training_data import (
    ApiDecodeSpan,
    ApiHistoryEntry,
    build_session_init_payload,
    load_training_data_structure,
    materialize_response_artifacts,
    splice_api_decode_tracks,
)


def test_load_training_data_structure_supports_json_and_jsonl(
    tmp_path: Path,
) -> None:
    """样例应接受单 JSON 和 JSONL 指定行。"""

    first = {"data_id": "first"}
    second = {"data_id": "second"}
    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps(first), encoding="utf-8")
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text(
        "\n".join((json.dumps(first), json.dumps(second))) + "\n",
        encoding="utf-8",
    )

    assert load_training_data_structure(json_path, line_index=0) == first
    assert load_training_data_structure(jsonl_path, line_index=1) == second
    with pytest.raises(IndexError, match="line_index=2"):
        load_training_data_structure(jsonl_path, line_index=2)


def test_session_init_contains_request_semantics_without_evaluation_fields() -> None:
    """API 样例必须只携带请求语义，不能注入 GT 对拍字段。"""

    payload = build_session_init_payload(
        system_prompt="你是助手",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        unit_policy=O5UnitPolicy(
            unit_sec=1.0,
            non_spoken_budgets_while_listening=[30],
            non_spoken_budgets_while_speaking=[15],
        ),
        reference_audio_path=None,
    )

    assert payload["type"] == "session.init"
    assert payload["payload"]["fc_duplex"] is True
    assert payload["payload"]["generate_audio"] is True
    assert payload["payload"]["system_prompt"] == "你是助手"
    assert "evaluation" not in payload["payload"]
    assert "checkpoint_profile_id" not in payload["payload"]
    assert "model" not in payload["payload"]


def test_splice_api_decode_tracks_replaces_scaffold_without_comparison() -> None:
    """重建只替换 API decode 轨，不计算 expected/actual 指标。"""

    reconstructed = splice_api_decode_tracks(
        scaffold_token_ids=[10, 11, 12, 13, 14, 15],
        decode_spans=[
            ApiDecodeSpan(
                unit_index=0,
                track="spoken",
                start_token_index=1,
                end_token_index_exclusive=3,
            ),
            ApiDecodeSpan(
                unit_index=0,
                track="non_spoken",
                start_token_index=4,
                end_token_index_exclusive=5,
            ),
        ],
        actual_decode_tokens={
            (0, "spoken"): [101, 102, 103],
            (0, "non_spoken"): [201],
        },
    )

    assert reconstructed == [10, 101, 102, 103, 13, 201, 15]


def test_splice_api_decode_tracks_rejects_missing_history_track() -> None:
    """API history 缺少轨道时必须显式失败，不能保留 GT scaffold。"""

    with pytest.raises(ValueError, match="缺少 decode 轨"):
        splice_api_decode_tracks(
            scaffold_token_ids=[10, 11, 12],
            decode_spans=[
                ApiDecodeSpan(
                    unit_index=0,
                    track="spoken",
                    start_token_index=1,
                    end_token_index_exclusive=2,
                )
            ],
            actual_decode_tokens={},
        )


def test_materialize_response_artifacts_saves_history_and_ai_audio(
    tmp_path: Path,
) -> None:
    """API 返回的语义事件和真实 AI 音频必须完整落盘。"""

    samples = np.linspace(-0.25, 0.25, 2_400, dtype=np.float32)
    audio_base64 = base64.b64encode(samples.tobytes()).decode("ascii")
    history = [
        ApiHistoryEntry(
            sequence=0,
            direction="down",
            event={
                "type": "response.think.end",
                "unit_index": 0,
                "full_text": "先思考",
            },
        ),
        ApiHistoryEntry(
            sequence=1,
            direction="down",
            event={
                "type": "response.tool_call.done",
                "tool_call_id": "tc_1",
                "unit_index": 0,
                "call": {"name": "lookup", "arguments": {"q": "猫"}},
            },
        ),
        ApiHistoryEntry(
            sequence=2,
            direction="down",
            event={
                "type": "response.spoken.delta",
                "unit_index": 0,
                "steps": [{"kind": "text", "text": "你好"}],
                "audio": audio_base64,
                "sample_rate": 24_000,
            },
        ),
        ApiHistoryEntry(
            sequence=3,
            direction="down",
            event={
                "type": "response.spoken.end",
                "unit_index": 0,
                "reason": "turn_eos",
                "full_text": "你好",
            },
        ),
        ApiHistoryEntry(
            sequence=4,
            direction="down",
            event={
                "type": "response.spoken.end",
                "unit_index": 1,
                "reason": "listen",
            },
        ),
    ]

    artifacts = materialize_response_artifacts(
        history=history,
        output_dir=tmp_path,
    )

    assert artifacts.think_texts == ["先思考"]
    assert artifacts.tool_calls[0]["call"]["name"] == "lookup"
    assert len(artifacts.spoken_turns) == 1
    assert artifacts.spoken_turns[0].text == "你好"
    assert artifacts.spoken_turns[0].sample_rate == 24_000
    audio_path = tmp_path / artifacts.spoken_turns[0].audio_path
    replay_audio_path = (
        tmp_path / artifacts.spoken_turns[0].replay_audio_path
    )
    assert audio_path.is_file()
    assert replay_audio_path.is_file()
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
    with wave.open(str(replay_audio_path), "rb") as wav:
        assert wav.getframerate() == 16_000
