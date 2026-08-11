"""FC Board 从通用进程环境暴露 Checkpoint Profile 默认值的测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_extract_fc_board_defaults_preserves_v3_system_segment_order(
    tmp_path: Path,
) -> None:
    """Gateway 应保留文本/音频顺序与空文本，并独立解析 TTS prompt audio。"""

    import gateway

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "system.wav").write_bytes(b"system")
    (audio_dir / "tts.wav").write_bytes(b"tts")
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "system": {
                    "segments": [
                        {"kind": "text", "text": ""},
                        {
                            "kind": "audio",
                            "audio": {"file_path": "audio/system.wav"},
                        },
                        {"kind": "text", "text": "second"},
                    ]
                },
                "tools": [{"type": "function", "function": {"name": "test_tool"}}],
                "tts_prompt_audio": {
                    "source": "path",
                    "file_path": "audio/tts.wav",
                },
            }
        ),
        encoding="utf-8",
    )

    defaults = gateway._extract_fc_board_defaults_from_case(
        str(case_path)
    ).model_dump(mode="json")

    assert defaults == {
        "default_system": {
            "segments": [
                {"kind": "text", "text": ""},
                {
                    "kind": "audio",
                    "audio": {
                        "source": "path",
                        "file_path": str(tmp_path / "audio/system.wav"),
                    },
                },
                {"kind": "text", "text": "second"},
            ],
            "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "description": None,
                            "parameters": {},
                            "strict": None,
                        },
                    }
            ],
        },
        "default_tts_prompt_audio": {
            "source": "path",
            "file_path": str(tmp_path / "audio/tts.wav"),
        },
    }


def test_board_defaults_use_first_system_audio_as_explicit_tts_default(
    tmp_path: Path,
) -> None:
    """Case 未单列 TTS prompt 时，Board 模板应显式投影第一段 system audio。"""

    import gateway

    audio_path = tmp_path / "system.wav"
    audio_path.write_bytes(b"system")
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "system": {
                    "segments": [
                        {
                            "kind": "audio",
                            "audio": {"file_path": "system.wav"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    defaults = gateway._extract_fc_board_defaults_from_case(str(case_path))

    assert defaults.default_tts_prompt_audio is not None
    assert defaults.default_tts_prompt_audio.file_path == str(audio_path)


def test_board_defaults_load_canonical_training_data_jsonl(
    tmp_path: Path,
) -> None:
    """Canonical Bundle 应读取首条 JSONL，并保持 audio-first system 顺序。"""

    import gateway

    media_dir = tmp_path / "media" / "system_reference"
    media_dir.mkdir(parents=True)
    audio_path = media_dir / "HTRef06.wav"
    audio_path.write_bytes(b"audio")
    training_data_path = tmp_path / "traindata.jsonl"
    training_data_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "system": {
                            "segments": [
                                {
                                    "kind": "audio",
                                    "audio": {
                                        "file_path": "media/system_reference/HTRef06.wav"
                                    },
                                },
                                {"kind": "text", "text": "board-task"},
                            ],
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {"name": "display_object_on_board"},
                                }
                            ],
                        }
                    }
                ),
                json.dumps({"system": {"segments": [{"kind": "text", "text": "unused"}]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "selection_manifest.json").write_text("{}", encoding="utf-8")

    case_path = gateway._default_fc_board_case_path(str(tmp_path))
    defaults = gateway._extract_fc_board_defaults_from_case(case_path)

    assert case_path == str(training_data_path)
    assert [segment.kind for segment in defaults.default_system.segments] == [
        "audio",
        "text",
    ]
    assert defaults.default_system.segments[0].audio.file_path == str(audio_path)
    assert defaults.default_system.segments[1].text == "board-task"
    assert defaults.default_tts_prompt_audio is not None
    assert defaults.default_tts_prompt_audio.file_path == str(audio_path)


@pytest.mark.asyncio
async def test_fc_board_defaults_expose_profile_runtime_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway defaults 应把 launcher 注入值透传给通用应用壳。"""

    import gateway

    monkeypatch.setenv("CHECKPOINT_PROFILE_ID", "profile_test")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    monkeypatch.setenv("FC_DUPLEX_UNIT_SEC", "1.0")
    monkeypatch.setattr(gateway, "_fc_board_case_folder", lambda: None)

    defaults = await gateway.fc_board_defaults()

    assert defaults["checkpoint_profile_id"] == "profile_test"
    assert defaults["non_spoken_scheduling"] == "quality"
    assert defaults["non_spoken_budget_while_listening"] == 30
    assert defaults["non_spoken_budget_while_speaking"] == 15
    assert defaults["unit_sec"] == 1.0
    assert defaults["default_system"]["segments"] == []
    assert defaults["default_system"]["tools"]
    assert defaults["default_tts_prompt_audio"] is None
    assert "default_system_prompt" not in defaults
    assert "default_ref_audio_path" not in defaults
