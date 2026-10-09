"""Gallery core 从 O5 TrainingData 抽取推理侧默认值的测试。

原先长在 gateway.py 里、名字叫 board，实际逻辑是通用的（任何想用训练数据当默认
system/tools 的 demo 都要）。现在归 demo_gallery.training_data。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_extract_fc_board_defaults_preserves_v3_system_segment_order(
    tmp_path: Path,
) -> None:
    """Gateway 应保留文本/音频顺序与空文本，并独立解析 TTS prompt audio。"""

    from demo_gallery import training_data

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

    defaults = training_data.extract_defaults_from_case(str(case_path)).model_dump(mode="json")

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

    from demo_gallery import training_data

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

    defaults = training_data.extract_defaults_from_case(str(case_path))

    assert defaults.default_tts_prompt_audio is not None
    assert defaults.default_tts_prompt_audio.file_path == str(audio_path)


def test_board_defaults_load_canonical_training_data_jsonl(
    tmp_path: Path,
) -> None:
    """Canonical Bundle 应读取首条 JSONL，并保持 audio-first system 顺序。"""

    from demo_gallery import training_data

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

    case_path = training_data.pick_default_case(str(tmp_path))
    defaults = training_data.extract_defaults_from_case(case_path)

    assert case_path == str(training_data_path)
    assert [segment.kind for segment in defaults.default_system.segments] == [
        "audio",
        "text",
    ]
    assert defaults.default_system.segments[0].audio.file_path == str(audio_path)
    assert defaults.default_system.segments[1].text == "board-task"
    assert defaults.default_tts_prompt_audio is not None
    assert defaults.default_tts_prompt_audio.file_path == str(audio_path)
