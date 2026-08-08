"""FC Semantic Realtime API v3 system/prepare 破坏性契约测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from minicpm_o5_sdk.protocols.duplex.training_data import (
    O5SystemAudioSegment,
    O5SystemTextSegment,
)

from core.fc_duplex.system_input import (
    FcAudioPathInput,
    FcSystemContentInput,
    materialize_o5_system_content,
)
from py_backend.fc_duplex_runtime import FcDuplexSessionRuntime


def _write_float32_wav(path: Path, *, channels: int = 1) -> None:
    """写入测试使用的 16 kHz float32 WAV。

    参数:
        path: 输出 WAV 路径。
        channels: 输出声道数。

    返回:
        无返回值。
    """

    samples = np.linspace(-0.25, 0.25, 1_600, dtype=np.float32)
    if channels > 1:
        samples = np.repeat(samples[:, None], channels, axis=1)
    sf.write(path, samples, 16_000, subtype="FLOAT")


def test_materialize_preserves_order_and_validates_audio(tmp_path: Path) -> None:
    """v3 system 必须无损保留 text/audio 顺序并生成 SDK LazyAudio。"""

    audio_path = tmp_path / "system.wav"
    _write_float32_wav(audio_path)
    wire = FcSystemContentInput.model_validate(
        {
            "segments": [
                {"kind": "text", "text": "前缀"},
                {
                    "kind": "audio",
                    "audio": {
                        "source": "path",
                        "file_path": str(audio_path),
                    },
                },
                {"kind": "text", "text": "后缀"},
            ],
            "tools": [],
        }
    )

    content = materialize_o5_system_content(wire)

    assert isinstance(content.segments[0], O5SystemTextSegment)
    assert isinstance(content.segments[1], O5SystemAudioSegment)
    assert isinstance(content.segments[2], O5SystemTextSegment)
    assert content.segments[1].audio.duration_sec == pytest.approx(0.1)
    assert content.segments[1].audio.get_tensor().shape == (1_600,)


def test_audio_path_requires_absolute_16k_mono_file(tmp_path: Path) -> None:
    """路径、采样率和声道 invariant 不允许隐式修复。"""

    with pytest.raises(ValueError, match="绝对路径"):
        FcAudioPathInput(file_path="relative.wav")

    stereo_path = tmp_path / "stereo.wav"
    _write_float32_wav(stereo_path, channels=2)
    wire = FcSystemContentInput.model_validate(
        {
            "segments": [
                {
                    "kind": "audio",
                    "audio": {
                        "source": "path",
                        "file_path": str(stereo_path),
                    },
                }
            ],
            "tools": [],
        }
    )
    with pytest.raises(ValueError, match="单声道"):
        materialize_o5_system_content(wire)


@pytest.mark.asyncio
async def test_runtime_rejects_non_v3_and_legacy_prepare_fields() -> None:
    """Runtime 必须在触发模型前拒绝旧版本和旧 prepare 字段。"""

    async def send(_: str, **__: object) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="semantic_v3_rejection",
        backend=object(),
        send=send,
    )
    with pytest.raises(RuntimeError, match="protocol_version='3'"):
        await runtime.prepare({"protocol_version": "fc-duplex-semantic-v2"})

    with pytest.raises(RuntimeError, match="legacy prepare fields.*system_prompt"):
        await runtime.prepare(
            {
                "protocol_version": "3",
                "system_prompt": "旧字段",
                "system": {"segments": [], "tools": []},
                "generate_audio": False,
            }
        )
