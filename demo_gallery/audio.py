"""把 Demo 服务所在机器上的音频文件读成协议 v4 的 base64 WAV（供 shell 建会话用）。

Demo 服务也对外公开，所以**只允许读白名单目录里的文件**：``DEMO_AUDIO_ROOTS``（用 ``:``
分隔的绝对目录）。未配置时一律拒绝（失败即关闭）。只收 16 kHz 单声道 WAV —— API 不重采样，
这里也不替调用方转换，格式不对直接报错，让问题暴露在配置上。
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

import soundfile as sf


class AudioResolveError(ValueError):
    pass


def audio_roots() -> list[Path]:
    raw = os.environ.get("DEMO_AUDIO_ROOTS") or ""
    return [Path(item).resolve() for item in raw.split(os.pathsep) if item.strip()]


def resolve_wav_base64(path: str) -> dict[str, object]:
    """返回 ``{source:"base64", format:"wav", data, file_path, duration_sec}``。"""

    roots = audio_roots()
    if not roots:
        raise AudioResolveError("Demo 服务未配置 DEMO_AUDIO_ROOTS，不读取任何音频文件")
    candidate = Path(path)
    if not candidate.is_absolute():
        raise AudioResolveError("音频路径必须是绝对路径")
    resolved = candidate.resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise AudioResolveError(f"音频路径不在允许的目录内（DEMO_AUDIO_ROOTS）: {path}")
    if not resolved.is_file():
        raise AudioResolveError(f"音频文件不存在: {path}")
    stat = resolved.stat()
    return dict(_encode(str(resolved), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=64)
def _encode(path: str, _mtime_ns: int, _size: int) -> tuple:
    info = sf.info(path)
    if info.format != "WAV":
        raise AudioResolveError(f"只支持 WAV，实际 {info.format}: {path}")
    if info.samplerate != 16_000 or info.channels != 1:
        raise AudioResolveError(
            f"必须是 16 kHz 单声道（API 不重采样），实际 {info.samplerate} Hz / {info.channels} 声道: {path}"
        )
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (("source", "base64"), ("format", "wav"), ("data", data), ("file_path", path),
            ("duration_sec", round(info.frames / info.samplerate, 3)))
