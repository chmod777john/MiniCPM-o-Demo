"""客户端侧：把「按本机路径描述音频」的 session.init 转成协议 v4（音频为 base64 WAV）。

协议 v4 不接受服务端路径，音频由调用方读文件后随 session.init 上传。仓库里的推理 / 回放
工具内部仍用路径描述 TrainingData 的音频，在发送前统一过这一步。
"""

from __future__ import annotations

import base64
import copy
from pathlib import Path
from typing import Any, Dict

PROTOCOL_VERSION = "4"


def wav_file_audio(path: str | Path) -> Dict[str, str]:
    """读本机 WAV 文件，返回 v4 音频对象 ``{source:"base64", format:"wav", data}``。"""

    return {
        "source": "base64",
        "format": "wav",
        "data": base64.b64encode(Path(path).read_bytes()).decode("ascii"),
    }


def to_v4_session_init(frame: Dict[str, Any]) -> Dict[str, Any]:
    """返回新的 session.init 帧：版本改为 "4"，所有 ``{source:"path"}`` 音频读成 base64。不改原对象。"""

    converted = copy.deepcopy(frame)
    payload = converted.get("payload", converted)
    payload["protocol_version"] = PROTOCOL_VERSION
    _convert_audio(payload)
    return converted


def _convert_audio(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, dict) and item.get("source") == "path" and "file_path" in item:
                value[key] = wav_file_audio(item["file_path"])
            else:
                _convert_audio(item)
    elif isinstance(value, list):
        for item in value:
            _convert_audio(item)
