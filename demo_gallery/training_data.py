"""从 O5 TrainingData 抽取推理侧默认值。

这是 Gallery 级能力：任何想「用训练数据当默认 system / tools / TTS prompt」的 demo 都能用，
不属于任何一个具体 demo（原先长在 gateway.py 里、名字叫 board，实际逻辑是通用的）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from core.fc_duplex.system_input import (
    FcAudioPathInput,
    FcSystemAudioInput,
    FcSystemContentInput,
    FcSystemTextInput,
)
from minicpm_o5_sdk import OpenAIToolDefinition


class TrainingDataDefaults(BaseModel):
    """从一条 TrainingData 投影出的、可直接喂给 session.init 的默认值。"""

    default_system: FcSystemContentInput = Field(default_factory=FcSystemContentInput)
    default_tts_prompt_audio: FcAudioPathInput | None = None


def resolve_case_folder(*candidates: str | None) -> str | None:
    """按顺序返回第一个真实存在的目录。

    参数:
        *candidates: 候选目录（允许 None / 空串，直接跳过）。

    返回:
        第一个存在的目录路径；都不存在时返回 None。
    """

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def pick_default_case(case_folder: str | Path) -> str | None:
    """选择默认 case，canonical ``traindata.jsonl`` 优先。

    参数:
        case_folder: TrainingData case 目录或 canonical Bundle 根目录。

    返回:
        默认 case 路径；目录里没有可用 case 时返回 None。
    """

    training_data_path = Path(case_folder) / "traindata.jsonl"
    if training_data_path.is_file():
        return str(training_data_path)
    cases = sorted(
        str(path)
        for path in Path(case_folder).iterdir()
        if (
            path.is_file()
            and path.suffix == ".json"
            and path.name != "selection_manifest.json"
        )
    )
    return cases[0] if cases else None


def load_training_data_structure(case_path: str | Path) -> dict[str, Any]:
    """读取单文件 JSON case 或 canonical Bundle 的首条 JSONL。

    参数:
        case_path: ``*.json`` case 或 ``traindata.jsonl`` 路径。

    返回:
        一条 TrainingData 的结构化字典。

    异常:
        ValueError: JSONL 为空，或内容不是 JSON object。
    """

    path = Path(case_path)
    if path.suffix != ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            structure = json.load(file)
    else:
        structure = None
        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if line:
                    structure = json.loads(line)
                    break
        if structure is None:
            raise ValueError(f"TrainingData JSONL 为空: {case_path}")
    if not isinstance(structure, dict):
        raise ValueError(f"TrainingData 必须是 JSON object: {case_path}")
    return structure


def extract_defaults_from_case(
    case_path: str | Path,
    *,
    fallback_tools: Sequence[OpenAIToolDefinition] = (),
) -> TrainingDataDefaults:
    """从训练 case 提取 Semantic Realtime API v3 的默认值。

    参数:
        case_path: 训练 case JSON / canonical JSONL 路径。
        fallback_tools: case 自身没声明 tools 时使用的兜底工具（demo 专属，由调用方给）。

    返回:
        含有序 ``default_system`` 与独立 ``default_tts_prompt_audio`` 的默认值。
        段序原样保留（text/audio 的先后顺序是训练口径的一部分，不能重排）。
    """

    structure = load_training_data_structure(case_path)

    data_root = os.path.dirname(str(case_path))
    system_segments: list[FcSystemTextInput | FcSystemAudioInput] = []
    first_system_audio: FcAudioPathInput | None = None
    for segment in (structure.get("system", {}) or {}).get("segments", []) or []:
        kind = segment.get("kind")
        if kind == "text":
            text = segment.get("text")
            if isinstance(text, str):
                system_segments.append(FcSystemTextInput(text=text))
        elif kind == "audio":
            file_path = (segment.get("audio") or {}).get("file_path")
            if isinstance(file_path, str) and file_path:
                audio_input = FcAudioPathInput(
                    file_path=os.path.realpath(os.path.join(data_root, file_path))
                )
                system_segments.append(FcSystemAudioInput(audio=audio_input))
                if first_system_audio is None:
                    first_system_audio = audio_input

    source_system = structure.get("system", {}) or {}
    tools = source_system.get("tools")
    if not isinstance(tools, list) or not tools:
        tools = structure.get("tools")
    if not isinstance(tools, list) or not tools:
        typed_tools = list(fallback_tools)
    else:
        typed_tools = [OpenAIToolDefinition.model_validate(tool) for tool in tools]

    source_tts_prompt_audio = structure.get("tts_prompt_audio")
    tts_file_path = (
        source_tts_prompt_audio.get("file_path")
        if isinstance(source_tts_prompt_audio, dict)
        else None
    )
    default_tts_prompt_audio: FcAudioPathInput | None = None
    if isinstance(tts_file_path, str) and tts_file_path:
        default_tts_prompt_audio = FcAudioPathInput(
            file_path=os.path.realpath(os.path.join(data_root, tts_file_path))
        )
    elif first_system_audio is not None:
        default_tts_prompt_audio = first_system_audio

    return TrainingDataDefaults(
        default_system=FcSystemContentInput(
            segments=system_segments,
            tools=typed_tools,
        ),
        default_tts_prompt_audio=default_tts_prompt_audio,
    )
