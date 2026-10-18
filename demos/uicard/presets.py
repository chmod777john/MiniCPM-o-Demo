"""uicard Demo 的 preset：每个 preset 对齐一份 ui_card 训练数据（也就对齐用它训的那些 checkpoint）。

system 的 ``[参考音频, per-case 正文]`` 与 tools 从该份数据里**一条真实样本**原样取出；统一前缀（#2 中文
no-think + 填充句）由前端拼在最前，与训练侧 ``system_prefix_text`` 一致（三个 data-mix 已逐字核对）。

选哪条样本：要求它的 ``show_card`` 支持 ``card_id``（训练数据里约 6–9 成是这种；另一种只收 card 对象的
是旧口径），1016 批还要求带 ``tavily_search``，并且正文里有会话时间块（建会话时前端把它刷新成当前时间）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from minicpm_o5_sdk import OpenAIToolDefinition

from demo_gallery.training_data import (
    DemoAudioPath,
    DemoAudioSegment,
    DemoSystemContent,
    DemoTextSegment,
)

logger = logging.getLogger("demo_gallery.uicard")

UI_CARD_ROOT = Path("/user/sunweiyue/dataset/o5_sdk_training_data/ui_card")

# preset id → 数据目录与样本必须带的工具。下拉框里的对齐说明在前端 web/quick-settings.js。
PRESET_BUNDLES: dict[str, dict[str, Any]] = {
    "uicard_1016_r002": {
        "bundle": "agent_duplex_uicard_1016_min2k_max8k_sysalign_train_recipe_002_v3_swy",
        "required_tools": ("show_card", "tavily_search"),
    },
    "uicard_1004_r011": {
        "bundle": "agent_duplex_uicard_1004_min2k_max8k_sysalign_train_recipe_011_v3",
        "required_tools": ("show_card",),
    },
}

SESSION_TIME_RE = re.compile(r"服务端本地时间：\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} CST")


def _show_card_takes_id(tools: list[dict[str, Any]]) -> bool:
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name") == "show_card":
            return "card_id" in ((function.get("parameters") or {}).get("properties") or {})
    return False


def _case_matches(case: dict[str, Any], required_tools: tuple[str, ...]) -> bool:
    system = case.get("system") or {}
    tools = system.get("tools") or []
    names = {(tool.get("function") or {}).get("name") for tool in tools}
    if not all(name in names for name in required_tools) or not _show_card_takes_id(tools):
        return False
    texts = [segment.get("text") or "" for segment in system.get("segments") or [] if segment.get("kind") == "text"]
    return any(SESSION_TIME_RE.search(text) for text in texts)


def pick_case(bundle_root: Path, required_tools: tuple[str, ...]) -> dict[str, Any]:
    """按文件顺序取第一条满足条件的样本。

    异常:
        FileNotFoundError: 数据目录没有 traindata.jsonl。
        ValueError: 没有满足条件的样本。
    """

    traindata = bundle_root / "traindata.jsonl"
    if not traindata.is_file():
        raise FileNotFoundError(f"{bundle_root} 下没有 traindata.jsonl")
    with traindata.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            if _case_matches(case, required_tools):
                return case
    raise ValueError(f"{bundle_root} 里没有 show_card 收 card_id、带 {required_tools} 且有会话时间的样本")


def case_system(case: dict[str, Any], bundle_root: Path) -> tuple[DemoSystemContent, DemoAudioPath | None]:
    """样本的 system 段（音频转成 Demo 服务机器上的绝对路径）与工具，段序原样保留。"""

    segments: list[DemoTextSegment | DemoAudioSegment] = []
    first_audio: DemoAudioPath | None = None
    for segment in (case.get("system") or {}).get("segments") or []:
        if segment.get("kind") == "text":
            segments.append(DemoTextSegment(text=segment.get("text") or ""))
        elif segment.get("kind") == "audio":
            file_path = (segment.get("audio") or {}).get("file_path") or ""
            audio = DemoAudioPath(file_path=os.path.realpath(bundle_root / file_path))
            segments.append(DemoAudioSegment(audio=audio))
            first_audio = first_audio or audio
    tools = [OpenAIToolDefinition.model_validate(tool) for tool in (case.get("system") or {}).get("tools") or []]
    return DemoSystemContent(segments=segments, tools=tools), first_audio


def uicard_presets() -> dict[str, Any]:
    """各 preset 的训练原样 system / tools / TTS prompt。

    返回:
        ``{preset_id: {"source", "data_id", "system", "tts_prompt_audio"}}``；
        某份数据不可读时该 preset 记为 ``{"source", "error"}``，不静默丢弃。
    """

    presets: dict[str, Any] = {}
    for preset_id, spec in PRESET_BUNDLES.items():
        root = UI_CARD_ROOT / spec["bundle"]
        try:
            case = pick_case(root, spec["required_tools"])
            system, audio = case_system(case, root)
            presets[preset_id] = {
                "source": str(root),
                "data_id": case.get("data_id"),
                "system": system.model_dump(mode="json"),
                "tts_prompt_audio": audio.model_dump(mode="json") if audio else None,
            }
        except Exception as exc:  # noqa: BLE001 一个 preset 坏了不该让整个 Demo 不可用
            logger.warning("[uicard] preset %s 不可用: %s", preset_id, exc)
            presets[preset_id] = {
                "source": str(root),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return presets
