"""Board Demo 的默认值：从训练 case 抽 system / tools / TTS prompt。"""

from __future__ import annotations

import logging
import os
from typing import Any

from minicpm_o5_sdk import OpenAIFunctionDefinition, OpenAIToolDefinition

from demo_gallery.training_data import (
    TrainingDataDefaults,
    extract_defaults_from_case,
    pick_default_case,
    resolve_case_folder,
)

logger = logging.getLogger("demo_gallery.board")

# 环境变量名保持 FC_BOARD_CASE_FOLDER 不变：core/fc_duplex/profiles.py 在按部署 Profile
# 注入它，改名会连带动到 Profile 投影链路。
CASE_FOLDER_CANDIDATES = (
    os.environ.get("FC_BOARD_CASE_FOLDER"),
    "/home/weihongliang/o45_fc_assets/training/delivery_train_data",
    "/user/weihongliang/o45_fc_assets/training/delivery_train_data",
)


def display_object_tool_default() -> OpenAIToolDefinition:
    """case 自身没声明 tools 时的兜底工具定义。"""

    return OpenAIToolDefinition(
        function=OpenAIFunctionDefinition(
            name="display_object_on_board",
            description=(
                "Display a named concrete object on the visual board so the user can see it. "
                "Use only for concrete, visualizable objects mentioned in user speech."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
    )


def board_defaults() -> dict[str, Any]:
    """Board 专属 defaults；Gallery 会把通用运行时字段合并进来。

    返回:
        ``case_folder`` / ``default_case_path`` / ``default_system`` /
        ``default_tts_prompt_audio``；case 目录不可用时给出可用的兜底（只有工具定义）。
    """

    case_folder = resolve_case_folder(*CASE_FOLDER_CANDIDATES)
    default_case_path = None
    defaults = TrainingDataDefaults(
        default_system={"segments": [], "tools": [display_object_tool_default()]}
    )
    if case_folder:
        default_case_path = pick_default_case(case_folder)
        if default_case_path is not None:
            try:
                defaults = extract_defaults_from_case(
                    default_case_path, fallback_tools=(display_object_tool_default(),)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[board] 无法从 %s 抽取默认值: %s: %s",
                    default_case_path,
                    type(exc).__name__,
                    exc,
                )
    return {
        "case_folder": case_folder,
        "default_case_path": default_case_path,
        "default_system": defaults.default_system.model_dump(mode="json"),
        "default_tts_prompt_audio": (
            defaults.default_tts_prompt_audio.model_dump(mode="json")
            if defaults.default_tts_prompt_audio is not None
            else None
        ),
    }
