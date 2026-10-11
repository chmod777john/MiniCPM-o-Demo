"""subagent Demo 的 preset：system 与 tools 取自 sub_agent 训练数据（每份数据的首条样本）。

``[参考音频, per-case 正文]`` 与 tools 原样保留；统一前缀由前端按数据类型拼在最前，与训练侧
``prepend_system_prefix_text`` 一致。工具顺序在每份数据内固定，属于训练口径，所以随 preset 切换。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from demo_gallery.training_data import extract_defaults_from_case, pick_default_case

logger = logging.getLogger("demo_gallery.subagent")

SUB_AGENT_ROOT = Path("/user/sunweiyue/dataset/o5_sdk_training_data/sub_agent")

# preset id → (训练数据目录, 语言)。前缀编号与 F9-T3 data-mix 一致：zh no-think #2 / zh think #1 / en no-think #6。
PRESET_BUNDLES: dict[str, tuple[str, str]] = {
    "subagent_zh_nothink": ("agent_duplex_sub_agent_0929_min3k_max8k_zh_nothink_by_swy", "zh"),
    "subagent_zh_think": ("agent_duplex_sub_agent_0928_min3k_max8k_zh_think_by_swy", "zh"),
    "subagent_en_nothink": ("agent_duplex_sub_agent_0929_min3k_max7k_en_nothink_by_swy", "en"),
}


def subagent_presets() -> dict[str, Any]:
    """读取各 preset 的训练原样 system / tools / TTS prompt。

    返回:
        ``{preset_id: {"source", "language", "system", "tts_prompt_audio"}}``；
        某份数据不可读时该 preset 记为 ``{"source", "error"}``，不静默丢弃。
    """

    presets: dict[str, Any] = {}
    for preset_id, (bundle, language) in PRESET_BUNDLES.items():
        root = SUB_AGENT_ROOT / bundle
        try:
            case_path = pick_default_case(root)
            if case_path is None:
                raise FileNotFoundError(f"{root} 下没有 traindata.jsonl")
            extracted = extract_defaults_from_case(case_path)
            presets[preset_id] = {
                "source": str(root),
                "language": language,
                "system": extracted.default_system.model_dump(mode="json"),
                "tts_prompt_audio": (
                    extracted.default_tts_prompt_audio.model_dump(mode="json")
                    if extracted.default_tts_prompt_audio else None
                ),
            }
        except Exception as exc:  # noqa: BLE001 一个 preset 坏了不该让整个 Demo 不可用
            logger.warning("[subagent] preset %s 不可用: %s", preset_id, exc)
            presets[preset_id] = {"source": str(root), "error": f"{type(exc).__name__}: {exc}"}
    return presets
