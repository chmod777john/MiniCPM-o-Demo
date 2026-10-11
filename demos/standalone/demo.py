"""Standalone Event Demo：用户在输入框里输入任意文本，回车即作为 ``input.standalone_event`` 注入。

这是注入外部内容最通用的方式：不关联任何 tool_call，模型在下一个被处理的 Unit 的
input_event 槽里看到 ``<standalone_event>正文</standalone_event>``。

preset 的 system 取自训练过 standalone 的 sub_agent 数据（每份数据的首条样本）：
``[参考音频, per-case 正文]`` 与 tools 原样保留，统一前缀由前端按数据类型拼在最前，
与训练侧 ``prepend_system_prefix_text`` 一致。工具顺序在每份数据内固定，属于训练口径，
所以 tools 跟着 preset 走。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from demo_gallery import demo
from demo_gallery.training_data import extract_defaults_from_case, pick_default_case

logger = logging.getLogger("demo_gallery.standalone")

SUB_AGENT_ROOT = Path("/user/sunweiyue/dataset/o5_sdk_training_data/sub_agent")

# preset id → 训练数据目录（均为已登记的数据资产目录）。
# 前缀编号与 F9-T3 data-mix 一致：zh no-think #2 / zh think #1 / en no-think #6。
PRESET_BUNDLES: dict[str, str] = {
    "subagent_zh_nothink": "agent_duplex_sub_agent_0929_min3k_max8k_zh_nothink_by_swy",
    "subagent_zh_think": "agent_duplex_sub_agent_0928_min3k_max8k_zh_think_by_swy",
    "subagent_en_nothink": "agent_duplex_sub_agent_0929_min3k_max7k_en_nothink_by_swy",
}


@demo(
    demo_id="standalone",
    title="MiniCPM-o · Standalone Event",
    summary="在输入框里输入任意文本、回车，作为 standalone input_event 注入模型",
    order=30,
    styles=("standalone.css",),
)
class StandaloneDemo:
    """defaults 只提供各 preset 的训练原样 system；通用运行时字段由 Gallery 合并。"""

    def defaults(self) -> dict[str, Any]:
        return {"standalone_presets": standalone_presets()}


def standalone_presets() -> dict[str, Any]:
    """读取各 preset 的训练原样 system / tools / TTS prompt。

    返回:
        ``{preset_id: {"source": 数据目录, "system": {...}, "tts_prompt_audio": {...}}}``；
        某份数据不可读时该 preset 记为 ``{"source": ..., "error": "..."}``，不静默丢弃。
    """

    presets: dict[str, Any] = {}
    for preset_id, bundle in PRESET_BUNDLES.items():
        root = SUB_AGENT_ROOT / bundle
        try:
            case_path = pick_default_case(root)
            if case_path is None:
                raise FileNotFoundError(f"{root} 下没有 traindata.jsonl")
            extracted = extract_defaults_from_case(case_path)
            presets[preset_id] = {
                "source": str(root),
                "system": extracted.default_system.model_dump(mode="json"),
                "tts_prompt_audio": (
                    extracted.default_tts_prompt_audio.model_dump(mode="json")
                    if extracted.default_tts_prompt_audio else None
                ),
            }
        except Exception as exc:  # noqa: BLE001 一个 preset 坏了不该让整个 Demo 不可用
            logger.warning("[standalone] preset %s 不可用: %s", preset_id, exc)
            presets[preset_id] = {"source": str(root), "error": f"{type(exc).__name__}: {exc}"}
    return presets
