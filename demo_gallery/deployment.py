"""Gallery 级部署信息：与具体 demo 无关，所有 demo 页面共用。

这些值全部来自部署 Profile 投影出的环境变量（见 ``core/fc_duplex/profiles.py``），
Gallery 只负责读出来给前端展示，不做任何推断。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("demo_gallery")


def optional_environment_number(
    name: str,
    *,
    value_type: type[int] | type[float],
) -> int | float | None:
    """读取部署 Profile 投影的可选数值。

    参数:
        name: 环境变量名。
        value_type: 目标数值类型。

    返回:
        环境变量缺失时返回 None，否则返回目标数值。

    异常:
        RuntimeError: 环境变量存在但无法转换成目标类型。
    """

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return value_type(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a valid {value_type.__name__}") from exc


def deployment_model_identity() -> dict[str, Any]:
    """从部署 Profile 读出「现在加载的是哪个模型」，给前端原样展示完整路径。

    返回:
        含 ``pt_path`` / ``backbone_dir`` / ``base_model_path`` / ``checkpoint_sha256`` /
        ``deployment_mode`` / ``sdk_version`` / ``tts_pad_protocol`` / ``profile_path``
        的字典；Profile 不可读时各项为 None。

        不做任何缩写或省略 —— 使用者要靠它确认跑的到底是哪个 checkpoint。
        注意 ``sdk_version`` 是**推理侧**装的包（启动硬门禁），反映训练血统的是
        ``tts_pad_protocol``（≤0.0.6a3 → legacy，≥0.0.6a4 → revised）。
    """

    identity: dict[str, Any] = {
        "pt_path": None,
        "backbone_dir": None,
        "base_model_path": None,
        "checkpoint_sha256": None,
        "deployment_mode": None,
        "sdk_version": None,
        "tts_pad_protocol": os.environ.get("FC_DUPLEX_TTS_PAD_PROTOCOL") or None,
        "profile_path": os.environ.get("FC_DEPLOYMENT_PROFILE"),
    }
    profile_path = identity["profile_path"]
    if not profile_path or not os.path.isfile(profile_path):
        return identity
    try:
        with open(profile_path, encoding="utf-8") as handle:
            profile = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[gallery] failed to read deployment profile %s: %s: %s",
            profile_path,
            type(exc).__name__,
            exc,
        )
        return identity
    identity.update(
        {
            "pt_path": profile.get("pt_path"),
            "backbone_dir": profile.get("backbone_dir"),
            "base_model_path": profile.get("model_path"),
            "checkpoint_sha256": profile.get("checkpoint_sha256"),
            "deployment_mode": profile.get("deployment_mode"),
            "sdk_version": profile.get("sdk_version"),
        }
    )
    return identity


def gallery_runtime() -> dict[str, Any]:
    """所有 demo 页面共用的运行时信息。

    返回:
        checkpoint 身份、模型路径、以及 Profile 投影的 UnitPolicy 字段。
        每个 demo 的 ``/defaults`` 会在这之上合并自己的专属字段。
    """

    return {
        "checkpoint_profile_id": os.environ.get("CHECKPOINT_PROFILE_ID"),
        "model_identity": deployment_model_identity(),
        # 前端 preset id；不属于当前 demo 的值由前端回退到该 demo 自己的默认值。
        "default_quick_setting": (
            os.environ.get("DEMO_DEFAULT_PRESET")
            or os.environ.get("FC_BOARD_DEFAULT_QUICK_SETTING")
        ),
        "tts_pad_protocol": os.environ.get("FC_DUPLEX_TTS_PAD_PROTOCOL") or None,
        "non_spoken_scheduling": os.environ.get("FC_DUPLEX_NON_SPOKEN_SCHEDULING"),
        "non_spoken_budget_while_listening": optional_environment_number(
            "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", value_type=int
        ),
        "non_spoken_budget_while_speaking": optional_environment_number(
            "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", value_type=int
        ),
        "unit_sec": optional_environment_number("FC_DUPLEX_UNIT_SEC", value_type=float),
    }
