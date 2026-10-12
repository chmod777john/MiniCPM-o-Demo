"""当前 Worker 进程加载的模型摘要（API 侧的单一事实源）。

来源是部署 Profile 投影出的环境变量与 Profile 文件本身（见 ``profiles.py``），所以只在
加载模型的 Worker 进程里有意义。用途：

- ``session.created.model``：告诉客户端这次会话实际用的是哪个模型；
- Worker 就绪文件：控制面板据此登记 Worker，``/e/<端点>/v1/info`` 再对外公布。

``session_defaults`` 只是 **Session 没传时的推荐默认值**（tts_pad、Unit 策略、调度），
Session 传了就以 Session 为准；本次会话实际生效的值见 ``session.created.session_config``。
"""

from __future__ import annotations

import json
import logging
import os
from importlib import metadata
from typing import Any, Dict

logger = logging.getLogger(__name__)


def worker_model_info() -> Dict[str, Any]:
    """返回可 JSON 序列化的模型摘要；缺的字段为 None，不抛异常。"""

    info: Dict[str, Any] = {
        "profile_id": os.environ.get("CHECKPOINT_PROFILE_ID"),
        "model_family": os.environ.get("FC_MODEL_FAMILY"),
        "session_defaults": {
            "tts_pad_protocol": os.environ.get("FC_DUPLEX_TTS_PAD_PROTOCOL") or None,
            "non_spoken_scheduling": os.environ.get("FC_DUPLEX_NON_SPOKEN_SCHEDULING") or None,
            "unit_policy": _json_env("FC_DUPLEX_UNIT_POLICY_JSON"),
        },
        "deployment_mode": os.environ.get("O5_DEPLOY_MODE"),
        "sdk_version": _installed_sdk_version(),
        "pt_path": None,
        "backbone_dir": None,
        "base_model_path": None,
        "checkpoint_sha256": None,
    }
    profile_path = os.environ.get("FC_DEPLOYMENT_PROFILE")
    if profile_path and os.path.isfile(profile_path):
        try:
            with open(profile_path, encoding="utf-8") as handle:
                profile = json.load(handle)
            info.update({
                "pt_path": profile.get("pt_path"),
                "backbone_dir": profile.get("backbone_dir"),
                "base_model_path": profile.get("model_path"),
                "checkpoint_sha256": profile.get("checkpoint_sha256"),
            })
        except (OSError, ValueError) as exc:
            logger.warning("failed to read deployment profile %s: %s", profile_path, exc)
    return info


def _json_env(name: str) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _installed_sdk_version() -> str | None:
    try:
        return metadata.version("minicpm-o5-sdk")
    except metadata.PackageNotFoundError:
        return None
