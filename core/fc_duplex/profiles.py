"""O45/O5 双工 FC 部署 Profile。

部署 Profile 是模型选择的唯一入口。调用方不传模型参数；backend 启动时加载一个
Profile，并据此选择 tokenizer target、checkpoint、ModelAdapter 与部署引擎。
"""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal, Union

from minicpm_o5_sdk import O5UnitPolicy
from o5_paths import has_complete_bundle
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class _BaseFcDeploymentProfile(BaseModel):
    """两个模型部署时共享的严格配置。"""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, description="稳定且可审计的部署 Profile ID。")
    sdk_version: Literal["0.0.5"] = Field(
        default="0.0.5",
        description="训练与推理共同使用的正式 MiniCPMO5 SDK 版本。",
    )
    # O45 keeps these legacy model artifacts. O5 uses weights_dir instead and
    # deliberately does not require a standalone code/checkpoint/backbone path.
    model_path: str | None = Field(
        default=None,
        description="O45 legacy model/code asset directory; not used by O5.",
    )
    pt_path: str | None = Field(
        default=None,
        description="O45 legacy PyTorch checkpoint; not used by O5.",
    )
    assets_dir: str | None = Field(
        default=None,
        description="Shared processor and Token2Wav assets directory.",
    )
    checkpoint_sha256: str | None = Field(
        default=None,
        description="完整 checkpoint 的可选 SHA256；正式部署应填写。",
    )
    reference_audio_path: str | None = Field(
        default=None,
        description="Session 默认使用的系统/TTS 参考音频。",
    )
    case_folder: str | None = Field(
        default=None,
        description="FC Board 静态 TrainingData case 目录。",
    )
    unit_policy: O5UnitPolicy = Field(description="SDK 0.0.5 定义的完整 UnitPolicy。")
    non_spoken_scheduling: Literal["quality", "latency"] = Field(
        default="quality",
        description="模型已验证的 non-spoken 调度策略。",
    )

    @field_validator("profile_id", "model_path", "pt_path", "assets_dir")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        """拒绝空白标识和路径。

        参数:
            value: 待校验的字符串。

        返回:
            原始非空字符串。
        """

        if value is None:
            return value
        if value != value.strip():
            raise ValueError("Profile 字符串字段不能包含首尾空白")
        return value

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        """校验可选 SHA256 使用 64 位小写十六进制。

        参数:
            value: checkpoint SHA256 或 None。

        返回:
            已校验的 SHA256 或 None。
        """

        if value is None:
            return None
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("checkpoint_sha256 必须是 64 位小写十六进制")
        return value


class O45FcDeploymentProfile(_BaseFcDeploymentProfile):
    """O45 单卡双工 FC 部署配置。"""

    model_family: Literal["o45"] = "o45"
    model_path: str = Field(min_length=1, description="O45 model/code asset directory.")
    pt_path: str = Field(min_length=1, description="O45 PyTorch checkpoint.")
    tokenizer_target: Literal["o45_fc"] = "o45_fc"
    required_model_rows: Literal[151772] = 151772
    modeling_package: Literal["modeling.o45"] = "modeling.o45"
    deployment_mode: Literal["single_eager", "single_opt"] = "single_eager"


class O5FcDeploymentProfile(_BaseFcDeploymentProfile):
    """O5 MoE TP2 + LLM Graph 双工 FC 部署配置。"""

    model_family: Literal["o5"] = "o5"
    tokenizer_target: Literal["o5"] = "o5"
    required_model_rows: Literal[248168] = 248168
    modeling_package: Literal["modeling.o5"] = "modeling.o5"
    deployment_mode: Literal["tp2"] = "tp2"
    weights_dir: str | None = Field(
        default=None,
        description=(
            "完整 O5 safetensors bundle。正式 O5 部署使用此字段；为空时只保留"
            " legacy profile 的可读性，backend 会拒绝启动。"
        ),
    )
    backbone_dir: str | None = Field(
        default=None,
        description="Deprecated legacy TP2 backbone; not consumed by the integrated loader.",
    )
    backbone_manifest_sha256: str | None = Field(
        default=None,
        description="backbone manifest 的可选 SHA256；正式部署应填写。",
    )
    tp_size: Literal[2] = 2
    llm_graph: Literal[True] = True
    llm_cache: int = Field(default=32768, ge=1)
    spmd_heartbeat_interval_sec: float = Field(default=30.0, gt=0)
    attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"] = "auto"

    @field_validator("weights_dir", "backbone_dir")
    @classmethod
    def validate_artifact_dir(cls, value: str | None) -> str | None:
        """拒绝带首尾空白的 backbone 路径。

        参数:
            value: backbone 目录字符串。

        返回:
            原始非空路径。
        """

        if value is None:
            return value
        if value != value.strip():
            raise ValueError("artifact path 不能包含首尾空白")
        return value

    @field_validator("backbone_manifest_sha256")
    @classmethod
    def validate_backbone_sha256(cls, value: str | None) -> str | None:
        """校验 backbone manifest 的可选 SHA256。

        参数:
            value: manifest SHA256 或 None。

        返回:
            已校验的 SHA256 或 None。
        """

        return _BaseFcDeploymentProfile.validate_sha256(value)


FcDeploymentProfile = Annotated[
    Union[O45FcDeploymentProfile, O5FcDeploymentProfile],
    Field(discriminator="model_family"),
]
"""Backend 启动时唯一允许的 FC 部署 Profile 联合类型。"""

_PROFILE_ADAPTER = TypeAdapter(FcDeploymentProfile)


def load_fc_deployment_profile(
    profile_path: str | Path,
    *,
    check_paths: bool = True,
    check_sdk_version: bool = True,
) -> FcDeploymentProfile:
    """读取并验证一个部署 Profile。

    参数:
        profile_path: Profile JSON 文件路径。
        check_paths: 是否同时检查模型资产路径存在。
        check_sdk_version: 是否要求当前环境安装正式 ``0.0.5``。

    返回:
        按 ``model_family`` 解析出的 O45 或 O5 Profile。

    异常:
        FileNotFoundError: Profile 或声明的模型资产不存在。
        pydantic.ValidationError: Profile schema、target 或 required rows 不合法。
    """

    path = Path(profile_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = _PROFILE_ADAPTER.validate_python(data)
    if check_paths:
        _validate_profile_paths(profile)
    if check_sdk_version:
        _validate_sdk_version(profile.sdk_version)
    return profile


def apply_fc_deployment_profile_environment(
    profile: FcDeploymentProfile,
) -> None:
    """把部署 Profile 投影为现有启动链消费的环境变量。

    参数:
        profile: 已完成 schema、路径和 SDK 校验的部署 Profile。

    返回:
        无返回值；只写当前 backend 子进程的环境。
    """

    os.environ["FC_MODEL_FAMILY"] = profile.model_family
    os.environ["CHECKPOINT_PROFILE_ID"] = profile.profile_id
    os.environ["FC_DUPLEX_UNIT_POLICY_JSON"] = profile.unit_policy.model_dump_json()
    os.environ["FC_DUPLEX_NON_SPOKEN_SCHEDULING"] = (
        profile.non_spoken_scheduling
    )
    policy_data = profile.unit_policy.model_dump(mode="json")
    os.environ["FC_DUPLEX_UNIT_SEC"] = str(policy_data["unit_sec"])
    _set_uniform_budget_environment(
        "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING",
        policy_data.get("non_spoken_budgets_while_listening"),
    )
    _set_uniform_budget_environment(
        "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING",
        policy_data.get("non_spoken_budgets_while_speaking"),
    )
    os.environ["O5_DEPLOY_MODE"] = profile.deployment_mode
    if profile.reference_audio_path is not None:
        os.environ["FC_REFERENCE_AUDIO_PATH"] = profile.reference_audio_path
    if profile.case_folder is not None:
        os.environ["FC_BOARD_CASE_FOLDER"] = profile.case_folder
    if isinstance(profile, O5FcDeploymentProfile):
        # The integrated O5 loader consumes one complete bundle. Clear the old
        # independent-backbone variable so a stale shell cannot change loading.
        os.environ.pop("O5_BACKBONE_DIR", None)
        os.environ.pop("O5_WEIGHTS_DIR", None)
        if profile.weights_dir is not None:
            os.environ["O5_WEIGHTS_DIR"] = profile.weights_dir
        os.environ.pop("O5_ASSETS_DIR", None)
        if profile.assets_dir is not None:
            os.environ["O5_ASSETS_DIR"] = profile.assets_dir
        os.environ["O5_LLM_CACHE"] = str(profile.llm_cache)
        os.environ["O5_LLM_GRAPH"] = "1"
        os.environ["O5_SPMD_HEARTBEAT_INTERVAL"] = str(
            profile.spmd_heartbeat_interval_sec
        )
        os.environ["O5_ATTN_IMPLEMENTATION"] = profile.attn_implementation


def _set_uniform_budget_environment(name: str, values: object) -> None:
    """只为恒定整数 budget 投影 legacy 前端标量。

    参数:
        name: 目标环境变量名。
        values: UnitPolicy 中的 budget 序列。

    返回:
        无返回值；可安全降级为标量时才写入。
    """

    os.environ.pop(name, None)
    if not isinstance(values, list) or not values:
        return
    first = values[0]
    if isinstance(first, bool) or not isinstance(first, int):
        return
    if any(value != first for value in values):
        return
    os.environ[name] = str(first)


def _validate_profile_paths(profile: FcDeploymentProfile) -> None:
    """检查 Profile 引用的模型资产存在。

    参数:
        profile: 已通过 schema 校验的部署 Profile。

    返回:
        无返回值；所有路径存在即通过。
    """

    if isinstance(profile, O45FcDeploymentProfile):
        model_path = Path(profile.model_path)
        pt_path = Path(profile.pt_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"model_path 不存在或不是目录: {model_path}")
        if not pt_path.is_file():
            raise FileNotFoundError(f"pt_path 不存在或不是文件: {pt_path}")
    elif profile.weights_dir is not None:
        weights_dir = Path(profile.weights_dir)
        if not has_complete_bundle(weights_dir):
            raise FileNotFoundError(
                "O5 weights_dir is not a complete safetensors bundle: "
                f"{weights_dir}"
            )
    else:
        # Keep old profiles loadable for inspection and checkpoint validation,
        # but never let them silently select a different default O5 bundle.
        for name, value, kind in (
            ("model_path", profile.model_path, "directory"),
            ("pt_path", profile.pt_path, "file"),
            ("backbone_dir", profile.backbone_dir, "directory"),
        ):
            if not value:
                raise FileNotFoundError(
                    "legacy O5 profile must provide model_path, pt_path and backbone_dir"
                )
            path = Path(value)
            if (kind == "directory" and not path.is_dir()) or (
                kind == "file" and not path.is_file()
            ):
                raise FileNotFoundError(
                    f"{name} 不存在或类型错误: {path}"
                )
    if profile.assets_dir is not None and not Path(profile.assets_dir).is_dir():
        raise FileNotFoundError(
            f"assets_dir 不存在或不是目录: {profile.assets_dir}"
        )
    if (
        profile.reference_audio_path is not None
        and not Path(profile.reference_audio_path).is_file()
    ):
        raise FileNotFoundError(
            f"reference_audio_path 不存在或不是文件: {profile.reference_audio_path}"
        )
    if profile.case_folder is not None and not Path(profile.case_folder).is_dir():
        raise FileNotFoundError(
            f"case_folder 不存在或不是目录: {profile.case_folder}"
        )


def _validate_sdk_version(expected_version: str) -> None:
    """确认运行环境安装了 Profile 要求的正式 SDK。

    参数:
        expected_version: Profile 固定的 SDK 版本。

    返回:
        无返回值；版本一致即通过。
    """

    try:
        installed_version = version("minicpm-o5-sdk")
    except PackageNotFoundError as exc:
        raise RuntimeError("当前环境未安装 minicpm-o5-sdk") from exc
    if installed_version != expected_version:
        raise RuntimeError(
            "minicpm-o5-sdk version mismatch: "
            f"expected={expected_version}, installed={installed_version}"
        )
