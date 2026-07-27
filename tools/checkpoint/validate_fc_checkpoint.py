"""校验 SDK 0.0.5 FC checkpoint 与部署 Profile 的硬兼容性。

该工具只读取完整 PT 和 O5 TP2 backbone 元数据，不加载模型到 GPU。它在服务启动前
确认 embedding/lm_head rows、SDK target 和 backbone config 一致，避免把错误权重组合
推进到昂贵的 TP2 启动阶段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from core.fc_duplex.profiles import (
    FcDeploymentProfile,
    O5FcDeploymentProfile,
    load_fc_deployment_profile,
)


class FcCheckpointValidationReport(BaseModel):
    """一次 checkpoint/Profile 兼容性检查结果。"""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    model_family: str
    tokenizer_target: str
    required_model_rows: int
    embedding_rows: int
    lm_head_rows: int
    backbone_vocab_size: int | None = None
    checkpoint_sha256_verified: bool | None = None
    valid: bool
    errors: list[str] = Field(default_factory=list)


def load_checkpoint_state_dict(pt_path: str | Path) -> dict[str, Any]:
    """读取完整 PT 中的 state_dict。

    参数:
        pt_path: ``torch.save`` 生成的完整 checkpoint 文件。

    返回:
        去除常见 ``state_dict/model/module`` wrapper 后的 tensor 字典。
    """

    path = str(pt_path)
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        nested = state.get(key) if isinstance(state, dict) else None
        if isinstance(nested, dict):
            return nested
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 顶层必须是 dict: {type(state)!r}")
    return state


def validate_fc_checkpoint(
    profile: FcDeploymentProfile,
) -> FcCheckpointValidationReport:
    """校验 Profile 引用的完整 PT 与可选 O5 backbone。

    参数:
        profile: 已通过 schema、路径和 SDK 版本校验的部署 Profile。

    返回:
        包含所有硬错误的结构化报告。
    """

    state = load_checkpoint_state_dict(profile.pt_path)
    embed = _require_tensor(
        state,
        (
            "llm.model.embed_tokens.weight",
            "model.llm.model.embed_tokens.weight",
        ),
        "embedding",
    )
    lm_head = _require_tensor(
        state,
        (
            "llm.lm_head.weight",
            "llm.model.lm_head.weight",
            "model.llm.lm_head.weight",
            "lm_head.weight",
        ),
        "lm_head",
    )
    embedding_rows = int(embed.shape[0])
    lm_head_rows = int(lm_head.shape[0])
    errors: list[str] = []
    if embedding_rows != profile.required_model_rows:
        errors.append(
            "embedding rows mismatch: "
            f"expected={profile.required_model_rows}, actual={embedding_rows}"
        )
    if lm_head_rows != profile.required_model_rows:
        errors.append(
            "lm_head rows mismatch: "
            f"expected={profile.required_model_rows}, actual={lm_head_rows}"
        )

    backbone_vocab_size: int | None = None
    if isinstance(profile, O5FcDeploymentProfile):
        config_path = Path(profile.backbone_dir) / "config.json"
        backbone_config = json.loads(config_path.read_text(encoding="utf-8"))
        backbone_vocab_size = int(backbone_config["vocab_size"])
        if backbone_vocab_size != profile.required_model_rows:
            errors.append(
                "backbone vocab_size mismatch: "
                f"expected={profile.required_model_rows}, "
                f"actual={backbone_vocab_size}"
            )

    sha_verified: bool | None = None
    if profile.checkpoint_sha256 is not None:
        actual_sha256 = _sha256_file(Path(profile.pt_path))
        sha_verified = actual_sha256 == profile.checkpoint_sha256
        if not sha_verified:
            errors.append(
                "checkpoint SHA256 mismatch: "
                f"expected={profile.checkpoint_sha256}, actual={actual_sha256}"
            )

    return FcCheckpointValidationReport(
        profile_id=profile.profile_id,
        model_family=profile.model_family,
        tokenizer_target=profile.tokenizer_target,
        required_model_rows=profile.required_model_rows,
        embedding_rows=embedding_rows,
        lm_head_rows=lm_head_rows,
        backbone_vocab_size=backbone_vocab_size,
        checkpoint_sha256_verified=sha_verified,
        valid=not errors,
        errors=errors,
    )


def _require_tensor(
    state: dict[str, Any],
    candidates: tuple[str, ...],
    label: str,
) -> Any:
    """从候选 key 中读取一个 tensor。

    参数:
        state: checkpoint state_dict。
        candidates: 允许的完整 key。
        label: 错误消息使用的字段名。

    返回:
        首个命中的 tensor。
    """

    for key in candidates:
        value = state.get(key)
        if value is not None and hasattr(value, "shape"):
            return value
    raise KeyError(f"checkpoint 缺少 {label} tensor，候选 key={candidates}")


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA256。

    参数:
        path: 待检查文件。

    返回:
        64 位小写十六进制 SHA256。
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """命令行入口。

    返回:
        校验通过返回 0，否则返回 1。
    """

    parser = argparse.ArgumentParser(
        description="Validate an SDK 0.0.5 FC checkpoint deployment profile"
    )
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    profile = load_fc_deployment_profile(args.profile)
    report = validate_fc_checkpoint(profile)
    print(report.model_dump_json(indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
