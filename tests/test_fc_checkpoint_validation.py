"""SDK 0.0.5 FC checkpoint/Profile 硬兼容性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from core.fc_duplex.profiles import O45FcDeploymentProfile, O5FcDeploymentProfile
from minicpm_o5_sdk import O5UnitPolicy
from tools.checkpoint.validate_fc_checkpoint import validate_fc_checkpoint


def _unit_policy() -> O5UnitPolicy:
    """构造最小测试 UnitPolicy。"""

    return O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[30],
        non_spoken_budgets_while_speaking=[15],
    )


def _write_checkpoint(path: Path, rows: int) -> None:
    """写入只含词表边界 tensor 的轻量 checkpoint。"""

    torch.save(
        {
            "llm.model.embed_tokens.weight": torch.zeros(rows, 2),
            "llm.lm_head.weight": torch.zeros(rows, 2),
        },
        path,
    )


def test_o45_checkpoint_rows_match_sdk005(tmp_path: Path) -> None:
    """O45 checkpoint 的 embedding/lm_head 必须都是 151772 行。"""

    pt_path = tmp_path / "o45.pt"
    _write_checkpoint(pt_path, 151772)
    model_path = tmp_path / "o45-model"
    model_path.mkdir()
    profile = O45FcDeploymentProfile(
        profile_id="o45-valid",
        model_path=str(model_path),
        pt_path=str(pt_path),
        unit_policy=_unit_policy(),
    )

    report = validate_fc_checkpoint(profile)

    assert report.valid is True
    assert report.embedding_rows == 151772


def test_o5_checkpoint_and_backbone_match_sdk005(tmp_path: Path) -> None:
    """O5 完整 PT 与 TP2 backbone 必须共同使用 248168 rows。"""

    pt_path = tmp_path / "o5.pt"
    _write_checkpoint(pt_path, 248168)
    model_path = tmp_path / "o5-model"
    model_path.mkdir()
    backbone = tmp_path / "o5-backbone"
    backbone.mkdir()
    (backbone / "config.json").write_text(
        json.dumps({"vocab_size": 248168}),
        encoding="utf-8",
    )
    (backbone / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}}),
        encoding="utf-8",
    )
    profile = O5FcDeploymentProfile(
        profile_id="o5-valid",
        model_path=str(model_path),
        pt_path=str(pt_path),
        backbone_dir=str(backbone),
        unit_policy=_unit_policy(),
    )

    report = validate_fc_checkpoint(profile)

    assert report.valid is True
    assert report.backbone_vocab_size == 248168


def test_o5_rejects_legacy_005a1_rows(tmp_path: Path) -> None:
    """正式 SDK 0.0.5 必须拒绝 0.0.5a1 的 248174 行权重。"""

    pt_path = tmp_path / "o5-legacy.pt"
    _write_checkpoint(pt_path, 248174)
    model_path = tmp_path / "o5-model"
    model_path.mkdir()
    backbone = tmp_path / "o5-backbone"
    backbone.mkdir()
    (backbone / "config.json").write_text(
        json.dumps({"vocab_size": 248174}),
        encoding="utf-8",
    )
    (backbone / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}}),
        encoding="utf-8",
    )
    profile = O5FcDeploymentProfile(
        profile_id="o5-invalid-a1",
        model_path=str(model_path),
        pt_path=str(pt_path),
        backbone_dir=str(backbone),
        unit_policy=_unit_policy(),
    )

    report = validate_fc_checkpoint(profile)

    assert report.valid is False
    assert len(report.errors) == 3
    assert all("248168" in error for error in report.errors)
