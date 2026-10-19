"""SDK FC checkpoint/Profile 硬兼容性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from core.fc_duplex.profiles import O5FcDeploymentProfile
from minicpm_o5_sdk import O5UnitPolicy
from tools.checkpoint.convert_mcore_to_pt import (
    _runtime_key,
    convert_model_only_dcp_to_pt,
)
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


def test_o5_full_checkpoint_matches_required_rows(tmp_path: Path) -> None:
    """O5 完整 PT 必须使用 248168 rows；独立 backbone 不再参与校验。"""

    pt_path = tmp_path / "o5.pt"
    _write_checkpoint(pt_path, 248168)
    model_path = tmp_path / "o5-model"
    model_path.mkdir()
    profile = O5FcDeploymentProfile(
        profile_id="o5-valid",
        model_path=str(model_path),
        pt_path=str(pt_path),
        unit_policy=_unit_policy(),
    )

    report = validate_fc_checkpoint(profile)

    assert report.valid is True
    assert report.backbone_vocab_size is None


def test_o5_rejects_legacy_005a1_rows(tmp_path: Path) -> None:
    """正式 SDK 必须拒绝 0.0.5a1 的 248174 行权重。"""

    pt_path = tmp_path / "o5-legacy.pt"
    _write_checkpoint(pt_path, 248174)
    model_path = tmp_path / "o5-model"
    model_path.mkdir()
    profile = O5FcDeploymentProfile(
        profile_id="o5-invalid-a1",
        model_path=str(model_path),
        pt_path=str(pt_path),
        unit_policy=_unit_policy(),
    )

    report = validate_fc_checkpoint(profile)

    assert report.valid is False
    assert len(report.errors) == 3
    assert all("248168" in error for error in report.errors)


def test_mcore_model_only_dcp_converts_to_runtime_pt(tmp_path: Path) -> None:
    """DCP converter 应去掉 model.module wrapper 并输出普通 state_dict。"""

    source_dir = tmp_path / "dcp"
    dcp.save(
        {
            "model.module.language_model.embedding.word_embeddings.weight": torch.ones(
                4, 2
            ),
            "model.module.language_model.output_layer.weight": torch.ones(4, 2),
            "model.module.language_model.decoder.final_layernorm.weight": torch.ones(
                2
            ),
            "model.module.apm.weight": torch.ones(2, 2),
            "optimizer.state": torch.ones(1),
        },
        checkpoint_id=source_dir,
    )
    output_pt = tmp_path / "dense.pt"
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "layer_types": [],
                "num_experts": 0,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 1,
                "linear_num_value_heads": 1,
                "linear_value_head_dim": 1,
            }
        ),
        encoding="utf-8",
    )

    manifest = convert_model_only_dcp_to_pt(
        source_dir=source_dir,
        model_path=model_path,
        output_pt=output_pt,
    )
    state = torch.load(output_pt, map_location="cpu", weights_only=True)

    assert manifest.tensor_count == 4
    assert manifest.embedding_rows == 4
    assert manifest.lm_head_rows == 4
    assert "llm.model.embed_tokens.weight" in state
    assert "optimizer.state" not in state
    assert _runtime_key("model.foo") == "foo"
