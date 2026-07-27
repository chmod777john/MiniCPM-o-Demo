"""SDK 0.0.5 FC checkpoint/Profile 硬兼容性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from core.fc_duplex.profiles import O45FcDeploymentProfile, O5FcDeploymentProfile
from minicpm_o5_sdk import O5UnitPolicy
from tools.checkpoint.convert_mcore_to_pt import (
    _runtime_key,
    convert_model_only_dcp_to_pt,
)
from tools.checkpoint.extract_o5_tp2_backbone import build_o5_text_config
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


def test_o5_backbone_config_strips_multimodal_remote_code_fields() -> None:
    """TP2 backbone 必须保存纯 text config，不能继续引用外部 modeling。"""

    config = build_o5_text_config(
        {
            "auto_map": {"AutoConfig": "configuration_minicpmo.MiniCPMOConfig"},
            "model_type": "minicpmo",
            "text_config": {"vocab_size": 999999},
            "vision_config": {"hidden_size": 1},
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
        },
        expected_rows=248168,
    )
    serialized = config.to_dict()

    assert config.model_type == "qwen3_5_moe_text"
    assert config.vocab_size == 248168
    assert "auto_map" not in serialized
    assert "text_config" not in serialized
    assert "vision_config" not in serialized
