"""O5 API v3 副本池入口的 CPU 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.deploy.modes import _checkpoint_vocab_size
from core.fc_duplex.profiles import load_fc_deployment_profile
from scripts.run_o5_api import O5ApiRuntimeSettings
from scripts.run_o5_replica_pool import (
    ReplicaPoolSettings,
    _validate_group_width,
    default_gpu_groups,
    parse_gpu_groups,
    prepare_runtime_plan,
)


def _internal_settings(tmp_path: Path) -> O5ApiRuntimeSettings:
    runtime_python = tmp_path / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.touch()
    runtime_python.chmod(0o755)
    base_model_path = tmp_path / "base-model"
    base_model_path.mkdir()
    return O5ApiRuntimeSettings(
        runtime_python=runtime_python,
        base_model_path=base_model_path,
        unit_policy={
            "unit_sec": 1.0,
            "non_spoken_budgets_while_listening": [45],
            "non_spoken_budgets_while_speaking": [25],
        },
    )


def _pt(tmp_path: Path) -> Path:
    pt_path = tmp_path / "merged_model.pt"
    pt_path.touch()
    return pt_path


def test_parse_gpu_groups_preserves_replica_boundaries() -> None:
    """分号应分隔副本，逗号只分隔一个 TP2 副本内的 GPU。"""

    assert parse_gpu_groups("0;1;2") == ((0,), (1,), (2,))
    assert parse_gpu_groups("0,1;2,3") == ((0, 1), (2, 3))


def test_default_gpu_groups_fill_all_visible_gpus() -> None:
    """省略 --gpu-groups 时单卡每卡一副本、TP2 相邻两卡一副本。"""

    assert default_gpu_groups("single_opt", 8) == tuple((gpu,) for gpu in range(8))
    assert default_gpu_groups("tp2", 4) == ((0, 1), (2, 3))
    with pytest.raises(ValueError, match="整数倍"):
        default_gpu_groups("tp2", 3)


def test_checkpoint_vocab_size_uses_matching_embedding_and_lm_head() -> None:
    """单卡 loader 应在建模前从完整 PT 解析扩展后的词表行数。"""

    assert (
        _checkpoint_vocab_size(
            {
                "llm.model.embed_tokens.weight": torch.empty(7, 2),
                "llm.lm_head.weight": torch.empty(7, 2),
            }
        )
        == 7
    )


def test_checkpoint_vocab_size_rejects_inconsistent_rows() -> None:
    """embedding 与 lm_head 行数不一致时必须 fail-fast。"""

    with pytest.raises(ValueError, match="词表行数不一致"):
        _checkpoint_vocab_size(
            {
                "llm.model.embed_tokens.weight": torch.empty(7, 2),
                "llm.lm_head.weight": torch.empty(8, 2),
            }
        )


@pytest.mark.parametrize(
    ("mode", "raw_groups"),
    [
        ("single_eager", "0,1"),
        ("single_opt", "0,1"),
        ("tp2", "0;1"),
    ],
)
def test_gpu_group_width_must_match_mode(mode: str, raw_groups: str) -> None:
    """单卡与 TP2 模式必须拒绝错误的 GPU 分组宽度。"""

    with pytest.raises(ValueError, match="每个副本必须使用"):
        _validate_group_width(mode, parse_gpu_groups(raw_groups))  # type: ignore[arg-type]


def test_single_opt_plan_is_session_bound_and_uses_requested_cache(
    tmp_path: Path,
) -> None:
    """单卡计划不绑定 UnitPolicy/调度策略，且 llm_cache 由调用方决定。"""

    settings = ReplicaPoolSettings(
        pt_path=_pt(tmp_path),
        deployment_mode="single_opt",
        gpu_groups=((0,), (1,)),
        runtime_dir=tmp_path / "single-runtime",
        llm_cache=65536,
    )

    plan = prepare_runtime_plan(settings=settings, internal=_internal_settings(tmp_path))
    raw_profile = json.loads(plan.profile_path.read_text(encoding="utf-8"))
    profile = load_fc_deployment_profile(plan.profile_path, check_sdk_version=False)

    assert raw_profile["unit_policy"] is None
    assert raw_profile["non_spoken_scheduling"] is None
    assert profile.unit_policy is None
    assert profile.deployment_mode == "single_opt"
    assert profile.tp_size == 1
    assert profile.backbone_dir is None
    assert profile.llm_cache == 65536
    assert plan.start_local_gateway
    assert plan.gateway_url == "http://127.0.0.1:8009"
    assert [replica.backend_port for replica in plan.replicas] == [22510, 22511]
    assert [replica.worker_port for replica in plan.replicas] == [22410, 22411]


def test_external_mode_rejects_backbone(tmp_path: Path) -> None:
    """外部模式不使用 backbone，显式传入时拒绝。"""

    backbone = tmp_path / "backbone"
    settings = ReplicaPoolSettings(
        pt_path=_pt(tmp_path),
        deployment_mode="single_opt",
        gpu_groups=((0,),),
        runtime_dir=tmp_path / "runtime",
        llm_cache=32768,
        backbone_dir=backbone,
    )
    with pytest.raises(ValueError, match="omit --backbone-dir"):
        prepare_runtime_plan(settings=settings, internal=_internal_settings(tmp_path))


def test_tp2_plan_uses_full_pt_and_two_gpu_contract(tmp_path: Path) -> None:
    """TP2 副本池使用完整 PT，不要求独立 backbone。"""

    settings = ReplicaPoolSettings(
        pt_path=_pt(tmp_path),
        deployment_mode="tp2",
        gpu_groups=((0, 1), (2, 3)),
        runtime_dir=tmp_path / "tp2-runtime",
        llm_cache=32768,
    )

    plan = prepare_runtime_plan(settings=settings, internal=_internal_settings(tmp_path))
    profile = json.loads(plan.profile_path.read_text(encoding="utf-8"))

    assert profile["deployment_mode"] == "tp2"
    assert profile["tp_size"] == 2
    assert profile["backbone_dir"] is None
    assert [replica.master_port for replica in plan.replicas] == [29500, 29501]


def test_remote_gateway_plan_registers_advertised_workers(tmp_path: Path) -> None:
    """多机：不起本机 Gateway，Worker 对外监听并以 advertise_host 注册到远端。"""

    settings = ReplicaPoolSettings(
        pt_path=_pt(tmp_path),
        deployment_mode="single_opt",
        gpu_groups=((0,),),
        runtime_dir=tmp_path / "remote-runtime",
        llm_cache=32768,
        remote_gateway="http://10.0.0.1:8010/",
        advertise_host="10.0.0.2",
    )

    plan = prepare_runtime_plan(settings=settings, internal=_internal_settings(tmp_path))

    assert not plan.start_local_gateway
    assert plan.registry_url == "http://10.0.0.1:8010"
    assert plan.worker_host == "0.0.0.0"
    assert plan.replicas[0].worker_id == "o5-single_opt-10-0-0-2-00"


@pytest.mark.parametrize("placement", ["spread", "pack"])
def test_placement_is_written_to_gateway_service_config(tmp_path: Path, placement: str) -> None:
    """--placement 透传到本机 Gateway 的 service.placement；默认 spread。"""

    settings = ReplicaPoolSettings(
        pt_path=_pt(tmp_path),
        deployment_mode="single_opt",
        gpu_groups=((0,),),
        runtime_dir=tmp_path / f"rt-{placement}",
        llm_cache=32768,
        placement=placement,
    )
    plan = prepare_runtime_plan(settings=settings, internal=_internal_settings(tmp_path))
    service = json.loads(plan.service_config_path.read_text(encoding="utf-8"))["service"]
    assert service["placement"] == placement


def test_placement_rejects_unknown_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReplicaPoolSettings(
            pt_path=_pt(tmp_path),
            deployment_mode="single_opt",
            gpu_groups=((0,),),
            runtime_dir=tmp_path / "rt-bad",
            llm_cache=32768,
            placement="nope",
        )
