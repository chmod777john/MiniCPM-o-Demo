"""统一 FC DeploymentProfile 与双 ModelAdapter 的 CPU 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from core.fc_duplex.model_adapter import (
    O45FcDuplexModelAdapter,
    O5FcDuplexModelAdapter,
    create_fc_duplex_model_adapter,
)
from core.fc_duplex.profiles import (
    O45FcDeploymentProfile,
    O5FcDeploymentProfile,
    apply_fc_deployment_profile_environment,
    load_fc_deployment_profile,
)
from minicpm_o5_sdk import O5TokenizerID, O5UnitPolicy


class _FakeCapability:
    """只提供 SDK tokenizer 和 Resume boundary 的模型能力。"""

    def __init__(self, tokenizer_id: O5TokenizerID) -> None:
        self.protocol_tokenizer = SimpleNamespace(target=tokenizer_id.value)

    def resume_boundary_status(self) -> dict[str, str]:
        """返回可恢复边界。"""

        return {"status": "available"}


class _FakeModel:
    """实现 Adapter 所需 primitive 的最小模型。"""

    def __init__(self, tokenizer_id: O5TokenizerID) -> None:
        self.fc_duplex = _FakeCapability(tokenizer_id)
        self.config = SimpleNamespace(_name_or_path=f"fake-{tokenizer_id.value}")

    def fc_duplex_prepare(self, **_: Any) -> dict[str, Any]:
        return {"prefill_ids": []}

    def fc_duplex_streaming_prefill(self, **_: Any) -> dict[str, Any]:
        return {"unit": 0}

    def fc_duplex_streaming_spoken_generate(self, **_: Any) -> dict[str, Any]:
        return {"spoken_ids": []}

    def fc_duplex_streaming_non_spoken_generate(
        self,
        **_: Any,
    ) -> dict[str, Any]:
        return {"token_ids": []}

    def fc_duplex_finalize_unit(self) -> dict[str, Any]:
        return {"unit": 0}

    def fc_duplex_replay_completed_unit(self, **_: Any) -> dict[str, Any]:
        return {"unit": 0}

    def fc_duplex_decode_output_ids(self, **_: Any) -> dict[str, Any]:
        return {"output_ids": []}

    def fc_duplex_cleanup(self) -> None:
        return None


def _unit_policy() -> O5UnitPolicy:
    """构造两个 Profile 共用的最小 UnitPolicy。"""

    return O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[30],
        non_spoken_budgets_while_speaking=[15],
    )


def test_o45_profile_has_fixed_sdk_target_and_rows() -> None:
    """O45 Profile 必须固定正式 SDK、target 与 required rows。"""

    profile = O45FcDeploymentProfile(
        profile_id="o45-test",
        model_path="/model/o45",
        pt_path="/checkpoint/o45.pt",
        unit_policy=_unit_policy(),
    )

    assert profile.sdk_version == "0.0.5"
    assert profile.tokenizer_target == "o45_fc"
    assert profile.required_model_rows == 151772


def test_o5_profile_has_fixed_tp2_graph_contract() -> None:
    """O5 Profile 必须绑定 O5 target、248168 rows 与 TP2 Graph。"""

    profile = O5FcDeploymentProfile(
        profile_id="o5-test",
        model_path="/model/o5",
        pt_path="/checkpoint/o5.pt",
        backbone_dir="/checkpoint/o5-backbone",
        unit_policy=_unit_policy(),
    )

    assert profile.tokenizer_target == "o5"
    assert profile.required_model_rows == 248168
    assert profile.tp_size == 2
    assert profile.llm_graph is True


def test_profile_rejects_cross_model_target() -> None:
    """Profile 不允许 O5 模型声明 O45 tokenizer target。"""

    with pytest.raises(ValidationError):
        O5FcDeploymentProfile(
            profile_id="bad",
            model_path="/model/o5",
            pt_path="/checkpoint/o5.pt",
            backbone_dir="/checkpoint/o5-backbone",
            tokenizer_target="o45_fc",
            unit_policy=_unit_policy(),
        )


def test_load_profile_uses_discriminated_union(tmp_path: Path) -> None:
    """JSON loader 应按 model_family 返回具体 Profile 类型。"""

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "o45-json",
                "model_family": "o45",
                "model_path": "/model/o45",
                "pt_path": "/checkpoint/o45.pt",
                "unit_policy": _unit_policy().model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    profile = load_fc_deployment_profile(
        profile_path,
        check_paths=False,
        check_sdk_version=False,
    )

    assert isinstance(profile, O45FcDeploymentProfile)


def test_profile_environment_is_server_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile 只投影 backend 环境，不要求 API 调用者传模型参数。"""

    profile = O5FcDeploymentProfile(
        profile_id="o5-env",
        model_path="/model/o5",
        pt_path="/checkpoint/o5.pt",
        backbone_dir="/checkpoint/o5-backbone",
        unit_policy=_unit_policy(),
    )
    for name in (
        "FC_MODEL_FAMILY",
        "CHECKPOINT_PROFILE_ID",
        "FC_DUPLEX_UNIT_POLICY_JSON",
        "FC_DUPLEX_NON_SPOKEN_SCHEDULING",
        "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING",
        "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING",
        "FC_DUPLEX_UNIT_SEC",
        "O5_DEPLOY_MODE",
        "O5_BACKBONE_DIR",
        "O5_LLM_CACHE",
        "O5_LLM_GRAPH",
        "O5_SPMD_HEARTBEAT_INTERVAL",
        "O5_ATTN_IMPLEMENTATION",
        "FC_REFERENCE_AUDIO_PATH",
        "FC_BOARD_CASE_FOLDER",
    ):
        # 先通过 monkeypatch 记录原值，确保被生产函数直接写入的环境在 teardown 恢复。
        monkeypatch.setenv(name, "__test_placeholder__")

    apply_fc_deployment_profile_environment(profile)

    assert __import__("os").environ["FC_MODEL_FAMILY"] == "o5"
    assert __import__("os").environ["O5_DEPLOY_MODE"] == "tp2"


def test_o45_adapter_exposes_matching_tokenizer_and_resume() -> None:
    """O45 Adapter 应使用 O45_FC tokenizer 并保留 Resume。"""

    adapter = O45FcDuplexModelAdapter(_FakeModel(O5TokenizerID.O45_FC))

    assert adapter.protocol_tokenizer.target == "o45_fc"
    assert adapter.resume_boundary_status() == {"status": "available"}


def test_o5_adapter_exposes_matching_tokenizer_and_defers_resume() -> None:
    """O5 MVP Adapter 使用 O5 tokenizer，并显式推迟 Stateless Resume。"""

    adapter = O5FcDuplexModelAdapter(_FakeModel(O5TokenizerID.O5))

    assert adapter.protocol_tokenizer.target == "o5"
    assert adapter.resume_boundary_status() == {
        "status": "unavailable",
        "reason": "resume_not_supported",
    }
    with pytest.raises(NotImplementedError, match="尚未实现 Stateless Resume"):
        adapter.replay_completed_unit()


def test_adapter_factory_never_guesses_model_family() -> None:
    """Factory 只接受 DeploymentProfile 已明确声明的模型族。"""

    model = _FakeModel(O5TokenizerID.O45_FC)
    adapter = create_fc_duplex_model_adapter(model=model, model_family="o45")

    assert isinstance(adapter, O45FcDuplexModelAdapter)
