"""O5 API 一键启动入口的外层参数与内部运行契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.run_o5_api import (
    O5ApiLaunchSettings,
    O5ApiRuntimeSettings,
    build_service_environment,
    discover_o5_model_assets,
    prepare_o5_api_runtime,
)


def _write_backbone(backbone_dir: Path) -> None:
    """创建满足启动器静态校验的最小 backbone 目录。

    参数:
        backbone_dir: 待创建的测试 backbone 目录。

    返回:
        无返回值。
    """

    backbone_dir.mkdir(parents=True)
    (backbone_dir / "config.json").write_text("{}", encoding="utf-8")
    (backbone_dir / "model.safetensors.index.json").write_text(
        "{}",
        encoding="utf-8",
    )


def _runtime_settings(tmp_path: Path) -> O5ApiRuntimeSettings:
    """构造不依赖生产绝对路径的内部运行设置。

    参数:
        tmp_path: pytest 临时目录。

    返回:
        可供单元测试使用的内部运行设置。
    """

    runtime_python = tmp_path / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.touch()
    runtime_python.chmod(0o755)

    base_model_path = tmp_path / "base-model"
    base_model_path.mkdir()
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ):
        (base_model_path / filename).write_text("{}", encoding="utf-8")

    return O5ApiRuntimeSettings(
        runtime_python=runtime_python,
        base_model_path=base_model_path,
        unit_policy={
            "unit_sec": 1.0,
            "non_spoken_budgets_while_listening": [30],
            "non_spoken_budgets_while_speaking": [15],
        },
    )


def test_discover_o5_model_assets_uses_canonical_layout(tmp_path: Path) -> None:
    """标准目录应解析固定的 model.pt 与 backbone。"""

    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    (model_dir / "model.pt").touch()
    _write_backbone(model_dir / "backbone")

    assets = discover_o5_model_assets(model_dir)

    assert assets.model_dir == model_dir.resolve()
    assert assets.pt_path == (model_dir / "model.pt").resolve()
    assert assets.backbone_dir == (model_dir / "backbone").resolve()
    assert assets.model_id == "model-a"


def test_discover_o5_model_assets_supports_unique_legacy_names(
    tmp_path: Path,
) -> None:
    """旧资产只要 PT 和 backbone 唯一，也应由一个模型目录启动。"""

    model_dir = tmp_path / "legacy-model"
    model_dir.mkdir()
    (model_dir / "iter_0000400_o5.pt").touch()
    _write_backbone(model_dir / "backbone_hf_o5deploy")

    assets = discover_o5_model_assets(model_dir)

    assert assets.pt_path.name == "iter_0000400_o5.pt"
    assert assets.backbone_dir.name == "backbone_hf_o5deploy"


def test_discover_o5_model_assets_rejects_ambiguous_pt_files(
    tmp_path: Path,
) -> None:
    """多个非标准 PT 不允许启动器猜测。"""

    model_dir = tmp_path / "ambiguous"
    model_dir.mkdir()
    (model_dir / "a.pt").touch()
    (model_dir / "b.pt").touch()
    _write_backbone(model_dir / "backbone")

    with pytest.raises(ValueError, match="找到多个 PT"):
        discover_o5_model_assets(model_dir)


def test_launch_settings_only_expose_model_storage_port_and_check(
    tmp_path: Path,
) -> None:
    """外层设置只包含部署使用者真正需要感知的字段。"""

    settings = O5ApiLaunchSettings(
        model_path=tmp_path / "model",
        storage_dir=tmp_path / "run",
        port=9000,
        check_only=True,
    )

    assert set(type(settings).model_fields) == {
        "model_path",
        "storage_dir",
        "port",
        "check_only",
    }
    with pytest.raises(ValidationError):
        O5ApiLaunchSettings(model_path=tmp_path, port=70000)


def test_prepare_runtime_hides_profile_and_session_defaults(
    tmp_path: Path,
) -> None:
    """生成的内部 Profile 不应注入参考音频或评测 case。"""

    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    (model_dir / "model.pt").touch()
    _write_backbone(model_dir / "backbone")
    assets = discover_o5_model_assets(model_dir)
    internal = _runtime_settings(tmp_path)
    launch = O5ApiLaunchSettings(
        model_path=model_dir,
        storage_dir=tmp_path / "run",
        port=9000,
    )

    prepared = prepare_o5_api_runtime(
        launch=launch,
        internal=internal,
        assets=assets,
    )
    profile = json.loads(prepared.profile_path.read_text(encoding="utf-8"))
    service_config = json.loads(
        prepared.service_config_path.read_text(encoding="utf-8")
    )

    assert profile["model_path"] == str(internal.base_model_path)
    assert profile["pt_path"] == str(assets.pt_path)
    assert profile["backbone_dir"] == str(assets.backbone_dir)
    assert profile["deployment_mode"] == "tp2"
    assert "reference_audio_path" not in profile
    assert "case_folder" not in profile
    assert service_config == {
        "service": {"data_dir": str(prepared.data_dir)}
    }


def test_prepare_runtime_uses_internal_default_gateway_port(
    tmp_path: Path,
) -> None:
    """使用者不传端口时，应由内部设置提供稳定默认值。"""

    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    (model_dir / "model.pt").touch()
    _write_backbone(model_dir / "backbone")
    assets = discover_o5_model_assets(model_dir)
    internal = _runtime_settings(tmp_path).model_copy(
        update={"gateway_port": 8123}
    )

    prepared = prepare_o5_api_runtime(
        launch=O5ApiLaunchSettings(model_path=model_dir),
        internal=internal,
        assets=assets,
    )

    assert prepared.gateway_port == 8123


def test_service_environment_projects_only_public_overrides(
    tmp_path: Path,
) -> None:
    """外层端口和存储目录应投影到内部环境，其余保持固定设置。"""

    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    (model_dir / "model.pt").touch()
    _write_backbone(model_dir / "backbone")
    assets = discover_o5_model_assets(model_dir)
    internal = _runtime_settings(tmp_path)
    launch = O5ApiLaunchSettings(
        model_path=model_dir,
        storage_dir=tmp_path / "run",
        port=9000,
    )
    prepared = prepare_o5_api_runtime(
        launch=launch,
        internal=internal,
        assets=assets,
    )

    environment = build_service_environment(
        project_dir=tmp_path / "repo",
        internal=internal,
        prepared=prepared,
        inherited_environment={"PATH": "/usr/bin"},
    )

    assert environment["GATEWAY_PORT"] == "9000"
    assert environment["LOG_DIR"] == str(prepared.log_dir)
    assert environment["O5_DEMO_CONFIG_PATH"] == str(
        prepared.service_config_path
    )
    assert environment["ENABLE_FRP"] == "0"
    assert environment["GATEWAY_HTTPS"] == "0"
    assert environment["PATH"] == "/usr/bin"
