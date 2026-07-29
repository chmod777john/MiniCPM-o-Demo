"""用一个模型目录启动 O5 TP2 API。

本模块是 Demo 仓库面向部署使用者的公开启动入口。使用者只感知模型目录、
可选存储目录和 Gateway 端口；基础模型、SDK、TP2、Graph、内部端口与服务编排
统一由仓库内的内部设置管理。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, NoReturn, Sequence

from minicpm_o5_sdk import O5UnitPolicy
from pydantic import BaseModel, ConfigDict, Field

from core.fc_duplex.profiles import (
    O5FcDeploymentProfile,
    load_fc_deployment_profile,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INTERNAL_SETTINGS_PATH = (
    PROJECT_DIR
    / "configs"
    / "fc_deployment"
    / "o5_api_internal_settings.json"
)


class O5ApiLaunchSettings(BaseModel):
    """部署使用者可以感知的全部启动设置。

    参数:
        model_path: 包含完整 PT 和 TP2 backbone 的模型目录。
        storage_dir: 日志、Session、cache 与内部运行文件的统一目录。
        port: Gateway 对外监听端口。
        check_only: 是否只校验和生成运行文件而不启动服务。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: Path
    storage_dir: Path | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    check_only: bool = False


class O5ApiRuntimeSettings(BaseModel):
    """Demo 维护者管理的 O5 API 内部固定设置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["o5_api_internal_settings.v1"] = (
        "o5_api_internal_settings.v1"
    )
    runtime_python: Path
    base_model_path: Path
    sdk_version: Literal["0.0.5"] = "0.0.5"
    unit_policy: O5UnitPolicy
    non_spoken_scheduling: Literal["quality", "latency"] = "quality"
    llm_cache: int = Field(default=32768, ge=1)
    spmd_heartbeat_interval_sec: float = Field(default=30.0, gt=0)
    attn_implementation: Literal[
        "auto",
        "flash_attention_2",
        "sdpa",
        "eager",
    ] = "auto"
    gateway_host: str = "0.0.0.0"
    gateway_port: int = Field(default=8009, ge=1, le=65535)
    backend_host: str = "127.0.0.1"
    backend_port: int = Field(default=22510, ge=1, le=65535)
    worker_host: str = "127.0.0.1"
    worker_port: int = Field(default=22410, ge=1, le=65535)
    gateway_internal_port: int = Field(default=8010, ge=1, le=65535)
    gateway_https: Literal[False] = False


class O5ModelAssets(BaseModel):
    """从一个用户模型目录解析出的内部 O5 部署资产。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    model_dir: Path
    pt_path: Path
    backbone_dir: Path


class PreparedO5ApiRuntime(BaseModel):
    """一键入口生成的运行目录和内部配置文件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_dir: Path
    log_dir: Path
    data_dir: Path
    cache_dir: Path
    runtime_dir: Path
    profile_path: Path
    service_config_path: Path
    gateway_port: int


class _ServiceDataSettings(BaseModel):
    """写给现有 ServiceConfig 的最小存储覆盖。"""

    data_dir: str


class _ServiceConfigOverride(BaseModel):
    """一键入口生成的现有服务配置覆盖。"""

    service: _ServiceDataSettings


def load_o5_api_runtime_settings(
    path: Path = DEFAULT_INTERNAL_SETTINGS_PATH,
) -> O5ApiRuntimeSettings:
    """读取由 Demo 维护者管理的内部固定设置。

    参数:
        path: 内部设置 JSON 路径。

    返回:
        经过严格校验的 O5 API 内部设置。
    """

    return O5ApiRuntimeSettings.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _sanitize_model_id(name: str) -> str:
    """把模型目录名转换为稳定且可用于 Profile 的标识。

    参数:
        name: 模型目录名。

    返回:
        仅包含字母、数字、点、下划线和连字符的模型标识。
    """

    model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-_")
    if not model_id:
        raise ValueError(f"无法从模型目录名生成 model_id: {name!r}")
    return model_id


def _discover_pt_path(model_dir: Path) -> Path:
    """发现模型目录内唯一的完整 PT。

    参数:
        model_dir: 已确认存在的模型目录。

    返回:
        标准 ``model.pt`` 或唯一的顶层 ``*.pt``。
    """

    canonical = model_dir / "model.pt"
    if canonical.is_file():
        return canonical.resolve()

    candidates = sorted(path for path in model_dir.glob("*.pt") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"模型目录缺少完整 PT；期望 {canonical} 或唯一的顶层 *.pt"
        )
    if len(candidates) > 1:
        raise ValueError(
            "模型目录找到多个 PT，启动器不会猜测: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0].resolve()


def _is_backbone_dir(path: Path) -> bool:
    """判断目录是否包含 O5 TP2 backbone 的必要 HF 文件。

    参数:
        path: 待检查目录。

    返回:
        同时存在 config 和 safetensors index 时返回 True。
    """

    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "model.safetensors.index.json").is_file()
    )


def _discover_backbone_dir(model_dir: Path) -> Path:
    """发现模型目录内唯一的 TP2 backbone。

    参数:
        model_dir: 已确认存在的模型目录。

    返回:
        标准 ``backbone`` 或唯一满足 HF 契约的一级子目录。
    """

    canonical = model_dir / "backbone"
    if _is_backbone_dir(canonical):
        return canonical.resolve()

    candidates = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_dir() and _is_backbone_dir(path)
    )
    if not candidates:
        raise FileNotFoundError(
            "模型目录缺少 TP2 backbone；期望 backbone/ 或唯一包含 "
            "config.json 与 model.safetensors.index.json 的一级子目录"
        )
    if len(candidates) > 1:
        raise ValueError(
            "模型目录找到多个 backbone，启动器不会猜测: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0].resolve()


def discover_o5_model_assets(model_path: Path) -> O5ModelAssets:
    """从一个模型目录解析完整 PT 和 TP2 backbone。

    参数:
        model_path: 用户提供的模型目录。

    返回:
        已解析的 O5 模型资产。

    异常:
        FileNotFoundError: 模型目录或必要资产不存在。
        ValueError: PT 或 backbone 不唯一。
    """

    model_dir = model_path.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型路径不存在或不是目录: {model_dir}")

    return O5ModelAssets(
        model_id=_sanitize_model_id(model_dir.name),
        model_dir=model_dir,
        pt_path=_discover_pt_path(model_dir),
        backbone_dir=_discover_backbone_dir(model_dir),
    )


def _default_storage_dir(model_id: str) -> Path:
    """生成仓库内默认运行目录。

    参数:
        model_id: 已清洗的模型标识。

    返回:
        带 UTC 时间戳的默认运行目录。
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_DIR / "run-logs" / f"{model_id}-{timestamp}"


def _write_json(path: Path, model: BaseModel) -> None:
    """以稳定 UTF-8 格式写入 Pydantic 配置。

    参数:
        path: 输出 JSON 路径。
        model: 待序列化的 Pydantic 模型。

    返回:
        无返回值。
    """

    path.write_text(
        json.dumps(
            model.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_o5_api_runtime(
    *,
    launch: O5ApiLaunchSettings,
    internal: O5ApiRuntimeSettings,
    assets: O5ModelAssets,
) -> PreparedO5ApiRuntime:
    """生成内部 Deployment Profile、ServiceConfig 和运行目录。

    参数:
        launch: 使用者可感知的外层启动设置。
        internal: Demo 维护者管理的内部固定设置。
        assets: 从模型目录解析出的 PT 与 backbone。

    返回:
        可直接投影为服务环境变量的运行描述。
    """

    storage_dir = (
        launch.storage_dir.expanduser().resolve()
        if launch.storage_dir is not None
        else _default_storage_dir(assets.model_id)
    )
    log_dir = storage_dir / "logs"
    data_dir = storage_dir / "data"
    cache_dir = storage_dir / "cache"
    runtime_dir = storage_dir / "runtime"
    for path in (log_dir, data_dir, cache_dir, runtime_dir):
        path.mkdir(parents=True, exist_ok=True)

    profile = O5FcDeploymentProfile(
        profile_id=f"o5-api-{assets.model_id}",
        sdk_version=internal.sdk_version,
        model_path=str(internal.base_model_path),
        pt_path=str(assets.pt_path),
        unit_policy=internal.unit_policy,
        non_spoken_scheduling=internal.non_spoken_scheduling,
        backbone_dir=str(assets.backbone_dir),
        llm_cache=internal.llm_cache,
        spmd_heartbeat_interval_sec=internal.spmd_heartbeat_interval_sec,
        attn_implementation=internal.attn_implementation,
    )
    profile_path = runtime_dir / "deployment_profile.json"
    _write_json(profile_path, profile)

    service_config = _ServiceConfigOverride(
        service=_ServiceDataSettings(data_dir=str(data_dir))
    )
    service_config_path = runtime_dir / "service_config.json"
    _write_json(service_config_path, service_config)

    return PreparedO5ApiRuntime(
        storage_dir=storage_dir,
        log_dir=log_dir,
        data_dir=data_dir,
        cache_dir=cache_dir,
        runtime_dir=runtime_dir,
        profile_path=profile_path,
        service_config_path=service_config_path,
        gateway_port=(
            launch.port
            if launch.port is not None
            else internal.gateway_port
        ),
    )


def validate_o5_api_runtime(
    *,
    internal: O5ApiRuntimeSettings,
    prepared: PreparedO5ApiRuntime,
) -> None:
    """校验复用环境、基础模型资产和生成的内部 Profile。

    参数:
        internal: Demo 内部固定设置。
        prepared: 已生成的运行描述。

    返回:
        无返回值；所有检查通过即返回。
    """

    if not internal.runtime_python.is_file():
        raise FileNotFoundError(
            f"可复用 Python 环境不存在: {internal.runtime_python}"
        )
    if not os.access(internal.runtime_python, os.X_OK):
        raise PermissionError(
            f"可复用 Python 不可执行: {internal.runtime_python}"
        )
    if not internal.base_model_path.is_dir():
        raise FileNotFoundError(
            f"内部基础模型目录不存在: {internal.base_model_path}"
        )

    required_base_assets = (
        internal.base_model_path / "config.json",
        internal.base_model_path / "tokenizer.json",
        internal.base_model_path / "tokenizer_config.json",
        internal.base_model_path / "special_tokens_map.json",
        internal.base_model_path / "preprocessor_config.json",
        internal.base_model_path / "chat_template.jinja",
    )
    missing = [str(path) for path in required_base_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "内部基础模型目录缺少 config/tokenizer/processor 资产: "
            + ", ".join(missing)
        )

    load_fc_deployment_profile(prepared.profile_path)


def build_service_environment(
    *,
    project_dir: Path,
    internal: O5ApiRuntimeSettings,
    prepared: PreparedO5ApiRuntime,
    inherited_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """把两层设置投影为现有内部启动链需要的环境变量。

    参数:
        project_dir: 当前克隆的 Demo 仓库根目录。
        internal: Demo 内部固定设置。
        prepared: 已生成的运行描述。
        inherited_environment: 需要保留的父进程环境；默认使用 os.environ。

    返回:
        可传给子进程的完整环境变量映射。
    """

    environment = dict(
        os.environ if inherited_environment is None else inherited_environment
    )
    existing_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "PROJECT_DIR": str(project_dir),
            "PYTHONPATH": (
                f"{project_dir}:{existing_pythonpath}"
                if existing_pythonpath
                else str(project_dir)
            ),
            "VENV_DIR": str(internal.runtime_python.parent.parent),
            "FC_DEPLOYMENT_PROFILE": str(prepared.profile_path),
            "O5_DEMO_CONFIG_PATH": str(prepared.service_config_path),
            "GATEWAY_HOST": internal.gateway_host,
            "GATEWAY_PORT": str(prepared.gateway_port),
            "GATEWAY_INTERNAL_PORT": str(internal.gateway_internal_port),
            "BACKEND_HOST": internal.backend_host,
            "BACKEND_PORT": str(internal.backend_port),
            "WORKER_HOST": internal.worker_host,
            "WORKER_PORT": str(internal.worker_port),
            "LOG_DIR": str(prepared.log_dir),
            "TORCHINDUCTOR_CACHE_DIR": str(
                prepared.cache_dir / "torchinductor"
            ),
            "GATEWAY_HTTPS": "1" if internal.gateway_https else "0",
        }
    )
    return environment


def _require_tp2_gpus() -> None:
    """确认当前进程至少能看到两张 CUDA GPU。"""

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(
            f"O5 TP2 API 至少需要 2 张可见 CUDA GPU，当前只有 {gpu_count} 张"
        )


def launch_o5_api(
    *,
    internal: O5ApiRuntimeSettings,
    prepared: PreparedO5ApiRuntime,
) -> NoReturn:
    """用可复用 Python 环境替换当前进程并启动 API。

    参数:
        internal: Demo 内部固定设置。
        prepared: 已生成并验证的运行描述。

    返回:
        不返回；成功时当前进程被现有 start_fc_service.py 替换。
    """

    _require_tp2_gpus()
    environment = build_service_environment(
        project_dir=PROJECT_DIR,
        internal=internal,
        prepared=prepared,
    )
    entry = PROJECT_DIR / "scripts" / "start_fc_service.py"
    os.execvpe(
        str(internal.runtime_python),
        [
            str(internal.runtime_python),
            str(entry),
            "--profile",
            str(prepared.profile_path),
        ],
        environment,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """构造面向部署使用者的最小命令行接口。"""

    parser = argparse.ArgumentParser(
        description=(
            "从一个模型目录启动 O5 TP2 API；环境、Profile 和内部服务参数由仓库管理"
        )
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="包含完整 PT 和 TP2 backbone 的模型目录",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="日志、Session、cache 和运行配置目录；默认写入仓库 run-logs/",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Gateway HTTP 端口；默认读取仓库内部设置（当前为 8009）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查模型资产和可复用环境，不启动服务",
    )
    return parser


def _print_ready_summary(
    *,
    assets: O5ModelAssets,
    prepared: PreparedO5ApiRuntime,
) -> None:
    """打印使用者需要感知的部署摘要。

    参数:
        assets: 已解析的模型资产。
        prepared: 已生成的运行描述。

    返回:
        无返回值。
    """

    print(f"[o5-api] model={assets.model_dir}")
    print(f"[o5-api] storage={prepared.storage_dir}")
    print(f"[o5-api] api=http://0.0.0.0:{prepared.gateway_port}")
    print(
        f"[o5-api] websocket=ws://0.0.0.0:{prepared.gateway_port}/v1/realtime"
    )
    print(
        f"[o5-api] health=http://0.0.0.0:{prepared.gateway_port}/health",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析外层设置，校验模型并按需启动 O5 API。

    参数:
        argv: 可选命令行参数序列；None 时读取 sys.argv。

    返回:
        check-only 成功返回 0；参数或环境错误返回 2。正常启动时不返回。
    """

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        launch = O5ApiLaunchSettings(
            model_path=args.model_path,
            storage_dir=args.storage_dir,
            port=args.port,
            check_only=args.check_only,
        )
        internal_settings_path = Path(
            os.environ.get(
                "O5_API_INTERNAL_SETTINGS_PATH",
                DEFAULT_INTERNAL_SETTINGS_PATH,
            )
        )
        internal = load_o5_api_runtime_settings(internal_settings_path)
        assets = discover_o5_model_assets(launch.model_path)
        prepared = prepare_o5_api_runtime(
            launch=launch,
            internal=internal,
            assets=assets,
        )
        validate_o5_api_runtime(internal=internal, prepared=prepared)
        _print_ready_summary(assets=assets, prepared=prepared)
        if launch.check_only:
            print("[o5-api] check passed")
            return 0
        launch_o5_api(internal=internal, prepared=prepared)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[o5-api] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
