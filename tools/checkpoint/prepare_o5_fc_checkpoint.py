"""把原始 O5 Megatron DCP 一次性制备为 Demo 可部署资产。

流程固定为 Cao Hao canonical DCP→PT 转换、按 checkpoint 内容决定是否补冻结 TTS、
抽取 SDK 0.0.5 TP2 backbone、校验词表/TTS/backbone，并写出可审计 Manifest。
所有大文件阶段都先写临时路径再原子改名；默认拒绝覆盖，``--resume`` 可安全续跑。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERTER = Path(
    "/user/caohao/scripts/convert_fsdp_dtensor_moe_omni_to_pt.py"
)
DEFAULT_BASE_MODEL = Path(
    "/user/wangchongyi/models/model_tunnel/qwen36_35bA3b_hybrid"
)
DEFAULT_MODEL_PATH = Path("/user/weihongliang/MiniCPM-o-4_6")
DEFAULT_FROZEN_TTS_BASE = Path(
    "/user/sunweiyue/training/o5-mvp-specialized/dense_pt/"
    "base_sft1_iter10000_sdk005/iter_0000000_o5_caohao.pt"
)
DEFAULT_BACKBONE_EXTRACTOR = (
    REPOSITORY_ROOT / "tools/o5deploy/extract_backbone.py"
)
DEFAULT_BACKBONE_PYTHON = Path(
    "/user/weihongliang/"
    "MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/"
    ".venv-accel/bin/python"
)
DEFAULT_BACKBONE_PYTHONPATH = Path(
    "/user/sunweiyue/runtime/unified-fc-sdk005"
)
REQUEST_FILENAME = "preparation_request.json"
MANIFEST_FILENAME = "preparation_manifest.json"


class TtsPolicy(str, Enum):
    """canonical PT 中 TTS 权重的处理策略。"""

    AUTO = "auto"
    CHECKPOINT_ONLY = "checkpoint-only"
    MERGE_FROZEN = "merge-frozen"


class PreparationStage(str, Enum):
    """制备阶段边界（唯一主链路拆分用）。"""

    ALL = "all"
    DCP_TO_PT = "dcp_to_pt"
    PT_TO_BACKBONE = "pt_to_backbone"


@dataclass(frozen=True)
class O5CheckpointPreparationRequest:
    """一次 O5 checkpoint 制备请求。"""

    source_checkpoint_dir: str
    output_dir: str
    base_model_dir: str
    model_path: str
    converter_script: str
    backbone_extractor_script: str
    backbone_python: str
    backbone_pythonpath: str
    tts_policy: str
    frozen_tts_base_pt: str | None
    dtype: str
    mtp_layer: int
    expected_rows: int
    expected_tensor_count: int
    expected_tts_tensor_count: int
    expected_backbone_key_count: int
    expected_backbone_shard_count: int
    keep_intermediate: bool
    stage: str = PreparationStage.ALL.value


@dataclass(frozen=True)
class CheckpointInspection:
    """完整 PT 的轻量结构检查结果。"""

    embedding_rows: int
    lm_head_rows: int
    tensor_count: int
    tts_tensor_count: int


@dataclass(frozen=True)
class BackboneInspection:
    """原始 O5 Demo extractor 产物的硬验收结果。"""

    vocab_size: int
    tensor_count: int
    shard_count: int
    embedding_rows: int
    lm_head_rows: int


@dataclass(frozen=True)
class O5CheckpointPreparationManifest:
    """最终 Demo 资产及其血缘、硬校验结果。"""

    schema_version: int
    request: O5CheckpointPreparationRequest
    deploy_pt: str
    deploy_pt_sha256: str
    deploy_pt_size_bytes: int
    backbone_dir: str
    embedding_rows: int
    lm_head_rows: int
    tensor_count: int
    tts_tensor_count: int
    backbone_tensor_count: int
    backbone_shard_count: int
    backbone_embedding_rows: int
    backbone_lm_head_rows: int
    frozen_tts_merged: bool
    intermediate_pt: str | None


def prepare_o5_fc_checkpoint(
    request: O5CheckpointPreparationRequest,
    *,
    resume: bool = False,
) -> O5CheckpointPreparationManifest:
    """执行 O5 FC Demo checkpoint 制备流程。

    参数:
        request: 已规范化为绝对路径的制备请求；``stage`` 可为
            ``all`` / ``dcp_to_pt`` / ``pt_to_backbone``。
        resume: 是否允许从同一请求留下的完整阶段产物继续执行。

    返回:
        含最终 PT SHA256、词表行数、TTS tensor 数和 backbone 路径的 Manifest。
        ``dcp_to_pt`` 阶段的 backbone 字段为占位 0 / 空目录说明。
    """

    stage = PreparationStage(request.stage)
    _validate_request_inputs(request)
    output_dir = Path(request.output_dir)
    _initialize_output_directory(output_dir, request=request, resume=resume)

    deploy_pt = output_dir / "model.pt"
    intermediate_pt = output_dir / "canonical_llm_only.pt"
    merge_manifest = deploy_pt.with_suffix(deploy_pt.suffix + ".manifest.json")
    frozen_tts_merged = intermediate_pt.is_file() or merge_manifest.is_file()

    if stage is not PreparationStage.PT_TO_BACKBONE and not deploy_pt.is_file():
        if not intermediate_pt.is_file():
            converted_pt = _convert_canonical_checkpoint(request)
            converted_inspection = _inspect_checkpoint(
                converted_pt,
                expected_rows=request.expected_rows,
                expected_tensor_count=None,
                expected_tts_tensor_count=None,
                require_tts=False,
            )
            has_checkpoint_tts = converted_inspection.tts_tensor_count > 0
            _validate_tts_policy(
                policy=TtsPolicy(request.tts_policy),
                has_checkpoint_tts=has_checkpoint_tts,
                frozen_tts_base_pt=request.frozen_tts_base_pt,
            )
            if has_checkpoint_tts:
                converted_pt.replace(deploy_pt)
            else:
                converted_pt.replace(intermediate_pt)

        if intermediate_pt.is_file() and not deploy_pt.is_file():
            _merge_frozen_tts(
                model_pt=intermediate_pt,
                base_pt=_require_frozen_tts_base(request),
                output_pt=deploy_pt,
            )
            frozen_tts_merged = True

    if not deploy_pt.is_file():
        raise FileNotFoundError(f"缺少 model.pt，无法继续 stage={stage.value}: {deploy_pt}")

    inspection = _inspect_checkpoint(
        deploy_pt,
        expected_rows=request.expected_rows,
        expected_tensor_count=request.expected_tensor_count,
        expected_tts_tensor_count=request.expected_tts_tensor_count,
        require_tts=True,
    )

    kept_intermediate: str | None = None
    if intermediate_pt.is_file():
        if request.keep_intermediate or stage is PreparationStage.DCP_TO_PT:
            kept_intermediate = str(intermediate_pt.resolve())
        else:
            intermediate_pt.unlink()

    if stage is PreparationStage.DCP_TO_PT:
        manifest = O5CheckpointPreparationManifest(
            schema_version=1,
            request=request,
            deploy_pt=str(deploy_pt.resolve()),
            deploy_pt_sha256=_sha256_file(deploy_pt),
            deploy_pt_size_bytes=deploy_pt.stat().st_size,
            backbone_dir="",
            embedding_rows=inspection.embedding_rows,
            lm_head_rows=inspection.lm_head_rows,
            tensor_count=inspection.tensor_count,
            tts_tensor_count=inspection.tts_tensor_count,
            backbone_tensor_count=0,
            backbone_shard_count=0,
            backbone_embedding_rows=0,
            backbone_lm_head_rows=0,
            frozen_tts_merged=frozen_tts_merged,
            intermediate_pt=kept_intermediate,
        )
        _write_json_atomic(output_dir / MANIFEST_FILENAME, asdict(manifest))
        return manifest

    backbone_dir = _prepare_backbone(
        request=request,
        deploy_pt=deploy_pt,
        output_dir=output_dir,
        resume=resume,
    )
    backbone_inspection = _validate_backbone(
        backbone_dir,
        expected_rows=request.expected_rows,
        expected_tensor_count=request.expected_backbone_key_count,
        expected_shard_count=request.expected_backbone_shard_count,
    )

    if intermediate_pt.is_file() and not request.keep_intermediate:
        intermediate_pt.unlink()
        kept_intermediate = None

    manifest = O5CheckpointPreparationManifest(
        schema_version=1,
        request=request,
        deploy_pt=str(deploy_pt.resolve()),
        deploy_pt_sha256=_sha256_file(deploy_pt),
        deploy_pt_size_bytes=deploy_pt.stat().st_size,
        backbone_dir=str(backbone_dir.resolve()),
        embedding_rows=inspection.embedding_rows,
        lm_head_rows=inspection.lm_head_rows,
        tensor_count=inspection.tensor_count,
        tts_tensor_count=inspection.tts_tensor_count,
        backbone_tensor_count=backbone_inspection.tensor_count,
        backbone_shard_count=backbone_inspection.shard_count,
        backbone_embedding_rows=backbone_inspection.embedding_rows,
        backbone_lm_head_rows=backbone_inspection.lm_head_rows,
        frozen_tts_merged=frozen_tts_merged,
        intermediate_pt=kept_intermediate,
    )
    _write_json_atomic(output_dir / MANIFEST_FILENAME, asdict(manifest))
    return manifest


def _validate_request_inputs(request: O5CheckpointPreparationRequest) -> None:
    """在执行昂贵转换前校验所有输入路径和策略组合。"""

    stage = PreparationStage(request.stage)
    if stage is not PreparationStage.PT_TO_BACKBONE:
        _require_directory(Path(request.source_checkpoint_dir), "原始 DCP checkpoint")
        converter = Path(request.converter_script)
        if not converter.is_file():
            raise FileNotFoundError(f"canonical converter 不存在: {converter}")
        _require_directory(Path(request.base_model_dir), "canonical base model")
    else:
        deploy_pt = Path(request.output_dir) / "model.pt"
        if not deploy_pt.is_file():
            raise FileNotFoundError(
                f"pt_to_backbone 需要已有 model.pt: {deploy_pt}"
            )
    # model_path / backbone 工具仅 pt_to_backbone|all 需要；dcp_to_pt 在 para 上不应依赖它们
    if stage is not PreparationStage.DCP_TO_PT:
        _require_directory(Path(request.model_path), "O5 model_path")
        extractor = Path(request.backbone_extractor_script)
        if not extractor.is_file():
            raise FileNotFoundError(f"原始 O5 backbone extractor 不存在: {extractor}")
        backbone_python = Path(request.backbone_python)
        if not backbone_python.is_file():
            raise FileNotFoundError(f"backbone Python 不存在: {backbone_python}")
        _require_directory(
            Path(request.backbone_pythonpath),
            "backbone SDK PYTHONPATH",
        )
    positive_fields = {
        "expected_rows": request.expected_rows,
        "expected_tensor_count": request.expected_tensor_count,
        "expected_tts_tensor_count": request.expected_tts_tensor_count,
        "expected_backbone_key_count": request.expected_backbone_key_count,
        "expected_backbone_shard_count": request.expected_backbone_shard_count,
    }
    for field_name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"{field_name} 必须大于 0")
    if request.mtp_layer < 0:
        raise ValueError("mtp_layer 不能小于 0")
    policy = TtsPolicy(request.tts_policy)
    if policy is TtsPolicy.MERGE_FROZEN:
        _require_frozen_tts_base(request)
    elif (
        policy is not TtsPolicy.CHECKPOINT_ONLY
        and request.frozen_tts_base_pt is not None
    ):
        # checkpoint-only 不消费 frozen TTS；para 上也不应要求 70G clean-base 存在
        frozen_base = Path(request.frozen_tts_base_pt)
        if not frozen_base.is_file():
            raise FileNotFoundError(f"冻结 TTS base PT 不存在: {frozen_base}")


def _initialize_output_directory(
    output_dir: Path,
    *,
    request: O5CheckpointPreparationRequest,
    resume: bool,
) -> None:
    """创建新输出目录，或验证续跑请求与原请求完全一致。"""

    request_path = output_dir / REQUEST_FILENAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"输出目录非空，拒绝覆盖；确认同一任务后可使用 --resume: {output_dir}"
            )
        if not request_path.is_file():
            raise FileNotFoundError(
                f"续跑目录缺少 {REQUEST_FILENAME}，无法证明资产血缘: {output_dir}"
            )
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        current = asdict(request)
        if previous != current:
            if _is_safe_resume_request_correction(
                previous=previous,
                current=current,
                output_dir=output_dir,
            ):
                _write_json_atomic(request_path, current)
            else:
                raise ValueError(
                    "续跑请求与原请求不一致，拒绝混合资产:\n"
                    f"previous={json.dumps(previous, ensure_ascii=False, sort_keys=True)}\n"
                    f"current={json.dumps(current, ensure_ascii=False, sort_keys=True)}"
                )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(request_path, asdict(request))


def _is_safe_resume_request_correction(
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    output_dir: Path,
) -> bool:
    """判断能否在不污染已发布阶段的前提下修正请求并续跑。

    参数:
        previous: 首次执行落盘的请求。
        current: 本次 ``--resume`` 请求。
        output_dir: 当前制备目录。

    返回:
        TTS 策略仅能在最终 PT 发布前修正；backbone Python 仅能在 backbone 发布前修正。
    """

    changed_fields = {
        key
        for key in set(previous) | set(current)
        if previous.get(key) != current.get(key)
    }
    if not changed_fields:
        return True
    has_model_pt = (output_dir / "model.pt").is_file()
    has_backbone = (output_dir / "backbone").exists()
    # model.pt 产出前：允许修正 para-first / checkpoint-only 相关字段
    pre_pt_mutable = {
        "stage",
        "source_checkpoint_dir",
        "frozen_tts_base_pt",
        "base_model_dir",
        "tts_policy",
        "model_path",
        "backbone_python",
        "backbone_pythonpath",
        "backbone_extractor_script",
    }
    if not has_model_pt and changed_fields <= pre_pt_mutable:
        return True
    # 主链路拆分：model.pt 已在后允许 dcp_to_pt → pt_to_backbone
    #（stage / 源占位 / para stub vs 廊坊完整 base_model_dir）
    post_pt_stage_split = {
        "stage",
        "source_checkpoint_dir",
        "base_model_dir",
    }
    if has_model_pt and not has_backbone and changed_fields <= post_pt_stage_split:
        return True
    if changed_fields <= {"stage", "source_checkpoint_dir"}:
        return has_model_pt
    if changed_fields == {"tts_policy"}:
        return not has_model_pt and not has_backbone
    if changed_fields == {"backbone_python"}:
        return not has_backbone
    return False


def _convert_canonical_checkpoint(
    request: O5CheckpointPreparationRequest,
) -> Path:
    """调用 Cao Hao canonical converter，并原子产出待检查 PT。"""

    output_dir = Path(request.output_dir)
    temporary_pt = output_dir / ".canonical.pt.part"
    if temporary_pt.is_file():
        try:
            _inspect_checkpoint(
                temporary_pt,
                expected_rows=request.expected_rows,
                expected_tensor_count=None,
                expected_tts_tensor_count=None,
                require_tts=False,
            )
        except Exception as error:
            print(
                f"[resume] canonical 临时 PT 不完整，将重新转换: {error}",
                flush=True,
            )
            temporary_pt.unlink()
        else:
            print(
                f"[resume] 复用已完成 canonical PT: {temporary_pt}",
                flush=True,
            )
            return temporary_pt
    command = [
        sys.executable,
        request.converter_script,
        "--ckpt-dir",
        request.source_checkpoint_dir,
        "--output-path",
        str(temporary_pt),
        "--base-model-dir",
        request.base_model_dir,
        "--dtype",
        request.dtype,
        "--mtp-layer",
        str(request.mtp_layer),
        # para-first：源集群只需 config + rotary stub；完整 base 校验在廊坊侧资产上已证明过
        "--skip-validate",
    ]
    _run_command(command, cwd=REPOSITORY_ROOT)
    if not temporary_pt.is_file():
        raise FileNotFoundError(f"canonical converter 未生成 PT: {temporary_pt}")
    return temporary_pt


def _validate_tts_policy(
    *,
    policy: TtsPolicy,
    has_checkpoint_tts: bool,
    frozen_tts_base_pt: str | None,
) -> None:
    """校验 checkpoint 实际内容与显式 TTS 策略一致。"""

    if policy is TtsPolicy.CHECKPOINT_ONLY and not has_checkpoint_tts:
        raise ValueError(
            "checkpoint-only 策略要求原始 checkpoint 包含 tts.*，实际为 0"
        )
    if policy is TtsPolicy.MERGE_FROZEN and has_checkpoint_tts:
        raise ValueError(
            "merge-frozen 策略禁止处理已包含 tts.* 的 checkpoint，避免覆盖已训练 TTS"
        )
    if not has_checkpoint_tts and frozen_tts_base_pt is None:
        raise ValueError(
            "原始 checkpoint 不含 tts.*；必须传 --frozen-tts-base-pt，"
            "或使用包含 TTS supervision 的 checkpoint"
        )


def _require_frozen_tts_base(
    request: O5CheckpointPreparationRequest,
) -> Path:
    """取得并校验冻结 TTS clean-base PT。"""

    if request.frozen_tts_base_pt is None:
        raise ValueError("当前 TTS 策略必须传 --frozen-tts-base-pt")
    path = Path(request.frozen_tts_base_pt)
    if not path.is_file():
        raise FileNotFoundError(f"冻结 TTS base PT 不存在: {path}")
    return path


def _merge_frozen_tts(
    *,
    model_pt: Path,
    base_pt: Path,
    output_pt: Path,
) -> None:
    """在独立进程中补回同源 clean-base 的冻结 TTS。"""

    script = REPOSITORY_ROOT / "tools/checkpoint/merge_frozen_tts.py"
    _run_command(
        [
            sys.executable,
            str(script),
            "--model-pt",
            str(model_pt),
            "--base-pt",
            str(base_pt),
            "--output-pt",
            str(output_pt),
        ],
        cwd=REPOSITORY_ROOT,
    )
    if not output_pt.is_file():
        raise FileNotFoundError(f"冻结 TTS 合并未生成 PT: {output_pt}")


def _prepare_backbone(
    *,
    request: O5CheckpointPreparationRequest,
    deploy_pt: Path,
    output_dir: Path,
    resume: bool,
) -> Path:
    """抽取 TP2 backbone；完整目录通过原子改名发布。"""

    backbone_dir = output_dir / "backbone"
    if backbone_dir.exists():
        if not resume:
            raise FileExistsError(f"backbone 已存在，拒绝覆盖: {backbone_dir}")
        manifest_path = backbone_dir / "fc_backbone_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"backbone 缺少制备 Manifest: {manifest_path}")
        _validate_backbone(
            backbone_dir,
            expected_rows=request.expected_rows,
            expected_tensor_count=request.expected_backbone_key_count,
            expected_shard_count=request.expected_backbone_shard_count,
        )
        return backbone_dir

    reusable = _find_reusable_backbone_temporary(
        output_dir=output_dir,
        request=request,
    )
    if reusable is not None:
        print(f"[resume] 复用已完成 backbone 临时目录: {reusable}", flush=True)
        reusable.replace(backbone_dir)
        return backbone_dir

    temporary_dir = output_dir / f".backbone.part.{uuid.uuid4().hex}"
    _run_command(
        [
            request.backbone_python,
            request.backbone_extractor_script,
        ],
        cwd=REPOSITORY_ROOT,
        extra_env={
            "WORKTREE": str(REPOSITORY_ROOT),
            "MODEL_PATH": request.model_path,
            "PT_PATH": str(deploy_pt),
            "BACKBONE_DIR": str(temporary_dir),
            "ATTN_IMPLEMENTATION": "sdpa",
            "PYTHONPATH": request.backbone_pythonpath,
        },
    )
    backbone_inspection = _validate_backbone(
        temporary_dir,
        expected_rows=request.expected_rows,
        expected_tensor_count=request.expected_backbone_key_count,
        expected_shard_count=request.expected_backbone_shard_count,
    )
    _write_json_atomic(
        temporary_dir / "fc_backbone_manifest.json",
        {
            "schema_version": 1,
            "extractor": request.backbone_extractor_script,
            "source_pt": str(deploy_pt.resolve()),
            "model_path": request.model_path,
            **asdict(backbone_inspection),
        },
    )
    temporary_dir.replace(backbone_dir)
    return backbone_dir


def _find_reusable_backbone_temporary(
    *,
    output_dir: Path,
    request: O5CheckpointPreparationRequest,
) -> Path | None:
    """查找、验收并补齐上次中断留下的完整 backbone 临时目录。"""

    completed: list[tuple[Path, BackboneInspection]] = []
    for candidate in sorted(output_dir.glob(".backbone.part*")):
        if not candidate.is_dir():
            continue
        try:
            inspection = _validate_backbone(
                candidate,
                expected_rows=request.expected_rows,
                expected_tensor_count=request.expected_backbone_key_count,
                expected_shard_count=request.expected_backbone_shard_count,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        completed.append((candidate, inspection))
    if not completed:
        return None
    if len(completed) > 1:
        paths = [str(path) for path, _ in completed]
        raise RuntimeError(f"发现多个完整 backbone 临时目录，拒绝猜测: {paths}")

    candidate, inspection = completed[0]
    _write_json_atomic(
        candidate / "fc_backbone_manifest.json",
        {
            "schema_version": 1,
            "extractor": request.backbone_extractor_script,
            "source_pt": str((output_dir / "model.pt").resolve()),
            "model_path": request.model_path,
            **asdict(inspection),
        },
    )
    return candidate


def _inspect_checkpoint(
    pt_path: Path,
    *,
    expected_rows: int,
    expected_tensor_count: int | None,
    expected_tts_tensor_count: int | None,
    require_tts: bool,
) -> CheckpointInspection:
    """使用 mmap 读取 PT 元数据并执行 SDK 0.0.5 硬门禁。"""

    if not pt_path.is_file():
        raise FileNotFoundError(f"checkpoint PT 不存在: {pt_path}")
    state = _load_checkpoint_state_dict(pt_path)
    embed = _require_tensor(state, "llm.model.embed_tokens.weight")
    lm_head = _require_tensor(state, "llm.lm_head.weight")
    inspection = CheckpointInspection(
        embedding_rows=int(embed.shape[0]),
        lm_head_rows=int(lm_head.shape[0]),
        tensor_count=len(state),
        tts_tensor_count=sum(key.startswith("tts.") for key in state),
    )
    if (
        inspection.embedding_rows != expected_rows
        or inspection.lm_head_rows != expected_rows
    ):
        raise ValueError(
            "PT vocab rows 不符合 SDK 0.0.5: "
            f"expected={expected_rows}, embedding={inspection.embedding_rows}, "
            f"lm_head={inspection.lm_head_rows}"
        )
    if require_tts and inspection.tts_tensor_count == 0:
        raise ValueError("最终部署 PT 不包含 tts.* tensor")
    if (
        expected_tensor_count is not None
        and inspection.tensor_count != expected_tensor_count
    ):
        raise ValueError(
            "最终部署 PT tensor 数不一致: "
            f"expected={expected_tensor_count}, actual={inspection.tensor_count}"
        )
    if (
        expected_tts_tensor_count is not None
        and inspection.tts_tensor_count != expected_tts_tensor_count
    ):
        raise ValueError(
            "最终部署 PT tts.* tensor 数不一致: "
            f"expected={expected_tts_tensor_count}, "
            f"actual={inspection.tts_tensor_count}"
        )
    return inspection


def _load_checkpoint_state_dict(pt_path: Path) -> dict[str, Any]:
    """读取 PT，并去除常见 state_dict wrapper。"""

    try:
        state = torch.load(
            pt_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state = torch.load(pt_path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        nested = state.get(key) if isinstance(state, dict) else None
        if isinstance(nested, dict):
            return nested
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 顶层必须是 dict: {type(state)!r}")
    return state


def _require_tensor(state: dict[str, Any], key: str) -> Any:
    """读取 checkpoint 必需 tensor。"""

    value = state.get(key)
    if value is None or not hasattr(value, "shape"):
        raise KeyError(f"checkpoint 缺少 tensor: {key}")
    return value


def _validate_backbone(
    backbone_dir: Path,
    *,
    expected_rows: int,
    expected_tensor_count: int,
    expected_shard_count: int,
) -> BackboneInspection:
    """校验原始 O5 Demo backbone 的 tensor、分片及真实词表形状。"""

    config_path = backbone_dir / "config.json"
    index_path = backbone_dir / "model.safetensors.index.json"
    for path in (config_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(f"backbone 缺少必需文件: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    actual_rows = int(config["vocab_size"])
    if actual_rows != expected_rows:
        raise ValueError(
            "backbone vocab_size 不符合 SDK 0.0.5: "
            f"expected={expected_rows}, actual={actual_rows}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("backbone safetensors index 缺少 weight_map object")
    tensor_count = len(weight_map)
    shard_count = len(set(weight_map.values()))
    if tensor_count != expected_tensor_count:
        raise ValueError(
            "backbone tensor 数不一致: "
            f"expected={expected_tensor_count}, actual={tensor_count}"
        )
    if shard_count != expected_shard_count:
        raise ValueError(
            "backbone safetensors shard 数不一致: "
            f"expected={expected_shard_count}, actual={shard_count}"
        )
    embedding_key = _select_backbone_tensor_key(
        weight_map,
        (
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        label="embedding",
    )
    lm_head_key = "lm_head.weight"
    embedding_rows = _read_safetensor_rows(
        backbone_dir,
        weight_map,
        embedding_key,
    )
    lm_head_rows = _read_safetensor_rows(
        backbone_dir,
        weight_map,
        lm_head_key,
    )
    if embedding_rows != expected_rows or lm_head_rows != expected_rows:
        raise ValueError(
            "backbone 实际 tensor rows 不符合 SDK 0.0.5: "
            f"expected={expected_rows}, embedding={embedding_rows}, "
            f"lm_head={lm_head_rows}"
        )
    return BackboneInspection(
        vocab_size=actual_rows,
        tensor_count=tensor_count,
        shard_count=shard_count,
        embedding_rows=embedding_rows,
        lm_head_rows=lm_head_rows,
    )


def _select_backbone_tensor_key(
    weight_map: dict[str, Any],
    candidates: tuple[str, ...],
    *,
    label: str,
) -> str:
    """从原始与新版 HF key 形式中选择实际存在的 backbone tensor key。"""

    for key in candidates:
        if key in weight_map:
            return key
    raise KeyError(f"backbone index 缺少 {label} tensor，候选 key={candidates}")


def _read_safetensor_rows(
    backbone_dir: Path,
    weight_map: dict[str, Any],
    tensor_key: str,
) -> int:
    """从 safetensors header 读取指定二维权重的第一维。"""

    shard_name = weight_map.get(tensor_key)
    if not isinstance(shard_name, str):
        raise KeyError(f"backbone index 缺少 tensor: {tensor_key}")
    from safetensors import safe_open

    shard_path = backbone_dir / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as stream:
        shape = stream.get_slice(tensor_key).get_shape()
    if len(shape) != 2:
        raise ValueError(f"backbone tensor 不是二维权重: {tensor_key} shape={shape}")
    return int(shape[0])


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """运行一个重型独立阶段，并显式传递仓库 PYTHONPATH。"""

    printable = " ".join(str(part) for part in command)
    print(f"[run] {printable}", flush=True)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{cwd}:{existing_pythonpath}" if existing_pythonpath else str(cwd)
    )
    if extra_env is not None:
        environment.update(extra_env)
    subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=environment,
        check=True,
    )


def _require_directory(path: Path, label: str) -> None:
    """要求输入路径是已存在目录。"""

    if not path.is_dir():
        raise FileNotFoundError(f"{label} 目录不存在: {path}")


def _sha256_file(path: Path) -> str:
    """流式计算大文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """以 UTF-8 原子写入 JSON。"""

    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _absolute_path(value: str | Path) -> str:
    """把 CLI 路径规范化为绝对路径字符串。"""

    return str(Path(value).expanduser().resolve())


def _executable_path(value: str | Path) -> str:
    """把可执行文件转成绝对路径，同时保留虚拟环境 Python symlink。

    参数:
        value: CLI 传入的可执行文件路径。

    返回:
        不解析最终 symlink 的绝对路径；Python 依靠该路径旁的 ``pyvenv.cfg`` 识别环境。
    """

    return str(Path(value).expanduser().absolute())


def _build_request(args: argparse.Namespace) -> O5CheckpointPreparationRequest:
    """从已解析 CLI 参数构造稳定、可比对的请求对象。"""

    stage = PreparationStage(args.stage)
    if stage is not PreparationStage.PT_TO_BACKBONE and not args.source_checkpoint_dir:
        raise ValueError("stage=dcp_to_pt/all 时必须提供 --source-checkpoint-dir")
    source_dir = (
        _absolute_path(args.source_checkpoint_dir)
        if args.source_checkpoint_dir
        else _absolute_path(args.output_dir)
    )
    return O5CheckpointPreparationRequest(
        source_checkpoint_dir=source_dir,
        output_dir=_absolute_path(args.output_dir),
        base_model_dir=_absolute_path(args.base_model_dir),
        model_path=_absolute_path(args.model_path),
        converter_script=_absolute_path(args.converter_script),
        backbone_extractor_script=_absolute_path(args.backbone_extractor_script),
        backbone_python=_executable_path(args.backbone_python),
        backbone_pythonpath=_absolute_path(args.backbone_pythonpath),
        tts_policy=args.tts_policy,
        frozen_tts_base_pt=(
            None
            if TtsPolicy(args.tts_policy) is TtsPolicy.CHECKPOINT_ONLY
            else (
                _absolute_path(args.frozen_tts_base_pt)
                if args.frozen_tts_base_pt is not None
                else None
            )
        ),
        dtype=args.dtype,
        mtp_layer=args.mtp_layer,
        expected_rows=args.expected_rows,
        expected_tensor_count=args.expected_tensor_count,
        expected_tts_tensor_count=args.expected_tts_tensor_count,
        expected_backbone_key_count=args.expected_backbone_key_count,
        expected_backbone_shard_count=args.expected_backbone_shard_count,
        keep_intermediate=args.keep_intermediate,
        stage=stage.value,
    )


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare deployable SDK 0.0.5 O5 PT + TP2 backbone from Megatron DCP"
        )
    )
    parser.add_argument(
        "--source-checkpoint-dir",
        default=None,
        help="Formal DCP iter 目录；stage=pt_to_backbone 时可省略",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in PreparationStage],
        default=PreparationStage.ALL.value,
        help="all=完整；dcp_to_pt=只产出 model.pt；pt_to_backbone=只抽 backbone",
    )
    parser.add_argument("--base-model-dir", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--converter-script", default=str(DEFAULT_CONVERTER))
    parser.add_argument(
        "--backbone-extractor-script",
        default=str(DEFAULT_BACKBONE_EXTRACTOR),
    )
    parser.add_argument("--backbone-python", default=str(DEFAULT_BACKBONE_PYTHON))
    parser.add_argument(
        "--backbone-pythonpath",
        default=str(DEFAULT_BACKBONE_PYTHONPATH),
    )
    parser.add_argument(
        "--tts-policy",
        choices=[policy.value for policy in TtsPolicy],
        default=TtsPolicy.AUTO.value,
    )
    parser.add_argument(
        "--frozen-tts-base-pt",
        default=str(DEFAULT_FROZEN_TTS_BASE),
        help=(
            "LLM-only checkpoint 缺少 tts.* 时使用的同源 clean-base canonical PT；"
            "当前 REF-AUDIO 血缘已有默认值"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--mtp-layer", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, default=248168)
    parser.add_argument("--expected-tensor-count", type=int, default=1748)
    parser.add_argument("--expected-tts-tensor-count", type=int, default=198)
    parser.add_argument("--expected-backbone-key-count", type=int, default=693)
    parser.add_argument("--expected-backbone-shard-count", type=int, default=16)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = prepare_o5_fc_checkpoint(
        _build_request(args),
        resume=args.resume,
    )
    print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
