"""把 Megatron/MCore model-only DCP 转成 Demo 可加载的完整 PyTorch PT。

当前 Demo backend 不直接读取 DCP。本工具在大内存 CPU Job 中把所有 ``model.*``
tensor 恢复为完整 CPU tensor，去掉 Megatron wrapper 前缀后原子写出单个 state_dict
``.pt``。输出可继续用于 O5 TP2 backbone 抽取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.metadata as dcp_metadata
from torch.distributed._shard.sharded_tensor.metadata import MEM_FORMAT_ENCODING
from torch.distributed.checkpoint import default_planner


@dataclass(frozen=True)
class DensePtManifest:
    """一次 DCP → PT 转换的审计摘要。"""

    source_dcp: str
    output_pt: str
    tensor_count: int
    logical_bytes: int
    embedding_rows: int
    lm_head_rows: int
    output_sha256: str


def _install_metadata_compatibility() -> None:
    """兼容较新 PyTorch 写出的 DCP metadata 符号。"""

    if not hasattr(dcp_metadata, "_MEM_FORMAT_ENCODING"):
        dcp_metadata._MEM_FORMAT_ENCODING = MEM_FORMAT_ENCODING
    if not hasattr(dcp_metadata, "StorageMeta"):

        @dataclass
        class StorageMeta:
            """旧 PyTorch 读取新 checkpoint 所需的最小 metadata 类型。"""

            checkpoint_id: Any = None
            size: Any = None

        dcp_metadata.StorageMeta = StorageMeta


def _runtime_key(dcp_key: str) -> str:
    """把 Megatron DCP FQN 映射为 Demo runtime state_dict key。

    参数:
        dcp_key: DCP metadata 中的完整 key。

    返回:
        去掉 ``model.module.`` 或 ``model.`` wrapper 的 runtime key。
    """

    for prefix in ("model.module.", "model."):
        if dcp_key.startswith(prefix):
            return dcp_key[len(prefix) :]
    raise ValueError(f"不是模型 tensor key: {dcp_key}")


def _allocate_model_state(
    metadata: dcp_metadata.Metadata,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """根据 DCP metadata 分配完整 CPU tensor。

    参数:
        metadata: model-only DCP metadata。

    返回:
        DCP key placeholder 与 ``DCP key -> runtime key`` 映射。
    """

    state: dict[str, torch.Tensor] = {}
    key_map: dict[str, str] = {}
    runtime_keys: set[str] = set()
    for dcp_key, entry in sorted(metadata.state_dict_metadata.items()):
        if not dcp_key.startswith("model."):
            continue
        if not isinstance(entry, dcp_metadata.TensorStorageMetadata):
            continue
        runtime_key = _runtime_key(dcp_key)
        if runtime_key in runtime_keys:
            raise ValueError(f"runtime key 映射冲突: {runtime_key}")
        runtime_keys.add(runtime_key)
        key_map[dcp_key] = runtime_key
        state[dcp_key] = torch.empty(
            tuple(int(value) for value in entry.size),
            dtype=entry.properties.dtype,
            device="cpu",
        )
    if not state:
        raise ValueError("DCP 不包含 model tensor")
    return state, key_map


def convert_model_only_dcp_to_pt(
    *,
    source_dir: str | Path,
    output_pt: str | Path,
) -> DensePtManifest:
    """把一个 model-only DCP 转成完整 PT。

    参数:
        source_dir: 含 ``.metadata`` 和 shard 文件的 model-only DCP。
        output_pt: 不允许预先存在的目标 PT 路径。

    返回:
        包含词表行数、逻辑字节和输出 SHA256 的转换 Manifest。
    """

    _install_metadata_compatibility()
    source = Path(source_dir)
    output = Path(output_pt)
    if not source.is_dir() or not (source / ".metadata").is_file():
        raise FileNotFoundError(f"model-only DCP 不完整: {source}")
    if output.exists():
        raise FileExistsError(f"目标 PT 已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    reader = dcp.FileSystemReader(source)
    metadata = reader.read_metadata()
    dcp_state, key_map = _allocate_model_state(metadata)
    try:
        planner = default_planner.DefaultLoadPlanner(allow_partial_load=True)
    except TypeError:
        planner = default_planner.DefaultLoadPlanner()
    dcp.load(
        state_dict=dcp_state,
        storage_reader=reader,
        planner=planner,
    )

    runtime_state = {
        key_map[dcp_key]: tensor
        for dcp_key, tensor in dcp_state.items()
    }
    embed = runtime_state["llm.model.embed_tokens.weight"]
    lm_head = runtime_state["llm.lm_head.weight"]
    logical_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in runtime_state.values()
    )

    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(runtime_state, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = DensePtManifest(
        source_dcp=str(source),
        output_pt=str(output),
        tensor_count=len(runtime_state),
        logical_bytes=logical_bytes,
        embedding_rows=int(embed.shape[0]),
        lm_head_rows=int(lm_head.shape[0]),
        output_sha256=_sha256_file(output),
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sha256_file(path: Path) -> str:
    """流式计算大文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description="Convert Megatron model-only DCP to a dense Demo PT"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-pt", required=True)
    args = parser.parse_args()

    manifest = convert_model_only_dcp_to_pt(
        source_dir=args.source_dir,
        output_pt=args.output_pt,
    )
    print(json.dumps(manifest.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
