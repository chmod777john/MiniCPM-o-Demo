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
from collections import OrderedDict
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


def _split_qkv_with_gate(
    weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """拆分 Megatron full-attention Q/Gate/K/V 交错权重。"""

    hidden_size = int(weight.shape[1])
    heads_per_group = num_heads // num_kv_heads
    q_size = heads_per_group * head_dim
    group_size = q_size * 2 + head_dim * 2
    q_parts: list[torch.Tensor] = []
    gate_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for group in range(num_kv_heads):
        offset = group * group_size
        q_parts.append(weight[offset : offset + q_size])
        offset += q_size
        gate_parts.append(weight[offset : offset + q_size])
        offset += q_size
        k_parts.append(weight[offset : offset + head_dim])
        offset += head_dim
        v_parts.append(weight[offset : offset + head_dim])
    query = torch.cat(q_parts).view(num_heads, head_dim, hidden_size)
    gate = torch.cat(gate_parts).view(num_heads, head_dim, hidden_size)
    q_proj = torch.stack((query, gate), dim=1).reshape(-1, hidden_size)
    return q_proj, torch.cat(k_parts), torch.cat(v_parts)


def _split_gdn_in_proj(
    weight: torch.Tensor,
    *,
    qk_dim: int,
    value_dim: int,
    num_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """拆分 Megatron linear-attention fused projection。"""

    index = 0
    query = weight[index : index + qk_dim]
    index += qk_dim
    key = weight[index : index + qk_dim]
    index += qk_dim
    value = weight[index : index + value_dim]
    index += value_dim
    z = weight[index : index + value_dim]
    index += value_dim
    beta = weight[index : index + num_value_heads]
    index += num_value_heads
    alpha = weight[index : index + num_value_heads]
    return torch.cat((query, key, value)), z, beta, alpha


def _stack_moe_weights(
    source: dict[str, torch.Tensor],
    *,
    source_prefix: str,
    target_prefix: str,
    output: OrderedDict[str, torch.Tensor],
    num_experts: int,
) -> None:
    """把逐 expert Megatron tensor 组合为 transformers grouped parameters。"""

    gate_up: list[torch.Tensor] = []
    down: list[torch.Tensor] = []
    for expert_index in range(num_experts):
        expert = f"{source_prefix}.experts.local_experts.{expert_index}"
        gate = source[f"{expert}.linear_fc1.weight_w"]
        up = source[f"{expert}.linear_fc1.weight_v"]
        gate_up.append(torch.cat((gate, up), dim=0))
        down.append(source[f"{expert}.linear_fc2.weight"])
    output[f"{target_prefix}.experts.gate_up_proj"] = torch.stack(gate_up).to(
        torch.bfloat16
    )
    output[f"{target_prefix}.experts.down_proj"] = torch.stack(down).to(
        torch.bfloat16
    )
    output[f"{target_prefix}.gate.weight"] = source[
        f"{source_prefix}.router.weight"
    ].to(torch.bfloat16)
    shared = f"{source_prefix}.shared_experts"
    output[f"{target_prefix}.shared_expert.gate_proj.weight"] = source[
        f"{shared}.linear_fc1.weight_w"
    ].to(torch.bfloat16)
    output[f"{target_prefix}.shared_expert.up_proj.weight"] = source[
        f"{shared}.linear_fc1.weight_v"
    ].to(torch.bfloat16)
    output[f"{target_prefix}.shared_expert.down_proj.weight"] = source[
        f"{shared}.linear_fc2.weight"
    ].to(torch.bfloat16)
    output[f"{target_prefix}.shared_expert_gate.weight"] = source[
        f"{shared}.gate_weight"
    ].to(torch.bfloat16)


def _convert_attention_layer(
    source: dict[str, torch.Tensor],
    *,
    source_prefix: str,
    target_prefix: str,
    layer_type: str,
    config: dict[str, Any],
    output: OrderedDict[str, torch.Tensor],
) -> None:
    """转换一个 Qwen3.5 hybrid attention layer。"""

    attention = f"{source_prefix}.self_attention"
    if layer_type == "linear_attention":
        qk_dim = int(config["linear_num_key_heads"]) * int(
            config["linear_key_head_dim"]
        )
        value_dim = int(config["linear_num_value_heads"]) * int(
            config["linear_value_head_dim"]
        )
        qkv, z, beta, alpha = _split_gdn_in_proj(
            source[f"{attention}.in_proj.weight"],
            qk_dim=qk_dim,
            value_dim=value_dim,
            num_value_heads=int(config["linear_num_value_heads"]),
        )
        output[f"{target_prefix}.linear_attn.in_proj_qkv.weight"] = qkv.to(
            torch.bfloat16
        )
        output[f"{target_prefix}.linear_attn.in_proj_z.weight"] = z.to(
            torch.bfloat16
        )
        output[f"{target_prefix}.linear_attn.in_proj_b.weight"] = beta.to(
            torch.bfloat16
        )
        output[f"{target_prefix}.linear_attn.in_proj_a.weight"] = alpha.to(
            torch.bfloat16
        )
        output[f"{target_prefix}.input_layernorm.weight"] = source[
            f"{attention}.in_proj.layer_norm_weight"
        ].to(torch.bfloat16)
        output[f"{target_prefix}.linear_attn.conv1d.weight"] = source[
            f"{attention}.conv1d.weight"
        ].to(torch.bfloat16)
        output[f"{target_prefix}.linear_attn.A_log"] = source[
            f"{attention}.A_log"
        ].to(torch.bfloat16)
        output[f"{target_prefix}.linear_attn.dt_bias"] = source[
            f"{attention}.dt_bias"
        ].to(torch.bfloat16)
        output[f"{target_prefix}.linear_attn.norm.weight"] = (
            source[f"{attention}.out_norm.weight"] + 1.0
        ).to(torch.bfloat16)
        output[f"{target_prefix}.linear_attn.out_proj.weight"] = source[
            f"{attention}.out_proj.weight"
        ].to(torch.bfloat16)
        return

    q_proj, k_proj, v_proj = _split_qkv_with_gate(
        source[f"{attention}.linear_qkv.weight"],
        num_heads=int(config["num_attention_heads"]),
        num_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
    )
    output[f"{target_prefix}.self_attn.q_proj.weight"] = q_proj.to(
        torch.bfloat16
    )
    output[f"{target_prefix}.self_attn.k_proj.weight"] = k_proj.to(
        torch.bfloat16
    )
    output[f"{target_prefix}.self_attn.v_proj.weight"] = v_proj.to(
        torch.bfloat16
    )
    output[f"{target_prefix}.input_layernorm.weight"] = source[
        f"{attention}.linear_qkv.layer_norm_weight"
    ].to(torch.bfloat16)
    output[f"{target_prefix}.self_attn.q_norm.weight"] = source[
        f"{attention}.q_layernorm.weight"
    ].to(torch.bfloat16)
    output[f"{target_prefix}.self_attn.k_norm.weight"] = source[
        f"{attention}.k_layernorm.weight"
    ].to(torch.bfloat16)
    output[f"{target_prefix}.self_attn.o_proj.weight"] = source[
        f"{attention}.linear_proj.weight"
    ].to(torch.bfloat16)


def _convert_mcore_state_to_runtime(
    dcp_state: dict[str, torch.Tensor],
    *,
    config: dict[str, Any],
) -> OrderedDict[str, torch.Tensor]:
    """把完整 Megatron key/布局转换为 O5 runtime HF key/布局。"""

    source = {
        _runtime_key(key): value
        for key, value in dcp_state.items()
    }
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in source.items():
        if key.startswith(("language_model.", "fused_mtp.")):
            continue
        output[key] = value.to(torch.bfloat16)

    output["llm.model.embed_tokens.weight"] = source[
        "language_model.embedding.word_embeddings.weight"
    ].to(torch.bfloat16)
    output["llm.lm_head.weight"] = source[
        "language_model.output_layer.weight"
    ].to(torch.bfloat16)
    output["llm.model.norm.weight"] = source[
        "language_model.decoder.final_layernorm.weight"
    ].to(torch.bfloat16)

    layer_types = list(config["layer_types"])
    num_experts = int(config["num_experts"])
    for layer_index, layer_type in enumerate(layer_types):
        source_layer = f"language_model.decoder.layers.{layer_index}"
        target_layer = f"llm.model.layers.{layer_index}"
        _convert_attention_layer(
            source,
            source_prefix=source_layer,
            target_prefix=target_layer,
            layer_type=layer_type,
            config=config,
            output=output,
        )
        output[f"{target_layer}.post_attention_layernorm.weight"] = source[
            f"{source_layer}.pre_mlp_layernorm.weight"
        ].to(torch.bfloat16)
        _stack_moe_weights(
            source,
            source_prefix=f"{source_layer}.mlp",
            target_prefix=f"{target_layer}.mlp",
            output=output,
            num_experts=num_experts,
        )

    mtp_source = "fused_mtp.layer"
    if f"{mtp_source}.eh_proj.weight" not in source:
        return output
    output["llm.mtp.pre_fc_norm_embedding.weight"] = source[
        f"{mtp_source}.enorm.weight"
    ].to(torch.bfloat16)
    output["llm.mtp.pre_fc_norm_hidden.weight"] = source[
        f"{mtp_source}.hnorm.weight"
    ].to(torch.bfloat16)
    output["llm.mtp.fc.weight"] = source[f"{mtp_source}.eh_proj.weight"].to(
        torch.bfloat16
    )
    output["llm.mtp.norm.weight"] = source[
        f"{mtp_source}.final_layernorm.weight"
    ].to(torch.bfloat16)
    mtp_layer_source = f"{mtp_source}.transformer_layer"
    mtp_layer_target = "llm.mtp.layers.0"
    _convert_attention_layer(
        source,
        source_prefix=mtp_layer_source,
        target_prefix=mtp_layer_target,
        layer_type="full_attention",
        config=config,
        output=output,
    )
    output[f"{mtp_layer_target}.post_attention_layernorm.weight"] = source[
        f"{mtp_layer_source}.pre_mlp_layernorm.weight"
    ].to(torch.bfloat16)
    _stack_moe_weights(
        source,
        source_prefix=f"{mtp_layer_source}.mlp",
        target_prefix=f"{mtp_layer_target}.mlp",
        output=output,
        num_experts=num_experts,
    )
    return output


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
    model_path: str | Path,
    output_pt: str | Path,
) -> DensePtManifest:
    """把一个 model-only DCP 转成完整 PT。

    参数:
        source_dir: 含 ``.metadata`` 和 shard 文件的 model-only DCP。
        model_path: O5 config/tokenizer/processor 资产目录。
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

    del key_map
    config = json.loads(
        (Path(model_path) / "config.json").read_text(encoding="utf-8")
    )
    runtime_state = _convert_mcore_state_to_runtime(
        dcp_state,
        config=config,
    )
    del dcp_state
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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-pt", required=True)
    args = parser.parse_args()

    manifest = convert_model_only_dcp_to_pt(
        source_dir=args.source_dir,
        model_path=args.model_path,
        output_pt=args.output_pt,
    )
    print(json.dumps(manifest.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
