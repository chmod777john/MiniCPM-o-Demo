"""为 O5 LLM-only checkpoint 补回统一训练起点中的冻结 TTS 权重。

LLM-only Megatron Job 构建模型时未启用 TTS，因此 model-only DCP 不包含 ``tts.*``。
部署 FC Duplex 时 MiniCPMO 仍需完整 TTS 模块；本工具只从同一训练起点的 canonical PT
复制冻结的 ``tts.*``，其余 tensor 保持目标 LLM-only checkpoint 原值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FrozenTtsMergeManifest:
    """冻结 TTS 合并产物的审计记录。"""

    model_pt: str
    base_pt: str
    output_pt: str
    model_key_count: int
    base_tts_key_count: int
    output_key_count: int
    embedding_rows: int
    lm_head_rows: int
    output_sha256: str


def merge_frozen_tts(
    *,
    model_pt: str | Path,
    base_pt: str | Path,
    output_pt: str | Path,
) -> FrozenTtsMergeManifest:
    """将 clean-base 冻结 TTS 合并到 LLM-only PT。

    参数:
        model_pt: 仅缺少 ``tts.*`` 的 canonical LLM-only PT。
        base_pt: 与训练血缘一致、包含冻结 TTS 的 canonical clean-base PT。
        output_pt: 不允许预先存在的完整部署 PT。

    返回:
        包含 key 数量、词表行数和 SHA256 的审计 Manifest。
    """

    model_path = Path(model_pt)
    base_path = Path(base_pt)
    output_path = Path(output_pt)
    if output_path.exists():
        raise FileExistsError(f"目标 PT 已存在，拒绝覆盖: {output_path}")

    model_state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    base_state = torch.load(
        base_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    existing_tts = [key for key in model_state if key.startswith("tts.")]
    if existing_tts:
        raise ValueError(
            f"目标 checkpoint 已包含 {len(existing_tts)} 个 tts.* tensor，拒绝混合"
        )
    base_tts_keys = [key for key in base_state if key.startswith("tts.")]
    if not base_tts_keys:
        raise ValueError("clean-base PT 不包含 tts.* tensor")

    output_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    inserted = False
    for key, value in model_state.items():
        if not inserted and key.startswith("llm.model.rotary_emb."):
            for tts_key in base_tts_keys:
                output_state[tts_key] = base_state[tts_key]
            inserted = True
        output_state[key] = value
    if not inserted:
        for tts_key in base_tts_keys:
            output_state[tts_key] = base_state[tts_key]

    embed_rows = int(
        output_state["llm.model.embed_tokens.weight"].shape[0]
    )
    lm_head_rows = int(output_state["llm.lm_head.weight"].shape[0])
    if embed_rows != lm_head_rows:
        raise ValueError(
            f"embedding/lm_head rows 不一致: {embed_rows} != {lm_head_rows}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(
        output_path.suffix + f".tmp.{os.getpid()}"
    )
    try:
        torch.save(output_state, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = FrozenTtsMergeManifest(
        model_pt=str(model_path.resolve()),
        base_pt=str(base_path.resolve()),
        output_pt=str(output_path.resolve()),
        model_key_count=len(model_state),
        base_tts_key_count=len(base_tts_keys),
        output_key_count=len(output_state),
        embedding_rows=embed_rows,
        lm_head_rows=lm_head_rows,
        output_sha256=_sha256_file(output_path),
    )
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sha256_file(path: Path) -> str:
    """流式计算大 PT 的 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description="Merge frozen clean-base TTS into an O5 LLM-only PT"
    )
    parser.add_argument("--model-pt", required=True)
    parser.add_argument("--base-pt", required=True)
    parser.add_argument("--output-pt", required=True)
    args = parser.parse_args()
    manifest = merge_frozen_tts(
        model_pt=args.model_pt,
        base_pt=args.base_pt,
        output_pt=args.output_pt,
    )
    print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
