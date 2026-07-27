"""从完整 O5 PT 抽取与 checkpoint 配套的 TP2 HF LLM backbone。

该工具使用仓库内 ``modeling.o5`` 代码构建 meta model，把完整 PT 中已训练的
248168-row LLM 权重加载后保存为 sharded safetensors。输出只能与同一个 PT 配套部署。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights

from modeling.o5.configuration_minicpmo import MiniCPMOConfig
from modeling.o5.modeling_minicpmo_unified import MiniCPMO
from tools.checkpoint.validate_fc_checkpoint import load_checkpoint_state_dict


def extract_o5_tp2_backbone(
    *,
    model_path: str | Path,
    pt_path: str | Path,
    output_dir: str | Path,
    expected_rows: int = 248168,
) -> Path:
    """抽取 O5 LLM 并保存为 HF sharded safetensors。

    参数:
        model_path: O5 config/tokenizer/processor 资产目录。
        pt_path: 正式 SDK 0.0.5 完整 O5 checkpoint。
        output_dir: 必须不存在或为空的输出目录。
        expected_rows: SDK 0.0.5 O5 required model rows。

    返回:
        已写出 ``model.safetensors.index.json`` 的输出目录。
    """

    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"backbone 输出目录非空，拒绝覆盖: {target}")
    target.mkdir(parents=True, exist_ok=True)

    state = load_checkpoint_state_dict(pt_path)
    embed = _require_state_tensor(
        state,
        "llm.model.embed_tokens.weight",
    )
    lm_head = _require_state_tensor(state, "llm.lm_head.weight")
    embedding_rows = int(embed.shape[0])
    lm_head_rows = int(lm_head.shape[0])
    if embedding_rows != expected_rows or lm_head_rows != expected_rows:
        raise ValueError(
            "O5 PT vocab rows 不符合正式 SDK: "
            f"expected={expected_rows}, embedding={embedding_rows}, "
            f"lm_head={lm_head_rows}"
        )

    config = MiniCPMOConfig.from_pretrained(str(model_path))
    config.vocab_size = expected_rows
    config._name_or_path = str(model_path)
    config.name_or_path = str(model_path)
    with init_empty_weights():
        model = MiniCPMO(config)
    load_info = model.load_state_dict(state, strict=False, assign=True)
    del state

    llm = model.llm.bfloat16()
    llm.config.vocab_size = expected_rows
    llm.save_pretrained(
        target,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    index_path = target / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"backbone 未生成 safetensors index: {index_path}")
    manifest = {
        "schema_version": 1,
        "source_pt": str(Path(pt_path).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "vocab_size": expected_rows,
        "missing_key_count": len(load_info.missing_keys),
        "unexpected_key_count": len(load_info.unexpected_keys),
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
    }
    (target / "fc_backbone_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _require_state_tensor(state: dict[str, Any], key: str) -> Any:
    """读取 checkpoint 必需 tensor。

    参数:
        state: 完整 PT state_dict。
        key: 必需 tensor key。

    返回:
        对应 tensor。
    """

    value = state.get(key)
    if value is None or not hasattr(value, "shape"):
        raise KeyError(f"O5 PT 缺少 tensor: {key}")
    return value


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description="Extract an SDK 0.0.5 O5 TP2 backbone"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=248168)
    args = parser.parse_args()

    output = extract_o5_tp2_backbone(
        model_path=args.model_path,
        pt_path=args.pt_path,
        output_dir=args.output_dir,
        expected_rows=args.expected_rows,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
