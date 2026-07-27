"""从完整 O5 PT 抽取与 checkpoint 配套的 TP2 HF LLM backbone。

该工具使用仓库内 ``modeling.o5`` 代码构建 meta model，把完整 PT 中已训练的
248168-row LLM 权重加载后保存为 sharded safetensors。输出只能与同一个 PT 配套部署。
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from tools.checkpoint.validate_fc_checkpoint import load_checkpoint_state_dict


def build_o5_text_config(
    raw_config: dict[str, Any],
    *,
    expected_rows: int,
) -> Qwen3_5MoeTextConfig:
    """从完整 MiniCPMO config 提取纯 LLM 配置。

    参数:
        raw_config: ``MODEL_PATH/config.json`` 的完整多模态配置。
        expected_rows: SDK 0.0.5 O5 required rows。

    返回:
        不含 ``auto_map``、vision/audio/TTS 或嵌套 ``text_config`` 的 text-only config。
    """

    allowed_fields = set(
        inspect.signature(Qwen3_5MoeTextConfig.__init__).parameters
    ) - {"self"}
    fields = {
        key: value
        for key, value in raw_config.items()
        if key in allowed_fields
    }
    fields.update(
        {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "vocab_size": expected_rows,
        }
    )
    config = Qwen3_5MoeTextConfig(**fields)
    config._attn_implementation = "sdpa"
    config._attn_implementation_internal = "sdpa"
    return config


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

    raw_config = json.loads(
        (Path(model_path) / "config.json").read_text(encoding="utf-8")
    )
    config = build_o5_text_config(
        raw_config,
        expected_rows=expected_rows,
    )
    config._name_or_path = str(model_path)
    config.name_or_path = str(model_path)
    with init_empty_weights():
        llm = Qwen3_5MoeForCausalLM(config)
    llm_state = {
        key.removeprefix("llm."): value
        for key, value in state.items()
        if key.startswith("llm.")
    }
    load_info = llm.load_state_dict(llm_state, strict=False, assign=True)
    del state

    llm = llm.bfloat16()
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
