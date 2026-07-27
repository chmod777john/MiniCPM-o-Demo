"""O5 LLM-only 部署 PT 的冻结 TTS 合并测试。"""

from pathlib import Path

import torch

from tools.checkpoint.merge_frozen_tts import merge_frozen_tts


def test_merge_frozen_tts_preserves_model_and_inserts_base_tts(
    tmp_path: Path,
) -> None:
    """只复制 tts.*，并保持目标模型权重与 MiniCPMO key 顺序。"""

    model_pt = tmp_path / "model.pt"
    base_pt = tmp_path / "base.pt"
    output_pt = tmp_path / "merged.pt"
    torch.save(
        {
            "llm.model.embed_tokens.weight": torch.ones(4, 2),
            "llm.lm_head.weight": torch.full((4, 2), 2.0),
            "apm.weight": torch.full((2, 2), 3.0),
            "llm.model.rotary_emb.inv_freq": torch.full((2,), 4.0),
        },
        model_pt,
    )
    torch.save(
        {
            "llm.model.embed_tokens.weight": torch.zeros(4, 2),
            "tts.model.weight": torch.full((2, 2), 5.0),
            "tts.head.weight": torch.full((2, 2), 6.0),
        },
        base_pt,
    )

    manifest = merge_frozen_tts(
        model_pt=model_pt,
        base_pt=base_pt,
        output_pt=output_pt,
    )
    merged = torch.load(output_pt, weights_only=True)

    assert manifest.model_key_count == 4
    assert manifest.base_tts_key_count == 2
    assert manifest.output_key_count == 6
    assert torch.equal(
        merged["llm.model.embed_tokens.weight"],
        torch.ones(4, 2),
    )
    assert torch.equal(
        merged["tts.model.weight"],
        torch.full((2, 2), 5.0),
    )
    assert list(merged) == [
        "llm.model.embed_tokens.weight",
        "llm.lm_head.weight",
        "apm.weight",
        "tts.model.weight",
        "tts.head.weight",
        "llm.model.rotary_emb.inv_freq",
    ]
