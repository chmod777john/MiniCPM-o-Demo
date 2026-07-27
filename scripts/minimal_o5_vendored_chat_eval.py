#!/usr/bin/env python3
"""Minimal chat speech probe for local o5 vendored inference code.

This intentionally avoids the full humanevalkit benchmark runner.  It imports the
local modeling.o5 vendored package, runs a small list of QA-style chat prompts, and
writes one result.json + assistant.wav per sample plus a compact index.
"""

from __future__ import annotations

import html
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig


DEFAULT_PROMPTS = [
    "请详细介绍一下北京。",
]


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


VENDORED_ROOT = env_path(
    "VENDORED_ROOT",
    "/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30",
)
MODEL_PATH = env_path("MODEL_PATH", "/user/weihongliang/MiniCPM-o-4_6")
PT_PATH = env_path("PT_PATH", "/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt")
OUT_DIR = env_path("OUT_DIR", "/user/weihongliang/o5_vendored_chat_eval")
REF_WAV = os.environ.get("REF_WAV", "assets/ref_audio/ref_minicpm_signature.wav")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
UI_STYLE = os.environ.get("UI_STYLE", "1") != "0"
DTYPE = os.environ.get("DTYPE", "bfloat16")


def add_vendored_to_path() -> None:
    root = str(VENDORED_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def patch_token2wav_bytesio_save() -> None:
    import stepaudio2.token2wav as token2wav_mod

    original_save = token2wav_mod.torchaudio.save
    if getattr(original_save, "_minicpmo_bytesio_patch", False):
        return

    def save(uri: Any, src: torch.Tensor, sample_rate: int, *args: Any, **kwargs: Any) -> None:
        if isinstance(uri, io.BytesIO):
            audio = src.detach().cpu().float().numpy()
            if audio.ndim == 2 and audio.shape[0] <= 8:
                audio = audio.T
            sf.write(uri, audio, sample_rate, format="WAV")
            return None
        return original_save(uri, src, sample_rate, *args, **kwargs)

    save._minicpmo_bytesio_patch = True  # type: ignore[attr-defined]
    token2wav_mod.torchaudio.save = save


def load_pt_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            return state_dict[key]
    return state_dict


def load_prompts() -> list[str]:
    prompts_json = os.environ.get("PROMPTS_JSON")
    prompts_file = os.environ.get("PROMPTS_FILE")
    if prompts_json:
        data = json.loads(prompts_json)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("PROMPTS_JSON must be a JSON list of strings")
        return data
    if prompts_file:
        return [line.strip() for line in Path(prompts_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return DEFAULT_PROMPTS


def load_model():
    add_vendored_to_path()
    from modeling.o5.modeling_minicpmo import MiniCPMO
    from modeling.o5.processing_minicpmo import MiniCPMOProcessor

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    config._name_or_path = str(MODEL_PATH)
    config.name_or_path = str(MODEL_PATH)

    with init_empty_weights():
        model = MiniCPMO(config)

    state_dict = load_pt_state_dict(PT_PATH)
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        "load_state_dict",
        "missing",
        len(info.missing_keys),
        "unexpected",
        len(info.unexpected_keys),
        flush=True,
    )
    if info.missing_keys:
        print("missing_head", info.missing_keys[:10], flush=True)
    if info.unexpected_keys:
        print("unexpected_head", info.unexpected_keys[:10], flush=True)
    del state_dict

    if DTYPE == "float16":
        model.float16()
    elif DTYPE == "float32":
        model.float()
    else:
        model.bfloat16()
    model.eval().cuda()
    model.processor = MiniCPMOProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.init_tts()
    patch_token2wav_bytesio_save()
    return model


def build_msgs(prompt: str):
    if not UI_STYLE:
        return [{"role": "user", "content": prompt}]
    ref_path = Path(REF_WAV)
    if not ref_path.is_absolute():
        ref_path = VENDORED_ROOT / ref_path
    ref_audio, _ = sf.read(ref_path, dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    return [
        {"role": "system", "content": ["模仿音频样本的音色并生成新的内容。", ref_audio]},
        {"role": "user", "content": prompt},
    ]


def audio_summary(path: Path) -> dict[str, Any]:
    wav, sr = sf.read(path, dtype="float32")
    return {"sample_rate": sr, "samples": len(wav), "duration_sec": len(wav) / sr if sr else None}


def write_index(records: list[dict[str, Any]]) -> None:
    lines = ["# o5 vendored chat eval", "", f"output: `{OUT_DIR}`", ""]
    html_rows = []
    for item in records:
        lines.append(f"## {item['sample_id']}")
        lines.append("")
        lines.append(f"prompt: {item['prompt']}")
        lines.append("")
        lines.append(f"text: {item['text']}")
        lines.append("")
        lines.append(f"audio: `{item['audio_path']}`")
        lines.append("")
        html_rows.append(
            "<section>"
            f"<h2>{html.escape(item['sample_id'])}</h2>"
            f"<p><b>Prompt:</b> {html.escape(item['prompt'])}</p>"
            f"<p><b>Text:</b> {html.escape(item['text'])}</p>"
            f"<p><b>Duration:</b> {item['duration_sec']:.2f}s</p>"
            f"<audio controls src=\"{html.escape(Path(item['audio_path']).name)}\"></audio>"
            f"<p><code>{html.escape(item['audio_path'])}</code></p>"
            "</section>"
        )
    (OUT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>o5 vendored chat eval</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:32px auto;line-height:1.5}"
        "section{border-top:1px solid #ddd;padding:20px 0}audio{width:100%}code{word-break:break-all}</style>"
        + "\n".join(html_rows),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    print("vendored_root", VENDORED_ROOT, flush=True)
    print("model_path", MODEL_PATH, flush=True)
    print("pt_path", PT_PATH, flush=True)
    print("out_dir", OUT_DIR, flush=True)
    print("num_prompts", len(prompts), "max_new_tokens", MAX_NEW_TOKENS, "ui_style", UI_STYLE, flush=True)

    model = load_model()
    records: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        sample_id = f"chat_{idx:04d}"
        sample_dir = OUT_DIR / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        wav_path = sample_dir / "assistant.wav"
        print(f"[{sample_id}] prompt={prompt}", flush=True)
        text = model.chat(
            msgs=build_msgs(prompt),
            use_tts_template=True,
            generate_audio=True,
            output_audio_path=str(wav_path),
            max_new_tokens=MAX_NEW_TOKENS,
            enable_thinking=False,
        )
        summary = audio_summary(wav_path)
        record = {
            "sample_id": sample_id,
            "request": {
                "model_family": "minicpmo",
                "inference_mode": "chat",
                "response_mode": "text_and_speech",
                "model_path": str(MODEL_PATH),
                "checkpoint_path": str(PT_PATH),
                "generate_audio": True,
                "use_tts_template": True,
                "max_new_tokens": MAX_NEW_TOKENS,
                "prompt": prompt,
                "ui_style": UI_STYLE,
                "reference_audio_path": str((VENDORED_ROOT / REF_WAV).resolve()) if UI_STYLE and not Path(REF_WAV).is_absolute() else REF_WAV,
            },
            "output": {
                "output_schema": "speech",
                "error": None,
                "response": {
                    "role": "assistant",
                    "text": text,
                    "audio_path": str(wav_path),
                },
                **summary,
            },
        }
        (sample_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        flat = {
            "sample_id": sample_id,
            "prompt": prompt,
            "text": text,
            "audio_path": str(wav_path),
            **summary,
            "result_path": str(sample_dir / "result.json"),
        }
        records.append(flat)
        print(json.dumps(flat, ensure_ascii=False), flush=True)

    merged = {
        "benchmark_name": "minimal_o5_vendored_chat_eval",
        "processed_sample_count": len(records),
        "error_sample_count": 0,
        "samples": records,
    }
    (OUT_DIR / "merged_report.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(records)
    print("wrote", OUT_DIR / "index.html", flush=True)
    print("wrote", OUT_DIR / "merged_report.json", flush=True)


if __name__ == "__main__":
    main()
