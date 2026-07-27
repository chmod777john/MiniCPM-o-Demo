#!/usr/bin/env python3
import json
import os
import io
from pathlib import Path

import soundfile as sf
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig

from modeling.o5.modeling_minicpmo import MiniCPMO
from modeling.o5.processing_minicpmo import MiniCPMOProcessor


MODEL_PATH = os.environ.get("MODEL_PATH", "/user/weihongliang/MiniCPM-o-4_6")
PT_PATH = os.environ.get("PT_PATH", "/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_vendored_chat_probe_iter1200"))
PROMPT = os.environ.get("PROMPT", "请尽量详细介绍自己")
REF_WAV = os.environ.get("REF_WAV", "assets/ref_audio/ref_minicpm_signature.wav")
UI_STYLE = os.environ.get("UI_STYLE", "0") == "1"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))


def patch_token2wav_bytesio_save():
    import stepaudio2.token2wav as token2wav_mod

    original_save = token2wav_mod.torchaudio.save
    if getattr(original_save, "_minicpmo_bytesio_patch", False):
        return

    def save(uri, src, sample_rate, *args, **kwargs):
        if isinstance(uri, io.BytesIO):
            audio = src.detach().cpu().float().numpy()
            if audio.ndim == 2 and audio.shape[0] <= 8:
                audio = audio.T
            sf.write(uri, audio, sample_rate, format="WAV")
            return None
        return original_save(uri, src, sample_rate, *args, **kwargs)

    save._minicpmo_bytesio_patch = True
    token2wav_mod.torchaudio.save = save


def load_pt_state_dict(path):
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            return state_dict[key]
    return state_dict


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = OUT_DIR / "vendored_chat_output.wav"

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    config._name_or_path = MODEL_PATH
    config.name_or_path = MODEL_PATH

    with init_empty_weights():
        model = MiniCPMO(config)

    state_dict = load_pt_state_dict(PT_PATH)
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    print("load_state_dict", "missing", len(info.missing_keys), "unexpected", len(info.unexpected_keys))
    del state_dict

    model.bfloat16().eval().cuda()
    model.processor = MiniCPMOProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.init_tts()
    patch_token2wav_bytesio_save()

    print("prompt", PROMPT, "ui_style", UI_STYLE, "max_new_tokens", MAX_NEW_TOKENS)
    if UI_STYLE:
        ref_audio, _ = sf.read(REF_WAV, dtype="float32")
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)
        msgs = [
            {"role": "system", "content": ["模仿音频样本的音色并生成新的内容。", ref_audio]},
            {"role": "user", "content": PROMPT},
        ]
    else:
        msgs = [{"role": "user", "content": PROMPT}]
    result = model.chat(
        msgs=msgs,
        use_tts_template=True,
        generate_audio=True,
        output_audio_path=str(out_wav),
        max_new_tokens=MAX_NEW_TOKENS,
        enable_thinking=False,
    )
    print("text", result)
    print("wav_path", out_wav)

    wav, sr = sf.read(out_wav, dtype="float32")
    summary = {
        "text": result,
        "wav_path": str(out_wav),
        "samples": len(wav),
        "sr": sr,
        "duration": len(wav) / sr if sr else -1,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT_DIR / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
