#!/usr/bin/env python3
import asyncio
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig

from modeling.o5.modeling_minicpmo_unified import MiniCPMO
from modeling.o5.processing_minicpmo import MiniCPMOProcessor


MODEL_PATH = os.environ.get("MODEL_PATH")
PT_PATH = os.environ.get("PT_PATH")
REF_WAV = os.environ.get("REF_WAV", "assets/ref_audio/ref_minicpm_signature.wav")
OUT_DIR = Path(os.environ.get("OUT_DIR", "o5_unified_model_duplex_probe"))
USER_TEXT = os.environ.get("USER_TEXT", "请详细介绍西安的历史文化、旅游景点、美食和城市特色。")
USER_WAV = os.environ.get("USER_WAV")
EDGE_TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TRAILING_SILENCE_SECONDS = 6
ENABLE_COMPILE = os.environ.get("ENABLE_COMPILE", "0") == "1"
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
COMPILE_WARMUP = os.environ.get("COMPILE_WARMUP", "1") == "1"
COMPILE_SKIP = [x for x in os.environ.get("COMPILE_SKIP", "").split(",") if x]
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
RUNTIME_INFO = {"attn_implementation": ATTN_IMPLEMENTATION}


def write_runtime_info():
    with open(OUT_DIR / "runtime_info.json", "w", encoding="utf-8") as f:
        json.dump(RUNTIME_INFO, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_16k(path):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32)


async def write_user_input_tts(path):
    import edge_tts

    await edge_tts.Communicate(USER_TEXT, EDGE_TTS_VOICE).save(str(path))


def chunk_audio(audio, first_samples, step_samples):
    if len(audio) < first_samples:
        return []
    chunks = [audio[:first_samples]]
    for start in range(first_samples, len(audio) - step_samples + 1, step_samples):
        chunks.append(audio[start : start + step_samples])
    return chunks


def load_pt_state_dict(path):
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            return state_dict[key]
    return state_dict


def load_o5_model():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config._attn_implementation = ATTN_IMPLEMENTATION
    config._name_or_path = MODEL_PATH
    config.name_or_path = MODEL_PATH
    print("attn_implementation", config._attn_implementation, flush=True)

    try:
        from transformers.utils import is_flash_attn_2_available
        from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen_moe

        RUNTIME_INFO["flash_attn_2_available"] = bool(is_flash_attn_2_available())
        RUNTIME_INFO["flash_linear_attention_available"] = bool(is_flash_linear_attention_available())
        RUNTIME_INFO["causal_conv1d_available"] = bool(is_causal_conv1d_available())
        RUNTIME_INFO["qwen_moe_fast_path"] = bool(getattr(qwen_moe, "is_fast_path_available", False))
        print("flash_attn_2_available", RUNTIME_INFO["flash_attn_2_available"], flush=True)
        print("flash_linear_attention_available", RUNTIME_INFO["flash_linear_attention_available"], flush=True)
        print("causal_conv1d_available", RUNTIME_INFO["causal_conv1d_available"], flush=True)
        print("qwen_moe_fast_path", RUNTIME_INFO["qwen_moe_fast_path"], flush=True)
    except Exception as exc:
        RUNTIME_INFO["fast_path_probe_error"] = f"{type(exc).__name__}: {exc}"
        print("fast_path_probe_error", type(exc).__name__, str(exc), flush=True)
    write_runtime_info()

    with init_empty_weights():
        model = MiniCPMO(config)

    state_dict = load_pt_state_dict(PT_PATH)
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    print("load_state_dict", "missing", len(info.missing_keys), "unexpected", len(info.unexpected_keys))
    if info.missing_keys:
        print("missing_head", info.missing_keys[:5])
    if info.unexpected_keys:
        print("unexpected_head", info.unexpected_keys[:5])
    del state_dict

    model.bfloat16().eval().cuda()
    model.processor = MiniCPMOProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model


def main():
    if not MODEL_PATH or not PT_PATH:
        raise SystemExit("MODEL_PATH and PT_PATH must be set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("user_input_*", "duplex_chunk_*.wav", "duplex_output_*.wav"):
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    for path in (OUT_DIR / "runtime_info.json", OUT_DIR / "timings.jsonl"):
        if path.exists():
            path.unlink()
    torch.manual_seed(1)

    if USER_WAV:
        user_audio = load_16k(USER_WAV)
    else:
        user_mp3 = OUT_DIR / "user_input_edge_tts.mp3"
        asyncio.run(write_user_input_tts(user_mp3))
        user_audio = load_16k(user_mp3)
    sf.write(OUT_DIR / "user_input_16k.wav", user_audio, 16000)

    model = load_o5_model()
    model.init_unified(
        preload_both_tts=False,
        duplex_config={
            "generate_audio": True,
            "ls_mode": "explicit",
            "max_new_speak_tokens_per_chunk": 20,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "force_listen_count": 3,
        },
        device="cuda",
        chat_vocoder="token2wav",
    )

    if ENABLE_COMPILE:
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        print("torch_compile", "enabled", "mode", COMPILE_MODE, "cache", cache_dir, "skip", COMPILE_SKIP)
        compile_start = time.perf_counter()
        model.apply_torch_compile(mode=COMPILE_MODE, dynamic=True, skip_modules=COMPILE_SKIP or None)
        if COMPILE_WARMUP:
            model.warmup_compile(ref_audio_path=REF_WAV, max_warmup_chunks=10, total_estimate_seconds=1000)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("torch_compile_ready_s", f"{time.perf_counter() - compile_start:.4f}")

    ref_audio = load_16k(REF_WAV)
    model.duplex_prepare(
        prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
        suffix_system_prompt="<|audio_end|><|im_end|>",
        ref_audio=ref_audio,
        prompt_wav_path=REF_WAV,
    )

    first = int(model.duplex.FIRST_CHUNK_MS * model.duplex.SAMPLE_RATE / 1000)
    step = int(model.duplex.CHUNK_MS * model.duplex.SAMPLE_RATE / 1000)
    chunks = chunk_audio(user_audio, first, step)
    chunks.extend(np.zeros(step, dtype=np.float32) for _ in range(TRAILING_SILENCE_SECONDS))
    print("input_seconds", len(user_audio) / 16000, "chunks", len(chunks))

    timeline = []
    speech_only = []
    timings = []
    for i, chunk in enumerate(chunks[:10]):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        prefill = model.duplex_prefill(audio_waveform=chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - prefill_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generate_start = time.perf_counter()
        result = model.duplex_generate(
            decode_mode="sampling",
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            listen_prob_scale=1.0,
            text_repetition_penalty=1.05,
            text_repetition_window_size=512,
            length_penalty=1.1,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generate_s = time.perf_counter() - generate_start

        text = result.get("text", "")
        n_tts_tokens = result.get("n_tts_tokens")
        round_timing = {
            "round": i,
            "prefill_s": prefill_s,
            "generate_s": generate_s,
            "total_s": prefill_s + generate_s,
            "is_listen": result.get("is_listen"),
            "n_tts_tokens": n_tts_tokens,
            "text": text,
            "prefill": prefill,
            "profile_events": result.get("profile_events"),
            "compile_enabled": ENABLE_COMPILE,
            "compile_mode": COMPILE_MODE if ENABLE_COMPILE else None,
            "compile_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR") if ENABLE_COMPILE else None,
        }
        timings.append(round_timing)
        print(
            "round",
            i,
            "prefill_s",
            f"{prefill_s:.4f}",
            "generate_s",
            f"{generate_s:.4f}",
            "total_s",
            f"{prefill_s + generate_s:.4f}",
            "listen",
            result.get("is_listen"),
            "tts_tokens",
            n_tts_tokens,
            "text",
            repr(text),
            "prefill",
            prefill,
        )
        wav = result.get("audio_waveform")
        if wav is not None and len(wav) > 0:
            wav = np.asarray(wav, dtype=np.float32)
            sf.write(OUT_DIR / f"duplex_chunk_{i:03d}.wav", wav, 24000)
            timeline.append(wav)
            if not result.get("is_listen"):
                speech_only.append(wav)
        model.duplex_finalize()

    sf.write(
        OUT_DIR / "duplex_output_timeline_24k.wav",
        np.concatenate(timeline) if timeline else np.zeros(0, dtype=np.float32),
        24000,
    )
    sf.write(
        OUT_DIR / "duplex_output_speech_only_24k.wav",
        np.concatenate(speech_only) if speech_only else np.zeros(0, dtype=np.float32),
        24000,
    )
    with open(OUT_DIR / "timings.jsonl", "w", encoding="utf-8") as f:
        for item in timings:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
