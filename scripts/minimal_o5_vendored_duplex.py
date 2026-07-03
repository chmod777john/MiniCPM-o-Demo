#!/usr/bin/env python3
import asyncio
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig

from MiniCPMO45.modeling_minicpmo import MiniCPMO, MiniCPMODuplex
from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor


MODEL_PATH = os.environ.get("MODEL_PATH", "/user/weihongliang/MiniCPM-o-4_6")
PT_PATH = os.environ.get("PT_PATH", "/user/weihongliang/o5_weights/omni_sft_main_run_iter7000.pt")
REF_WAV = os.environ.get("REF_WAV", "assets/ref_audio/ref_minicpm_signature.wav")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_vendored_duplex_probe"))
USER_TEXT = os.environ.get("USER_TEXT", "请详细介绍西安的历史文化、旅游景点、美食和城市特色。")
USER_WAV = os.environ.get("USER_WAV")
EDGE_TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TRAILING_SILENCE_SECONDS = 6


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
    config._attn_implementation = "sdpa"
    config._name_or_path = MODEL_PATH
    config.name_or_path = MODEL_PATH

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("user_input_*", "duplex_chunk_*.wav", "duplex_output_*.wav"):
        for path in OUT_DIR.glob(pattern):
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
    ref_audio = load_16k(REF_WAV)

    duplex = MiniCPMODuplex.from_existing_model(
        model,
        force_listen_count=3,
        top_k=20,
        n_timesteps=5,
    )
    duplex.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=ref_audio,
        prompt_wav_path=REF_WAV,
    )

    first = int(duplex.FIRST_CHUNK_MS * duplex.SAMPLE_RATE / 1000)
    step = int(duplex.CHUNK_MS * duplex.SAMPLE_RATE / 1000)
    chunks = chunk_audio(user_audio, first, step)
    chunks.extend(np.zeros(step, dtype=np.float32) for _ in range(TRAILING_SILENCE_SECONDS))
    print("input_seconds", len(user_audio) / 16000, "chunks", len(chunks))

    timeline = []
    speech_only = []
    for i, chunk in enumerate(chunks[:10]):
        prefill = duplex.streaming_prefill(audio_waveform=chunk)
        result = duplex.streaming_generate()
        text = result.get("text", "")
        print(i, prefill, result.get("is_listen"), repr(text), result.get("n_tts_tokens"))
        wav = result.get("audio_waveform")
        if wav is not None and len(wav) > 0:
            wav = np.asarray(wav, dtype=np.float32)
            sf.write(OUT_DIR / f"duplex_chunk_{i:03d}.wav", wav, 24000)
            timeline.append(wav)
            if not result.get("is_listen"):
                speech_only.append(wav)

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
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
