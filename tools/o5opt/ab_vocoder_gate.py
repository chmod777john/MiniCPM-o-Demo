#!/usr/bin/env python3
"""AUDIO-EQUIVALENCE GATE for the vocoder bucketing: run a real omni conversation with the vocoder in
DIFF mode (real output = eager, unchanged; bucketed mel computed on the SAME inputs for every FLUSH
chunk and compared). If mel_max_abs_diff ~0 across all flush chunks, the pad+mask preserves real-frame
output -> the bucketing is safe to enable. Also reports the flush chunk sizes seen."""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa"); os.environ.setdefault("O5_VISION_BATCH", "1")
OUT_DIR = Path(os.environ["OUT_DIR"]); OUT_DIR.mkdir(parents=True, exist_ok=True); os.environ["OUT_DIR"] = str(OUT_DIR)
import minimal_o5_unified_model_duplex as probe
from modeling.o5.opt_flags import OPT
import vocoder_graph as vg
VIDEO = os.environ.get("VIDEO", os.path.join(WORKTREE, "assets", "samples", "compile.mp4"))
REF_WAV = os.environ.get("REF_WAV", os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav"))
N = int(os.environ.get("N_UNITS", "40"))
DUP = {"generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
       "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3}
GEN = dict(temperature=0.7, top_k=20, top_p=0.8, listen_prob_scale=1.0,
           text_repetition_penalty=1.05, text_repetition_window_size=512, length_penalty=1.1)
def log(*a): print(*a, flush=True)
def extract_mp4(video, n, sr=16000):
    import subprocess, tempfile
    from PIL import Image
    import librosa
    tmp = tempfile.mkdtemp(prefix="vgate_")
    wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg","-i",video,"-ar",str(sr),"-ac","1","-t",str(n),"-f","wav","-y",wav], capture_output=True, check=True)
    audio,_ = librosa.load(wav, sr=sr, mono=True)
    cs=sr; na=min(n, len(audio)//cs)
    ac=[audio[i*cs:(i+1)*cs].astype(np.float32) for i in range(na)]
    fdir=os.path.join(tmp,"f"); os.makedirs(fdir, exist_ok=True)
    subprocess.run(["ffmpeg","-i",video,"-vf","fps=1","-t",str(n),os.path.join(fdir,"f%03d.jpg"),"-y"], capture_output=True, check=True)
    fr=[Image.open(os.path.join(fdir,f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    return ac, fr
def force_tts_eager(model):
    for mod in model.tts.model.modules():
        c=getattr(mod,"config",None)
        if c is not None and hasattr(c,"_attn_implementation"): c._attn_implementation="eager"
def set_experts_impl(m, impl):
    seen=set()
    for mod in m.modules():
        c=getattr(mod,"config",None)
        if c is not None and hasattr(c,"_experts_implementation") and id(c) not in seen:
            c._experts_implementation=impl; seen.add(id(c))
def prep(m, ref): m.duplex_prepare(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                                   suffix_system_prompt="<|audio_end|><|im_end|>", ref_audio=ref, prompt_wav_path=REF_WAV)
def main():
    model=probe.load_o5_model()
    model.init_unified(preload_both_tts=True, duplex_config=DUP, device="cuda", chat_vocoder="token2wav")
    ref=probe.load_16k(REF_WAV)
    for k in OPT: OPT[k]=False
    OPT["tts_fast"]=True; OPT["lmhead"]=True; OPT["tts_graph"]=True; OPT["fuse_vision_audio"]=True; OPT["llm_graph"]=True
    force_tts_eager(model); set_experts_impl(model,"batched_mm")
    ac,fr=extract_mp4(VIDEO, 20)
    # warmup (capture tts/llm graphs)
    prep(model,ref)
    for i in range(min(4,N)):
        model.duplex_prefill(audio_waveform=ac[i], frame_list=[fr[i]]); model.duplex_generate(decode_mode="sampling", **GEN); model.duplex_finalize()
    # install the DIFF gate (bucket=50) — real output stays eager; bucketed compared per flush chunk
    diffs = vg.install_bucket_diff(model, log, bucket=50)
    # long omni run to hit flush chunks (utterance boundaries)
    aloop=[ac[i%len(ac)] for i in range(N)]; floop=[fr[i%len(fr)] for i in range(N)]
    prep(model,ref)
    for i in range(N):
        model.duplex_prefill(audio_waveform=aloop[i], frame_list=[floop[i]]); model.duplex_generate(decode_mode="sampling", **GEN); model.duplex_finalize()
    valid=[d for d in diffs if "mel_max_abs_diff" in d]
    errs=[d for d in diffs if "error" in d]
    dit=vg.get_dit(model)
    eager_sizes=dict(getattr(dit,"_eager_sizes",{}))
    res={"n_flush_chunks":len(diffs),"n_valid":len(valid),"n_error":len(errs),
         "flush_sizes":sorted(set(d.get("cs") for d in diffs)),
         "n_first_chunk":sum(1 for d in valid if d.get("first_chunk")),
         "eager_residual_sizes_cs_ge_bucket":eager_sizes,
         "mel_max_abs_diff_overall":round(max([d["mel_max_abs_diff"] for d in valid], default=-1),6),
         "att_cache_max_abs_diff_overall":round(max([d["att_cache_max_abs_diff"] for d in valid], default=-1),6),
         "all_below_0p01": (max([d["mel_max_abs_diff"] for d in valid], default=1) < 0.01) if valid else None,
         "samples":diffs[:20]}
    json.dump(res, open(OUT_DIR/"vocoder_gate.json","w"), ensure_ascii=False, indent=2)
    log("VOCODER BUCKET GATE:", json.dumps(res, ensure_ascii=False, indent=1))
if __name__=="__main__":
    main()
