#!/usr/bin/env python3
"""DEFINITIVE full-stack OMNI equivalence: pure-eager omni (the trusted path) vs the FULL optimized
stack, END-TO-END (closes the transitivity gap: earlier gates verified each lever separately, never
all-together-vs-pure-eager). Real omni duplex path (video frames + audio), GREEDY (deterministic),
comparing per-unit first-decode logits (model.duplex.pending_logits) + greedy text + listen/speak
decision. Isolated from the TTS multinomial cascade (that's a red herring; compare pre-sampling logits).

REF  = all opts OFF: eager MoE, sdpa attn, DynamicCache, no graphs, no fuse/batch (== trusted demo path).
OPT  = batched_mm + tts_fast + lmhead + tts_graph + fuse_vision_audio + batch_vision_feed + llm_graph
       + vocoder graph{50}, LLM/TTS attn forced eager (for capture).
Verdict: argmax-agreement + identical decoded text every unit => numerically equivalent (argmax-preserving
float reorder, the accepted bar). Run REF first (pristine), then OPT (creates persistent graph runners)."""
import os, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")
OUT_DIR = Path(os.environ["OUT_DIR"]); OUT_DIR.mkdir(parents=True, exist_ok=True); os.environ["OUT_DIR"] = str(OUT_DIR)
import minimal_o5_unified_model_duplex as probe
from MiniCPMO45.opt_flags import OPT
import vocoder_graph as vg
VIDEO = os.environ.get("VIDEO", os.path.join(WORKTREE, "assets", "samples", "compile.mp4"))
REF_WAV = os.environ.get("REF_WAV", os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav"))
N_UNITS = int(os.environ.get("N_UNITS", "12"))
DUP = {"generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
       "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3}
GEN = dict(temperature=0.7, top_k=20, top_p=0.8, listen_prob_scale=1.0,
           text_repetition_penalty=1.05, text_repetition_window_size=512, length_penalty=1.1)
def log(*a): print(*a, flush=True)
def extract_mp4(video, n, sr=16000):
    import subprocess, tempfile
    from PIL import Image
    import librosa
    tmp = tempfile.mkdtemp(prefix="omnifeq_")
    wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg","-i",video,"-ar",str(sr),"-ac","1","-t",str(n),"-f","wav","-y",wav], capture_output=True, check=True)
    audio,_ = librosa.load(wav, sr=sr, mono=True)
    cs=sr; na=min(n, len(audio)//cs)
    ac=[audio[i*cs:(i+1)*cs].astype(np.float32) for i in range(na)]
    fdir=os.path.join(tmp,"f"); os.makedirs(fdir, exist_ok=True)
    subprocess.run(["ffmpeg","-i",video,"-vf","fps=1","-t",str(n),os.path.join(fdir,"f%03d.jpg"),"-y"], capture_output=True, check=True)
    fr=[Image.open(os.path.join(fdir,f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    return ac, fr
def set_experts_impl(m, impl):
    seen=set()
    for mod in m.modules():
        c=getattr(mod,"config",None)
        if c is not None and hasattr(c,"_experts_implementation") and id(c) not in seen:
            c._experts_implementation=impl; seen.add(id(c))
def set_attn_impl(m, impl):
    for mod in m.modules():
        c=getattr(mod,"config",None)
        if c is not None and hasattr(c,"_attn_implementation"): c._attn_implementation=impl
def prep(m, ref): m.duplex_prepare(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                                   suffix_system_prompt="<|audio_end|><|im_end|>", ref_audio=ref, prompt_wav_path=REF_WAV)
def run_pass(model, ac, fr, n, optimize):
    if optimize:
        os.environ["O5_VISION_BATCH"]="1"
        OPT.update({"tts_fast":True,"lmhead":True,"tts_graph":True,"fuse_vision_audio":True,"llm_graph":True,
                    "tts_static":False,"tts_greedy":False,"llm_static":False,"vocoder_graph":False})
        set_experts_impl(model,"batched_mm")   # eager attn is forced by the graph runners at creation
    else:
        os.environ["O5_VISION_BATCH"]=""
        for k in OPT: OPT[k]=False
        set_experts_impl(model,"eager"); set_attn_impl(model, os.environ.get("ATTN_IMPLEMENTATION","sdpa"))
        set_attn_impl(model.tts.model, os.environ.get("ATTN_IMPLEMENTATION","sdpa"))
    rows=[]
    prep(model, probe.load_16k(REF_WAV))
    if optimize:
        # capture vocoder cs=50 graph once (off the compared path)
        try: vg.enable(model, {50}, log)
        except Exception as e: log("vocoder enable failed:", e)
    for i in range(min(n,len(ac))):
        fl=[fr[i]] if (fr and i<len(fr)) else None
        model.duplex_prefill(audio_waveform=ac[i], frame_list=fl)
        pl=getattr(model.duplex,"pending_logits",None)
        lg=pl.detach().float().reshape(-1).cpu() if pl is not None else None
        r=model.duplex_generate(decode_mode="greedy", **GEN)
        rows.append({"logits":lg,"text":r.get("text",""),"is_listen":r.get("is_listen"),"n_tts":r.get("n_tts_tokens")})
        model.duplex_finalize()
    return rows
def main():
    model=probe.load_o5_model()
    model.init_unified(preload_both_tts=False, duplex_config=DUP, device="cuda", chat_vocoder="token2wav")
    ac,fr=extract_mp4(VIDEO, N_UNITS+2)
    log(f"extracted {len(ac)} audio, {len(fr)} frames")
    ref=run_pass(model, ac, fr, N_UNITS, optimize=False)   # pure-eager trusted omni FIRST (pristine)
    torch.cuda.empty_cache()
    opt=run_pass(model, ac, fr, N_UNITS, optimize=True)     # full optimized stack (creates graph runners)
    per=[]; text_ok=True; mx=0.0; am_ok=0; nc=0; flip=None; dec_ok=True
    for i,(a,b) in enumerate(zip(ref,opt)):
        tm=(a["text"]==b["text"]); dm=(a["is_listen"]==b["is_listen"])
        text_ok=text_ok and tm; dec_ok=dec_ok and dm; d=None
        if a["logits"] is not None and b["logits"] is not None:
            dd=(a["logits"]-b["logits"]).abs(); d=round(float(dd.max()),5); nc+=1; mx=max(mx,d)
            fa=int(a["logits"].argmax()); fb=int(b["logits"].argmax()); am_ok+=int(fa==fb)
            if fa!=fb and flip is None: flip=i
        per.append({"unit":i,"listen_ref":a["is_listen"],"listen_opt":b["is_listen"],"logit_max_abs":d,
                    "text_match":tm,"decision_match":dm,"ref_text":a["text"][:30],"opt_text":b["text"][:30]})
    res={"n_units":len(per),"llm_text_identical":all(p["ref_text"]==p["opt_text"] for p in per),
         "decision_identical":dec_ok,"logits_max_abs":round(mx,5),"argmax_agree":f"{am_ok}/{nc}",
         "first_argmax_flip_unit":flip,"per_unit":per}
    json.dump(res, open(OUT_DIR/"omni_full_equiv.json","w"), ensure_ascii=False, indent=2)
    log("FULL-STACK OMNI EQUIV (pure-eager vs full-opt):", json.dumps({k:res[k] for k in
        ["n_units","llm_text_identical","decision_identical","logits_max_abs","argmax_agree","first_argmax_flip_unit"]}, ensure_ascii=False))
    for p in per: log("  ", p)
if __name__=="__main__":
    main()
