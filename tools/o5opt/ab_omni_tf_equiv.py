#!/usr/bin/env python3
"""GOLD-STANDARD per-token omni equivalence: TEACHER-FORCED. Run pure-eager omni greedy, record the
EXACT token at every decode step; then run the FULL optimized stack but FORCE it to replay those same
tokens (identical context by construction) while capturing its logits at every step. Compare REF vs OPT
logits at EVERY token position (not just first-of-unit) — this removes the greedy-cascade confound and
directly measures whether the optimized stack is argmax-preserving PER TOKEN. For any disagreement,
report REF's top1-top2 gap (a small gap => benign near-tie float-reorder; a large gap => real bug)."""
import os, sys, json, types
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")
OUT_DIR = Path(os.environ["OUT_DIR"]); OUT_DIR.mkdir(parents=True, exist_ok=True); os.environ["OUT_DIR"] = str(OUT_DIR)
import minimal_o5_unified_model_duplex as probe
from modeling.o5.opt_flags import OPT
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
    tmp = tempfile.mkdtemp(prefix="omnitf_")
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

# capture buffers for the decode monkeypatch
REC = {"logits": [], "tokens": []}      # REF: (logits, chosen token) per decode call
PLAY = {"i": 0, "opt_logits": [], "forced": []}

def main():
    model=probe.load_o5_model()
    model.init_unified(preload_both_tts=False, duplex_config=DUP, device="cuda", chat_vocoder="token2wav")
    ac,fr=extract_mp4(VIDEO, N_UNITS+2)
    log(f"extracted {len(ac)} audio, {len(fr)} frames")
    sd = model.duplex.decoder
    orig_decode = sd.decode.__func__ if hasattr(sd.decode, "__func__") else sd.decode

    def rec_decode(self, logits, **kw):
        tok = orig_decode(self, logits, **kw)
        tid = int(tok.item()) if torch.is_tensor(tok) else int(tok)
        REC["logits"].append(logits.detach().float().reshape(-1).cpu()); REC["tokens"].append(tid)
        return tok
    def play_decode(self, logits, **kw):
        i = PLAY["i"]; PLAY["i"] += 1
        PLAY["opt_logits"].append(logits.detach().float().reshape(-1).cpu())
        if i < len(REC["tokens"]):
            forced = REC["tokens"][i]; PLAY["forced"].append(forced)
            return torch.tensor([forced], device=logits.device, dtype=torch.long)
        tok = orig_decode(self, logits, **kw)
        return tok

    # ---- REF pass: pure-eager greedy, record every decode ----
    for k in OPT: OPT[k]=False
    os.environ["O5_VISION_BATCH"]=""
    set_experts_impl(model,"eager"); set_attn_impl(model, "sdpa"); set_attn_impl(model.tts.model, "sdpa")
    sd.decode = types.MethodType(rec_decode, sd)
    prep(model, probe.load_16k(REF_WAV))
    for i in range(min(N_UNITS,len(ac))):
        fl=[fr[i]] if (fr and i<len(fr)) else None
        model.duplex_prefill(audio_waveform=ac[i], frame_list=fl)
        model.duplex_generate(decode_mode="greedy", **GEN); model.duplex_finalize()
    n_ref = len(REC["tokens"]); log(f"REF recorded {n_ref} decode steps")
    torch.cuda.empty_cache()

    # ---- OPT pass: full stack, teacher-forced to REF tokens, capture logits ----
    os.environ["O5_VISION_BATCH"]="1"
    OPT.update({"tts_fast":True,"lmhead":True,"tts_graph":True,"fuse_vision_audio":True,"llm_graph":True})
    set_experts_impl(model,"batched_mm")   # eager attn forced by graph runners
    sd.decode = types.MethodType(play_decode, sd)
    prep(model, probe.load_16k(REF_WAV))
    try: vg.enable(model, {50}, log)
    except Exception as e: log("vocoder enable failed:", e)
    for i in range(min(N_UNITS,len(ac))):
        fl=[fr[i]] if (fr and i<len(fr)) else None
        model.duplex_prefill(audio_waveform=ac[i], frame_list=fl)
        model.duplex_generate(decode_mode="greedy", **GEN); model.duplex_finalize()
    n_opt = len(PLAY["opt_logits"]); log(f"OPT captured {n_opt} decode steps (forced {len(PLAY['forced'])})")

    # ---- per-token comparison (identical context by construction) ----
    n = min(n_ref, n_opt)
    agree=0; mx=0.0; diffs=[]; disagree=[]
    for i in range(n):
        A=REC["logits"][i]; B=PLAY["opt_logits"][i]
        if A.shape!=B.shape: continue
        d=float((A-B).abs().max()); mx=max(mx,d); diffs.append(d)
        fa=int(A.argmax()); fb=int(B.argmax())
        if fa==fb: agree+=1
        else:
            top2=torch.topk(A,2).values; gap=float(top2[0]-top2[1])
            disagree.append({"pos":i,"ref_argmax":fa,"opt_argmax":fb,"ref_top1_top2_gap":round(gap,4),"logit_diff":round(d,4)})
    res={"n_steps_compared":n,"per_token_argmax_agree":f"{agree}/{n}",
         "logits_max_abs":round(mx,5),"logits_mean_abs":round(float(np.mean(diffs)) if diffs else 0,6),
         "n_disagree":len(disagree),
         "disagreements":disagree[:30],
         "note":"teacher-forced: identical context per step; disagreements w/ small ref_top1_top2_gap => benign near-tie float-reorder"}
    json.dump(res, open(OUT_DIR/"omni_tf_equiv.json","w"), ensure_ascii=False, indent=2)
    log("TEACHER-FORCED per-token OMNI EQUIV:", json.dumps({k:res[k] for k in
        ["n_steps_compared","per_token_argmax_agree","logits_max_abs","logits_mean_abs","n_disagree"]}, ensure_ascii=False))
    for dd in disagree[:30]: log("  DISAGREE", dd)
if __name__=="__main__":
    main()
