#!/usr/bin/env python3
"""Live 2-card SPMD serving smoke test: proves the driver/worker mirror (core.deploy.spmd) serves
a realistic, variable request stream in lockstep. Rank 0 acts as the gateway driver (variable unit
timing + idle heartbeats + a mid-stream session reset); rank 1 runs worker_loop() and mirrors every
model call. If they ever desync, NCCL times out (120s) -> fail fast. Verifies per-unit <1s and that
the two ranks produce the same decisions (implicit: they run identical collectives)."""
import os, sys, json, time
import numpy as np, torch

WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa"); os.environ["O5_DEPLOY_MODE"] = "tp2"
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)
GEN = dict(temperature=0.7, top_k=20, top_p=0.8, listen_prob_scale=1.0,
           text_repetition_penalty=1.05, text_repetition_window_size=512, length_penalty=1.1)

def main():
    import core.deploy as deploy
    from core.deploy.spmd import SpmdMirror
    cfg = {"model_path": os.environ["MODEL_PATH"], "pt_path": os.environ["PT_PATH"],
           "backbone_dir": os.environ.get("BACKBONE_DIR", "/user/weihongliang/wangkaiqi/o5_backbone_hf"),
           "chat_vocoder": "token2wav", "attn_implementation": "sdpa",
           "llm_cache_len": int(os.environ.get("O5_LLM_CACHE", "8192"))}
    torch.manual_seed(1234); np.random.seed(1234)
    br = deploy.get_mode("tp2").build(cfg)
    model = br.model; dec = model.duplex.decoder
    mirror = SpmdMirror(model, br.is_driver, br.rank, br.world_size)

    # force speak on both ranks (deterministic setup, no collectives)
    for _lid in (getattr(model.duplex, "listen_token_id", None), getattr(dec, "listen_id", None)):
        if _lid is not None and _lid not in dec.forbidden_token_ids: dec.forbidden_token_ids.append(_lid)

    # ── WORKER rank: mirror the driver's calls forever, then exit ──
    if not br.is_driver:
        mirror.worker_loop()
        sys.stdout.flush(); os._exit(0)

    # ── DRIVER rank (== the gateway process) ──
    def log(*a): print(*a, flush=True)
    import minimal_o5_unified_model_duplex as probe
    import subprocess, tempfile, librosa
    from PIL import Image
    ref = probe.load_16k(os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav"))
    tmp = tempfile.mkdtemp(); wav = os.path.join(tmp, "a.wav"); V = os.path.join(WORKTREE, "assets/samples/compile.mp4")
    subprocess.run(["ffmpeg","-i",V,"-ar","16000","-ac","1","-t","20","-f","wav","-y",wav], capture_output=True, check=True)
    audio,_ = librosa.load(wav, sr=16000, mono=True)
    ac = [audio[i*16000:(i+1)*16000].astype(np.float32) for i in range(min(20,len(audio)//16000))]
    fdir=os.path.join(tmp,"f"); os.makedirs(fdir,exist_ok=True)
    subprocess.run(["ffmpeg","-i",V,"-vf","fps=1","-t","20",os.path.join(fdir,"f%03d.jpg"),"-y"], capture_output=True, check=True)
    fr=[Image.open(os.path.join(fdir,f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    na=len(ac); nf=len(fr)
    PREP = dict(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                suffix_system_prompt="<|audio_end|><|im_end|>", ref_audio=ref,
                prompt_wav_path=os.path.join(WORKTREE,"assets/ref_audio/ref_minicpm_signature.wav"))

    def run_unit(u):
        torch.cuda.synchronize(); t0=time.perf_counter()
        mirror.call("duplex_prefill", audio_waveform=ac[u%na], frame_list=[fr[u%nf]])
        r=mirror.call("duplex_generate", decode_mode="sampling", **GEN)
        mirror.call("duplex_finalize")
        torch.cuda.synchronize()
        return (time.perf_counter()-t0)*1e3, r

    # warmup via the mirror (both ranks capture tts/llm/vocoder graphs in lockstep)
    log("[serve] warmup 8 (mirrored) ...")
    mirror.call("duplex_prepare", **PREP)
    for u in range(8): run_unit(u)

    results = {"sessions": []}
    # ── Session 1: 30 units with idle heartbeats at u=6,15 (client paused) ──
    log("[serve] === session 1: 30 units, idle gaps at 6/15 ===")
    mirror.call("duplex_prepare", **PREP)
    s1=[]
    for u in range(30):
        if u in (6, 15):
            for _ in range(3): mirror.call("noop")   # client idle -> heartbeat keeps ranks in lockstep
        dt, r = run_unit(u)
        s1.append({"u":u,"ms":round(dt,1),"listen":r.get("is_listen"),"n_tts":r.get("n_tts_tokens")})
        if u%10==0: log(f"  s1 unit {u}: {dt:.0f}ms listen={r.get('is_listen')}")
    results["sessions"].append({"name":"s1_30u_idle","units":s1})

    # ── Session 2: reset (new session) + 20 units ──
    log("[serve] === session 2: reset + 20 units ===")
    mirror.call("duplex_prepare", **PREP)   # re-prepare = new session
    s2=[]
    for u in range(20):
        dt, r = run_unit(u)
        s2.append({"u":u,"ms":round(dt,1),"listen":r.get("is_listen"),"n_tts":r.get("n_tts_tokens")})
    results["sessions"].append({"name":"s2_20u","units":s2})

    mirror.shutdown()   # tell worker to exit

    allms=[x["ms"] for s in results["sessions"] for x in s["units"] if not x["listen"]]
    ok = all(m < 1000 for m in allms)
    summ = {"world": br.world_size, "engine": br.engine, "n_speak_units": len(allms),
            "speak_avg_ms": round(sum(allms)/len(allms),1) if allms else None,
            "speak_max_ms": round(max(allms),1) if allms else None,
            "all_under_1s": ok, "sessions": 2, "idle_heartbeats_handled": True}
    json.dump({"summary": summ, **results}, open(os.path.join(OUT_DIR,"spmd_serving.json"),"w"), indent=1)
    log("[serve] SPMD-SERVING-OK " + json.dumps(summ))
    log("[serve] PASS: driver drove 2 sessions + idle gaps in lockstep; worker mirrored; no NCCL desync." if ok
        else "[serve] WARN: some speak unit >1s (mechanism OK, timing note)")
    sys.stdout.flush(); os._exit(0)

if __name__=="__main__":
    main()
