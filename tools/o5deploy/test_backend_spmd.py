#!/usr/bin/env python3
"""Final integration test: the REAL serving stack in tp2 mode.
create_backend(tp2) -> PyTorchBackend.load_model() (framework build on both ranks) -> rank1 enters
worker_loop via model._spmd_mirror; rank0 drives be.duplex_prefill/generate/finalize which now route
through DuplexView._mcall -> the mirror -> model, replayed on rank1 in lockstep. Proves the actual
demo backend serves 2-card end-to-end."""
import os, sys, json, time
import numpy as np, torch
WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)
REF = os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav")

def main():
    from core.processors.backend_factory import create_backend
    conf = {
        "gpu_id": 0,
        "ref_audio_path": REF,
        "duplex_pause_timeout": 60.0,
        "compile": False,
        "chat_vocoder": "token2wav",
        "attn_implementation": "sdpa",
        "deployment_mode": "tp2",
        "llm_cache_len": int(os.environ.get("O5_LLM_CACHE", "8192")),
    }
    # Leave artifact paths absent by default so this smoke test exercises the
    # same local safetensors resolution as the serving entrypoint. Explicit
    # paths remain available for legacy and multi-checkpoint tests.
    for env_key, config_key in {
        "MODEL_PATH": "model_path",
        "PT_PATH": "pt_path",
        "BACKBONE_DIR": "backbone_dir",
        "O5_WEIGHTS_DIR": "weights_dir",
        "O5_ASSETS_DIR": "assets_dir",
    }.items():
        if os.environ.get(env_key):
            conf[config_key] = os.environ[env_key]
    torch.manual_seed(1234); np.random.seed(1234)
    be = create_backend(conf)
    be.load_model()                          # framework tp2 build on BOTH ranks
    model = be.processor.model
    mir = getattr(model, "_spmd_mirror", None)
    assert mir is not None, "no _spmd_mirror on model — framework tp2 wiring broken"
    # force speak (deterministic, both ranks)
    dec = model.duplex.decoder
    for _lid in (getattr(model.duplex, "listen_token_id", None), getattr(dec, "listen_id", None)):
        if _lid is not None and _lid not in dec.forbidden_token_ids: dec.forbidden_token_ids.append(_lid)

    if not mir.is_driver:
        mir.worker_loop(); sys.stdout.flush(); os._exit(0)

    def log(*a): print(*a, flush=True)
    log(f"[backend] driver: engine={be.processor._deploy.engine}")
    import subprocess, tempfile, librosa
    from PIL import Image
    tmp=tempfile.mkdtemp(); wav=os.path.join(tmp,"a.wav"); V=os.path.join(WORKTREE,"assets/samples/compile.mp4")
    subprocess.run(["ffmpeg","-i",V,"-ar","16000","-ac","1","-t","20","-f","wav","-y",wav],capture_output=True,check=True)
    audio,_=librosa.load(wav,sr=16000,mono=True)
    ac=[audio[i*16000:(i+1)*16000].astype(np.float32) for i in range(min(20,len(audio)//16000))]
    fdir=os.path.join(tmp,"f"); os.makedirs(fdir,exist_ok=True)
    subprocess.run(["ffmpeg","-i",V,"-vf","fps=1","-t","20",os.path.join(fdir,"f%03d.jpg"),"-y"],capture_output=True,check=True)
    fr=[Image.open(os.path.join(fdir,f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    na=len(ac); nf=len(fr)

    dv = be.processor.set_duplex_mode()
    dv.prepare(ref_audio_path=REF, prompt_wav_path=REF)   # -> _mcall duplex_prepare -> mirror
    log("[backend] warmup 8 via backend methods (mirrored) ...")
    for u in range(8):
        be.duplex_prefill(audio_waveform=ac[u%na], frame_list=[fr[u%nf]])
        be.duplex_generate(); be.duplex_finalize()
    log("[backend] timed 20 units via REAL backend serving path ...")
    ts=[]
    for u in range(20):
        torch.cuda.synchronize(); t0=time.perf_counter()
        be.duplex_prefill(audio_waveform=ac[u%na], frame_list=[fr[u%nf]])
        r=be.duplex_generate()                       # DuplexGenerateResult (formatted)
        be.duplex_finalize()
        torch.cuda.synchronize(); dt=(time.perf_counter()-t0)*1e3; ts.append((dt, r.is_listen, r.n_tts_tokens))
        if u%5==0: log(f"  unit {u}: {dt:.0f}ms listen={r.is_listen} n_tts={r.n_tts_tokens} text={ (r.text or '')[:24]!r}")
    mir.shutdown()
    sp=[m for m,li,_ in ts if not li]
    summ={"path":"create_backend->PyTorchBackend->DuplexView->mirror->model","world":mir.world_size,
          "engine":be.processor._deploy.engine,"n_speak":len(sp),
          "speak_avg_ms":round(sum(sp)/len(sp),1) if sp else None,"speak_max_ms":round(max(sp),1) if sp else None,
          "all_under_1s":all(m<1000 for m in sp)}
    json.dump({"summary":summ},open(os.path.join(OUT_DIR,"backend_spmd.json"),"w"),indent=1)
    log("[backend] BACKEND-SPMD-OK "+json.dumps(summ))
    log("[backend] PASS: real demo backend served 2-card duplex end-to-end in lockstep." if summ["all_under_1s"] else "[backend] WARN >1s")
    sys.stdout.flush(); os._exit(0)

if __name__=="__main__": main()
