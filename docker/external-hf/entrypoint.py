"""Image-local process composition; model inference remains external HF + accel."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_DIR", "/opt/o5/demo"))
PY = sys.executable
ROLE = os.environ.get("O5_ROLE", "worker")
CHILDREN = []
STOPPING = False

def signal_group(proc, signum):
    try:
        os.killpg(proc.pid, signum)
    except ProcessLookupError:
        pass  # Exited between poll and signal; still reap below.

def reap(proc, timeout):
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        signal_group(proc, signal.SIGKILL)
        proc.wait()

def stop(signum=None, frame=None):
    global STOPPING
    if STOPPING:
        return
    STOPPING = True
    for proc in reversed(CHILDREN):
        if proc.poll() is None:
            signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + 30
    for proc in reversed(CHILDREN):
        reap(proc, max(0.1, deadline-time.monotonic()))

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def launch(args):
    if STOPPING:
        raise SystemExit(0)
    print("[image] launch", " ".join(args), flush=True)
    p = subprocess.Popen(args, cwd=ROOT, start_new_session=True)
    CHILDREN.append(p)
    if STOPPING:
        if p.poll() is None:
            signal_group(p, signal.SIGTERM)
        reap(p, 30)
        raise SystemExit(0)
    return p

def wait_health(url, timeout=1200):
    until=time.monotonic()+timeout
    while not STOPPING and time.monotonic()<until:
        if any(p.poll() is not None for p in CHILDREN):
            raise RuntimeError("child exited while waiting for " + url)
        try:
            with urllib.request.urlopen(url,timeout=3) as r:
                if r.status==200:
                    print("[image] ready",url,flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)
    if STOPPING:
        raise SystemExit(0)
    raise TimeoutError(url)

try:
    if ROLE not in {"worker", "gateway", "standalone"}:
        raise ValueError("O5_ROLE must be worker, gateway, or standalone")
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT)
    data=Path(os.environ.get("O5_DATA_DIR","/tmp/o5-data"))
    data.mkdir(parents=True,exist_ok=True)
    cfg={"model":{"model_path":os.environ["O5_CANONICAL_PATH"],
                  "deployment_mode":os.environ["O5_DEPLOY_MODE"],
                  "attn_implementation":os.environ["O5_ATTN_IMPLEMENTATION"],"compile":False},
         "service":{"data_dir":str(data)}}
    if ROLE != "gateway":
        mount=Path(os.environ.get("MODEL_PATH","/models/o5"))
        ckpt=Path(os.environ.get("O5_CHECKPOINT_PATH",str(mount/"model.pt")))
        assets=Path(os.environ.get("O5_ASSETS_DIR",str(mount/"token2wav_causal_vocoder")))
        if not ckpt.is_file() or not (assets/"flow.pt").is_file():
            raise FileNotFoundError(f"MODEL_PATH needs model.pt and token2wav_causal_vocoder: {mount}")
        os.environ["O5_CHECKPOINT_PATH"]=str(ckpt)
        os.environ["O5_ASSETS_DIR"]=str(assets)
        cfg["model"]["pt_path"]=str(ckpt)
    config=data/"config.json"
    config.write_text(json.dumps(cfg,indent=2))
    os.environ["O5_DEMO_CONFIG_PATH"]=str(config)
    print("[image] source",Path("/opt/o5/source-manifest.json").read_text(),flush=True)
    print("[image] runtime",json.dumps({k:os.environ.get(k) for k in ("O5_ROLE","O5_DEPLOY_MODE","O5_ACCELERATION_HARDWARE","O5_CANONICAL_PATH","O5_ACCELERATION_PATH","O5_CHECKPOINT_PATH")}),flush=True)
    gateway=int(os.environ.get("GATEWAY_PORT","8006")); internal=int(os.environ.get("GATEWAY_INTERNAL_PORT","8007"))
    backend=int(os.environ.get("BACKEND_PORT","22500")); worker=int(os.environ.get("WORKER_PORT","22400"))
    if ROLE in {"gateway","standalone"}:
        args=[PY,"gateway.py","--host","0.0.0.0","--port",str(gateway),"--internal-port",str(internal),"--http"]
        launch(args); wait_health(f"http://127.0.0.1:{internal}/health",120)
    if ROLE in {"worker","standalone"}:
        launch([PY,"-m","py_backend.server","--host","127.0.0.1","--port",str(backend),"--gpu-id","0","--model-path",os.environ["O5_CANONICAL_PATH"],"--pt-path",str(ckpt)])
        wait_health(f"http://127.0.0.1:{backend}/health")
        launch([PY,"worker.py","--host","0.0.0.0","--port",str(worker),"--gpu-id","0","--backend-server-url",f"http://127.0.0.1:{backend}"])
        wait_health(f"http://127.0.0.1:{worker}/health",120)
        registry=os.environ.get("GATEWAY_REGISTRY_URL",f"http://127.0.0.1:{internal}/internal/workers" if ROLE=="standalone" else "")
        if registry:
            wid=os.environ.get("WORKER_ID",os.uname().nodename)
            endpoint=os.environ.get("WORKER_ENDPOINT",f"127.0.0.1:{worker}" if ROLE=="standalone" else f"{wid}:{worker}")
            payload=json.dumps({"endpoint":endpoint,"gpu_group":os.environ.get("WORKER_GPU_GROUP",wid),"labels":{"model":"o5","runtime":"external-hf","profile":"raw-ckpt"}}).encode()
            request=urllib.request.Request(registry.rstrip("/")+"/"+wid,data=payload,headers={"content-type":"application/json"},method="PUT")
            with urllib.request.urlopen(request,timeout=20) as r:
                print("[image] registered",r.status,flush=True)
    while not STOPPING:
        for proc in CHILDREN:
            rc=proc.poll()
            if rc is not None:
                raise RuntimeError(f"child pid={proc.pid} exited {rc}")
        time.sleep(1)
finally:
    stop()
