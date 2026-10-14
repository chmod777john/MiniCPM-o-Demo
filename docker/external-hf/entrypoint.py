"""Image-local process composition; model inference remains external HF + accel."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_DIR", "/opt/o5/demo"))
PY = sys.executable
CHILDREN = []
STOPPING = False
# The historical LIS worker does not mount publisher-runtime. These image-owned
# values must match the selected template's periodic_idempotent contract.
REGISTRATION_REFRESH_SECONDS = 20
REGISTRATION_MAX_BACKOFF_SECONDS = 20
REGISTRATION_HTTP_TIMEOUT_SECONDS = 5


def signal_group(proc, signum):
    try:
        os.killpg(proc.pid, signum)
    except ProcessLookupError:
        pass


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
        reap(proc, max(0.1, deadline - time.monotonic()))


def launch(args):
    if STOPPING:
        raise SystemExit(0)
    print("[image] launch", " ".join(args), flush=True)
    proc = subprocess.Popen(args, cwd=ROOT, start_new_session=True)
    CHILDREN.append(proc)
    if STOPPING:
        if proc.poll() is None:
            signal_group(proc, signal.SIGTERM)
        reap(proc, 30)
        raise SystemExit(0)
    return proc


def wait_health(url, timeout=1200):
    until = time.monotonic() + timeout
    while not STOPPING and time.monotonic() < until:
        if any(p.poll() is not None for p in CHILDREN):
            raise RuntimeError("child exited while waiting for " + url)
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    print("[image] ready", url, flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)
    if STOPPING:
        raise SystemExit(0)
    raise TimeoutError(url)


def read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return value


def configure(role, env, platform_config=Path("/app/config.json"),
              output_dir=Path("/tmp/o5-entrypoint")):
    """Copy platform values into an effective config; never write its mount."""
    supplied = Path(env["O5_DEMO_CONFIG_PATH"]) if env.get("O5_DEMO_CONFIG_PATH") else platform_config
    cfg = read_object(supplied) if supplied.is_file() else {}
    data = Path(env.get("O5_DATA_DIR", "/app/data" if supplied.is_file() else "/tmp/o5-data"))
    data.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("service", {}).setdefault("data_dir", str(data))
    if role != "gateway":
        mount = Path(env.get("MODEL_PATH", "/models/o5"))
        ckpt = Path(env.get("O5_CHECKPOINT_PATH", str(mount / "model.pt")))
        assets = Path(env.get("O5_ASSETS_DIR", str(mount / "token2wav_causal_vocoder")))
        if not ckpt.is_file() or not (assets / "flow.pt").is_file():
            raise FileNotFoundError(f"MODEL_PATH needs model.pt and token2wav_causal_vocoder: {mount}")
        env["O5_CHECKPOINT_PATH"] = str(ckpt)
        env["O5_ASSETS_DIR"] = str(assets)
        cfg.setdefault("model", {}).update(
            model_path=env["O5_CANONICAL_PATH"], deployment_mode=env["O5_DEPLOY_MODE"],
            attn_implementation=env["O5_ATTN_IMPLEMENTATION"], pt_path=str(ckpt))
        cfg["service"]["compile"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / "config.json"
    if config.resolve() == supplied.resolve():
        raise ValueError("effective config path must not overwrite supplied config")
    config.write_text(json.dumps(cfg, indent=2))
    env["O5_DEMO_CONFIG_PATH"] = str(config)
    print("[image] config", json.dumps({"supplied": str(supplied), "effective": str(config),
                                       "recording": cfg.get("recording"), "data_dir": cfg["service"]["data_dir"]}), flush=True)
    return cfg


def validate_publisher_runtime(runtime):
    registration = runtime.get("worker", {}).get("registration", {})
    expected = {"refresh_mode": "periodic_idempotent",
                "refresh_interval_seconds": REGISTRATION_REFRESH_SECONDS,
                "max_backoff_seconds": REGISTRATION_MAX_BACKOFF_SECONDS}
    for key, value in expected.items():
        if key in registration and registration[key] != value:
            raise ValueError(f"publisher-runtime {key} must match image contract: {value}")
    bind_host = runtime.get("worker", {}).get("backend_bind_host", "0.0.0.0")
    if bind_host != "0.0.0.0":
        raise ValueError("publisher-runtime backend_bind_host must be 0.0.0.0")


def gateway_command(argv, env, runtime):
    gateway = runtime.get("gateway", {})
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(gateway.get("public_port", env.get("GATEWAY_PORT", 8006))))
    parser.add_argument("--internal-port", type=int, default=int(gateway.get("registry_port", env.get("GATEWAY_INTERNAL_PORT", 8007))))
    parser.add_argument("--internal-host")
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--http", action="store_true")
    protocol.add_argument("--https", action="store_true")
    args, _ = parser.parse_known_args(argv)
    command = [PY, "gateway.py", "--host", args.host, "--port", str(args.port),
               "--internal-port", str(args.internal_port)]
    if not args.http and not args.https:
        command.append("--http")
    command.extend(argv)  # Gateway itself validates all remaining CLI options.
    host = args.internal_host or args.host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return command, f"http://{host}:{args.internal_port}/health", args.internal_port


def registration_request(registry, env, worker, standalone):
    wid = env.get("WORKER_ID", os.uname().nodename)
    endpoint = env.get("WORKER_ENDPOINT", f"127.0.0.1:{worker}" if standalone else f"{wid}:{worker}")
    payload = json.dumps({"endpoint": endpoint, "gpu_group": env.get("WORKER_GPU_GROUP", wid),
                          "labels": {"model": "o5", "runtime": "external-hf", "profile": "raw-ckpt"}}).encode()
    url = registry.rstrip("/") + "/" + urllib.parse.quote(wid, safe="")
    return urllib.request.Request(url, data=payload, headers={"content-type": "application/json"}, method="PUT")


class PeriodicRegistration:
    """One bounded request from the supervisor loop; no background orphan."""
    def __init__(self, request, opener=urllib.request.urlopen, clock=time.monotonic):
        self.request, self.opener, self.clock = request, opener, clock
        self.next_attempt = 0
        self.backoff = 1
        self.successes = 0
        self.failures = 0

    def tick(self):
        if STOPPING or self.clock() < self.next_attempt:
            return
        try:
            with self.opener(self.request, timeout=REGISTRATION_HTTP_TIMEOUT_SECONDS) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"registration status {response.status}")
            self.successes += 1
            self.backoff = 1
            delay = REGISTRATION_REFRESH_SECONDS
            print(f"[image] registered success={self.successes} refresh_seconds={delay}", flush=True)
        except Exception as exc:
            self.failures += 1
            delay = self.backoff
            self.backoff = min(2 * self.backoff, REGISTRATION_MAX_BACKOFF_SECONDS)
            print(f"[image] registration retry failures={self.failures} delay_seconds={delay} error={exc}", flush=True)
        self.next_attempt = self.clock() + delay


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    role = os.environ.get("O5_ROLE", "worker")
    if role not in {"worker", "gateway", "standalone"}:
        raise ValueError("O5_ROLE must be worker, gateway, or standalone")
    if role == "worker" and argv:
        raise ValueError("worker image accepts environment contract, not gateway CLI arguments")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        os.chdir(ROOT)
        os.environ["PYTHONPATH"] = str(ROOT)
        configure(role, os.environ)
        runtime_path = Path("/etc/lis/publisher-runtime/runtime.json")
        runtime = read_object(runtime_path) if runtime_path.is_file() else {}
        validate_publisher_runtime(runtime)
        print("[image] publisher-runtime", json.dumps(runtime), flush=True)
        print("[image] source", Path("/opt/o5/source-manifest.json").read_text(), flush=True)
        print("[image] runtime", json.dumps({k: os.environ.get(k) for k in (
            "O5_ROLE", "O5_DEPLOY_MODE", "O5_ACCELERATION_HARDWARE", "O5_CANONICAL_PATH",
            "O5_ACCELERATION_PATH", "O5_CHECKPOINT_PATH")}), flush=True)
        internal = int(os.environ.get("GATEWAY_INTERNAL_PORT", 8007))
        backend = int(os.environ.get("BACKEND_PORT", 22500))
        worker = int(os.environ.get("WORKER_PORT", 22400))
        registrar = None
        if role in {"gateway", "standalone"}:
            args, health, internal = gateway_command(argv, os.environ, runtime)
            launch(args)
            wait_health(health, 120)
        if role in {"worker", "standalone"}:
            launch([PY, "-m", "py_backend.server", "--host", "0.0.0.0", "--port", str(backend),
                    "--gpu-id", "0", "--model-path", os.environ["O5_CANONICAL_PATH"],
                    "--pt-path", os.environ["O5_CHECKPOINT_PATH"]])
            wait_health(f"http://127.0.0.1:{backend}/health")
            launch([PY, "worker.py", "--host", "0.0.0.0", "--port", str(worker), "--gpu-id", "0",
                    "--backend-server-url", f"http://127.0.0.1:{backend}"])
            wait_health(f"http://127.0.0.1:{worker}/health", 120)
            registry = os.environ.get("GATEWAY_REGISTRY_URL", f"http://127.0.0.1:{internal}/internal/workers" if role == "standalone" else "")
            if registry:
                registrar = PeriodicRegistration(registration_request(registry, os.environ, worker, role == "standalone"))
            else:
                print("[image] no GATEWAY_REGISTRY_URL; worker has no gateway registration", flush=True)
        while not STOPPING:
            for proc in CHILDREN:
                rc = proc.poll()
                if rc is not None:
                    raise RuntimeError(f"child pid={proc.pid} exited {rc}")
            if registrar:
                registrar.tick()
            time.sleep(1)
    finally:
        stop()


if __name__ == "__main__":
    main()
