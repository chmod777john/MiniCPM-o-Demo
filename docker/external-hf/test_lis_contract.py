"""CPU gates for the package-only historical LIS connection contract."""
import importlib.util
import json
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("entrypoint", Path(__file__).with_name("entrypoint.py"))
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    mounted = root / "platform.json"
    original = '{"recording":{"enabled":false},"service":{"max_queue_size":7},"custom":{"keep":true}}'
    mounted.write_text(original)
    mounted.chmod(0o444)
    env = {"O5_DATA_DIR": str(root / "data")}
    cfg = ep.configure("gateway", env, mounted, root / "effective")
    assert mounted.read_text() == original
    assert cfg["recording"]["enabled"] is False and cfg["custom"]["keep"] is True
    assert cfg["service"]["max_queue_size"] == 7
    assert Path(env["O5_DEMO_CONFIG_PATH"]).read_text() == json.dumps(cfg, indent=2)
    model = root / "unusual-cache-key" / "model"
    (model / "token2wav_causal_vocoder").mkdir(parents=True)
    (model / "model.pt").touch()
    (model / "token2wav_causal_vocoder" / "flow.pt").touch()
    env = {"MODEL_PATH": str(model), "O5_CANONICAL_PATH": "/hf", "O5_DEPLOY_MODE": "tp2",
           "O5_ATTN_IMPLEMENTATION": "sdpa", "O5_DATA_DIR": str(root / "worker-data")}
    cfg = ep.configure("worker", env, root / "absent", root / "worker-effective")
    assert cfg["model"]["pt_path"] == str(model / "model.pt")
    assert env["O5_ASSETS_DIR"] == str(model / "token2wav_causal_vocoder")
    assert cfg["service"]["compile"] is False

ep.validate_publisher_runtime({"worker": {"registration": {"refresh_mode": "periodic_idempotent", "refresh_interval_seconds": 20, "max_backoff_seconds": 20}}})
try:
    ep.validate_publisher_runtime({"worker": {"registration": {"refresh_mode": "startup_retry_only"}}})
except ValueError:
    pass
else:
    raise AssertionError("mismatched template must fail clearly")

cmd, health, port = ep.gateway_command(
    ["--host", "0.0.0.0", "--port", "18006", "--internal-port", "18007", "--http", "--timeout", "321"], {}, {})
assert port == 18007 and health == "http://127.0.0.1:18007/health"
assert cmd[-2:] == ["--timeout", "321"] and cmd.count("--http") == 1
cmd, health, port = ep.gateway_command([], {}, {"gateway": {"public_port": 19006, "registry_port": 19007}})
assert "19006" in cmd and port == 19007 and "--http" in cmd
cmd, _, _ = ep.gateway_command(["--https"], {}, {})
assert "--http" not in cmd

request = ep.registration_request("http://registry:18007/internal/workers/",
                                  {"WORKER_ID": "worker-0", "WORKER_ENDPOINT": "worker-0.headless.ns.svc:22424",
                                   "WORKER_GPU_GROUP": "gpu-0"}, 22424, False)
assert request.full_url == "http://registry:18007/internal/workers/worker-0"
assert request.method == "PUT"
assert json.loads(request.data)["endpoint"] == "worker-0.headless.ns.svc:22424"
now, calls, fail = [0], [], [True]
class Response:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): pass

def opener(req, timeout):
    calls.append(now[0])
    assert timeout == 5
    if fail[0]: raise OSError("gateway unavailable")
    return Response()

registrar = ep.PeriodicRegistration(request, opener=opener, clock=lambda: now[0])
registrar.tick()
assert registrar.next_attempt == 1
registrar.tick()
assert calls == [0]
now[0] = 1
registrar.tick()
assert registrar.next_attempt == 3
now[0], fail[0] = 3, False
registrar.tick()
assert registrar.successes == 1 and registrar.next_attempt == 23
now[0], fail[0] = 23, True  # Gateway disappears after successful initial registration.
registrar.tick()
assert registrar.next_attempt == 24
now[0], fail[0] = 24, False
registrar.tick()
assert registrar.successes == 2 and registrar.next_attempt == 44
now[0], fail[0] = 44, True
for _ in range(9):
    registrar.tick()
    assert registrar.next_attempt - now[0] <= 20
    now[0] = registrar.next_attempt
before = len(calls)
ep.STOPPING = True
registrar.tick()
assert len(calls) == before
print("LIS_CONTRACT_CPU_PASS: readonly config, raw MODEL_PATH, CLI, retry, refresh, restart, bounded backoff, stop")
