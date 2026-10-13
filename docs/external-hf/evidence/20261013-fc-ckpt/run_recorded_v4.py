"""Replay historical basic-1 fixture via official v4; preserve prompt/audio/policy."""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/user/weihongliang')
DEMO = ROOT / 'worktrees/o5-fc-dev-external-hf-20261013'
sys.path.insert(0, str(DEMO))
from clients.v4_payload import to_v4_session_init
spec = importlib.util.spec_from_file_location('recorded_probe', Path(__file__).with_name('recorded_probe_v4.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
original_load = probe.load_fixture

def load(root):
    params, results = original_load(root)
    params = to_v4_session_init({'payload': params})['payload']
    params['config']['tts_pad_protocol'] = 'legacy'
    params['config']['decoding_policy'] = 'constrained_decoding'
    return params, results

probe.load_fixture = load
args = SimpleNamespace(
    fixture=Path('/user/chenjinpeng/OmniAgentBench/runs/0812_cjp_2/20260924_064536/basic-1'),
    url='ws://10.156.16.89:18129/v1/realtime?mode=audio',
    profile_id='unprofiled', speech=True, units=8, timeout=180,
    output_dir=ROOT / 'analysis-runs/20261013-o5-three-materials/api-fc-ckpt/recorded-basic1-v4',
    missing_fixture='fail',
)
asyncio.run(probe.run(args))
