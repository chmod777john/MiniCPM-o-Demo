"""CPU check of actual deployment attach argument routing; no model/GPU load."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "core/deploy/modes.py"
functions = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {"_env_flag", "_enable_engine"}]
ns = dict(os=os, Any=Any, Dict=Dict, external_enabled=lambda: True)
exec(compile(ast.Module(functions, type_ignores=[]), str(source), "exec"), ns)

calls = []
accel = SimpleNamespace(AccelerationConfig=lambda **kwargs: dict(kwargs),
                        attach=lambda model, cfg: calls.append((model, cfg)))
module = ModuleType("core.external_o5")
module.acceleration_module = lambda: accel
model = SimpleNamespace(llm=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:3")])))
base = {"O5_ACCELERATION_PATH": "/isolated/accel", "O5_ACCELERATION_HARDWARE": "pro5000"}
flags = {"O5_TTS_FA2_WHOLE_DECODE": "tts_fa2_whole_decode",
         "O5_CANVAS_WINDOW_EXACT_CHAIN": "canvas_window_exact_chain"}

with patch.dict(sys.modules, {"core.external_o5": module}):
    with patch.dict(os.environ, base, clear=True):
        ns["_enable_engine"](model, tp=True, cfg={})
        cfg = calls[-1][1]
        assert cfg["hardware"] == "pro5000" and cfg["mode"] == "tp"
        assert cfg["devices"] == ("cuda:0", "cuda:1")
        assert not cfg["tts_fa2_whole_decode"] and not cfg["canvas_window_exact_chain"]
        assert all(cfg[k] for k in ("llm_graph", "tts_graph", "t2w_graph", "compact_logits", "merge_unit_feed"))
    for raw in ("1", "0", "false", "False", "no", "NO", "off", "OFF"):
        with patch.dict(os.environ, {**base, **{k: raw for k in flags}}, clear=True):
            ns["_enable_engine"](model, tp=True, cfg={"gpu_id": 2, "parallel": "ep"})
            cfg = calls[-1][1]
            assert all(cfg[k] is (raw == "1") for k in flags.values())
            assert cfg["devices"] == ("cuda:2", "cuda:3") and cfg["mode"] == "ep"
    override = {"tts_fa2_whole_decode": False, "canvas_window_exact_chain": False,
                "hardware": "h20", "devices": ("cuda:4", "cuda:5")}
    with patch.dict(os.environ, {**base, **{k: "1" for k in flags}}, clear=True):
        ns["_enable_engine"](model, tp=True, cfg={"acceleration_config": override})
        assert all(calls[-1][1][k] == v for k, v in override.items())
        assert override == {"tts_fa2_whole_decode": False, "canvas_window_exact_chain": False,
                            "hardware": "h20", "devices": ("cuda:4", "cuda:5")}
    with patch.dict(os.environ, {**base, "O5_ACCELERATION_HARDWARE": "a100"}, clear=True):
        ns["_enable_engine"](model, tp=False, cfg={})
        assert calls[-1][1]["mode"] == "single" and calls[-1][1]["devices"] == ("cuda:3",)
    with patch.dict(os.environ, {}, clear=True):
        before = len(calls)
        assert ns["_enable_engine"](model, tp=True, cfg={})["acceleration"] == "disabled"
        assert len(calls) == before
print("EXTERNAL_ACCEL_OPTIONS_PASS: opt-in flags, defaults, config precedence, TP/EP/single, disabled")
