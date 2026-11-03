"""CPU check of actual deployment attach argument routing; no model/GPU load."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "core/deploy/modes.py"
functions = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {
                 "_env_flag", "_acceleration_config", "_acceleration_setup", "_enable_engine"
             }]
ns = dict(os=os, Any=Any, Dict=Dict, nullcontext=nullcontext)
exec(compile(ast.Module(functions, type_ignores=[]), str(source), "exec"), ns)

calls = []
def attach(model, cfg):
    calls.append((model, cfg))
    model._acceleration_info = {"package": "minicpm_o_accel", **cfg, "resolved_marker": True}

accel = SimpleNamespace(AccelerationConfig=lambda **kwargs: dict(kwargs), attach=attach)
module = ModuleType("core.external_o5")
module.acceleration_module = lambda: accel
model = SimpleNamespace(llm=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:3")])))
base = {"O5_ACCELERATION_PATH": "/isolated/accel", "O5_ACCELERATION_HARDWARE": "pro5000"}
flags = {"O5_TTS_FA2_WHOLE_DECODE": "tts_fa2_whole_decode",
         "O5_CANVAS_WINDOW_EXACT_CHAIN": "canvas_window_exact_chain"}

with patch.dict(sys.modules, {"core.external_o5": module}):
    with patch.dict(os.environ, base, clear=True):
        result = ns["_enable_engine"](model, tp=True, cfg={})
        assert result["acceleration_info"] == model._acceleration_info
        assert result["acceleration_info"]["resolved_marker"] is True
        assert result["acceleration_info"] is not model._acceleration_info
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
    with patch.dict(os.environ, {
        **base, "O5_ACCELERATION_HARDWARE": "a100",
        "O5_PREFILL_GRAPH_MAX_GRAPHS": "1", "O5_T2W_MAX_GRAPHS": "1",
    }, clear=True):
        ns["_enable_engine"](model, tp=False, cfg={})
        assert calls[-1][1]["prefill_graph_max_graphs"] == 1
        assert calls[-1][1]["t2w_max_graphs"] == 1
    with patch.dict(os.environ, {}, clear=True):
        before = len(calls)
        assert ns["_enable_engine"](model, tp=True, cfg={})["acceleration"] == "disabled"
        assert len(calls) == before
print("EXTERNAL_ACCEL_OPTIONS_PASS: opt-in flags, defaults, config precedence, TP/EP/single, caps, disabled")
