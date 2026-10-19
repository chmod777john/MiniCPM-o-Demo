"""Resolve external HF code without importing the vendored model implementation."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path


def canonical_module(name):
    configured = os.environ.get("O5_CANONICAL_PATH")
    if not configured:
        raise ValueError("O5_CANONICAL_PATH is required: set it to the external HF source directory")
    root = Path(configured).expanduser().resolve()
    for filename in ("configuration_minicpmo.py", "modeling_minicpmo.py", "processing_minicpmo.py"):
        if not (root / filename).is_file():
            raise FileNotFoundError(root / filename)
    package_name = "_demo_external_o5"
    previous = sys.modules.get(package_name)
    if previous is not None and previous.__path__ != [str(root)]:
        raise RuntimeError("O5 canonical path cannot change after importing model code")
    if previous is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        importlib.import_module(package_name + ".processing_minicpmo")
        if (root / "contiguous_rope_qwen35moe.py").is_file():
            rope = importlib.import_module(package_name + ".contiguous_rope_qwen35moe")
            existing = sys.modules.get("contiguous_rope_qwen35moe")
            if existing is not None and Path(existing.__file__).resolve() != Path(rope.__file__).resolve():
                raise RuntimeError("Conflicting contiguous_rope_qwen35moe module")
            sys.modules["contiguous_rope_qwen35moe"] = rope
    return importlib.import_module(package_name + "." + name)


def checkpoint_path(cfg):
    path = os.environ.get("O5_CHECKPOINT_PATH") or cfg.get("pt_path")
    if not path:
        raise ValueError("O5_CHECKPOINT_PATH or pt_path is required for the full O5 checkpoint")
    return path


def checkpoint_state(path):
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "module"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no state dict: {path}")
    return state


def acceleration_module():
    root = Path(os.environ["O5_ACCELERATION_PATH"]).expanduser().resolve()
    source = root / "src"
    if not (source / "minicpm_o_accel").is_dir():
        raise FileNotFoundError(source / "minicpm_o_accel")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("minicpm_o_accel")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Acceleration package loaded from another checkout")
    return module
