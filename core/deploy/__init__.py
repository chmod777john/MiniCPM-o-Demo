"""Deployment-mode framework for the MiniCPM-O5 demo.

A pluggable registry of *deployment modes* — each mode knows how to BUILD a ready-to-serve
MiniCPMO model (construct + place weights + init_unified + enable the right optimization/parallel
engine) for a given (rank, world_size). New modes (EP, TP-4, quantized, …) are added by writing a
builder and registering it — nothing else in the serving stack changes.

Shipped modes (see modes.py):
  - single_eager : 1 GPU, trusted eager path (current production default).
  - single_opt   : 1 GPU, CUDA-graph optimization engine (batched_mm + tts/llm/vocoder graphs).
  - tp2          : 2 GPU tensor-parallel backbone (SPMD/torchrun) + graphs + token broadcast.
"""
from .base import DeploymentMode, BuildResult
from .registry import register_mode, get_mode, list_modes
from . import modes as _modes  # noqa: F401  (registers the shipped modes on import)

__all__ = ["DeploymentMode", "BuildResult", "register_mode", "get_mode", "list_modes"]
