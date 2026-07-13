"""Registry of deployment modes."""
from __future__ import annotations
from typing import Dict, List
from .base import DeploymentMode

_MODES: Dict[str, DeploymentMode] = {}


def register_mode(mode: DeploymentMode) -> None:
    if mode.name in _MODES:
        raise ValueError(f"deployment mode {mode.name!r} already registered")
    _MODES[mode.name] = mode


def get_mode(name: str) -> DeploymentMode:
    if name not in _MODES:
        raise ValueError(
            f"unknown deployment mode {name!r}; registered modes: {sorted(_MODES)}"
        )
    return _MODES[name]


def list_modes() -> List[str]:
    return sorted(_MODES)
