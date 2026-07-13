"""Deployment-mode abstraction."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class BuildResult:
    """What a deployment-mode builder returns."""
    model: Any                       # ready MiniCPMO: weights placed, init_unified done, engine enabled
    world_size: int                  # ranks participating (1 = single process, >1 = SPMD/torchrun)
    rank: int                        # this process's rank
    is_driver: bool                  # True on the rank that talks to the gateway (rank 0)
    engine: Dict[str, Any] = field(default_factory=dict)   # what optimization/parallel engine was enabled
    broadcast_input: Optional[Callable] = None             # SPMD: rank0 -> all ranks input sync (None if single)


@dataclass
class DeploymentMode:
    """A pluggable way to deploy the model.

    build(config, rank, world_size) -> BuildResult   constructs the ready model for this rank.
    world_size is the number of GPUs/ranks the mode needs. requires_spmd=True means the serving
    process must be launched under torchrun with `world_size` ranks (all run the same code in
    lockstep; rank 0 drives the gateway, other ranks mirror the compute for collectives).
    """
    name: str
    world_size: int
    requires_spmd: bool
    build: Callable[..., BuildResult]
    description: str = ""

    def __repr__(self) -> str:
        return f"DeploymentMode(name={self.name!r}, world_size={self.world_size}, spmd={self.requires_spmd})"
