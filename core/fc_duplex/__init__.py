"""O45/O5 共享双工 FC 核心抽象。

本包集中维护部署 Profile、模型 Adapter 与共享 View 的稳定边界。公共 API 和业务
Runtime 不应直接依赖具体模型实现。
"""

from .model_adapter import (
    FcDuplexModelAdapter,
    O45FcDuplexModelAdapter,
    O5FcDuplexModelAdapter,
    create_fc_duplex_model_adapter,
)
from .profiles import (
    FcDeploymentProfile,
    O45FcDeploymentProfile,
    O5FcDeploymentProfile,
    apply_fc_deployment_profile_environment,
    load_fc_deployment_profile,
)

__all__ = [
    "FcDeploymentProfile",
    "FcDuplexModelAdapter",
    "O45FcDeploymentProfile",
    "O45FcDuplexModelAdapter",
    "O5FcDeploymentProfile",
    "O5FcDuplexModelAdapter",
    "apply_fc_deployment_profile_environment",
    "create_fc_duplex_model_adapter",
    "load_fc_deployment_profile",
]
