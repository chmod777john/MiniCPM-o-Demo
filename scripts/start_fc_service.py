"""按一个 DeploymentProfile 启动统一双工 FC Demo。

调用者只选择服务端 Profile。脚本根据 Profile 启动 O45 单卡或 O5 TP2 backend，
公共 Gateway、Worker、Semantic API v2 和 FC Board 保持完全一致。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.fc_duplex.profiles import (
    O5FcDeploymentProfile,
    apply_fc_deployment_profile_environment,
    load_fc_deployment_profile,
)


def main() -> None:
    """读取 Profile 并用对应的单模型部署脚本替换当前进程。"""

    parser = argparse.ArgumentParser(
        description="Start one O45 or O5 FC deployment from a strict profile"
    )
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    profile_path = Path(args.profile).resolve()
    profile = load_fc_deployment_profile(profile_path)
    apply_fc_deployment_profile_environment(profile)

    project_dir = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_DIR": str(project_dir),
            "FC_DEPLOYMENT_PROFILE": str(profile_path),
            "FC_MODEL_FAMILY": profile.model_family,
            "MODEL_PATH": profile.model_path,
            "PT_PATH": profile.pt_path,
            "WORKER_ID": (
                environment.get("WORKER_ID")
                or f"{profile.model_family}-{profile.profile_id}-worker"
            ),
            "WORKER_GPU_GROUP": (
                environment.get("WORKER_GPU_GROUP")
                or (
                    f"{profile.model_family}-tp2"
                    if isinstance(profile, O5FcDeploymentProfile)
                    else f"{profile.model_family}-single"
                )
            ),
        }
    )
    if isinstance(profile, O5FcDeploymentProfile):
        environment["BACKBONE_DIR"] = profile.backbone_dir
        entry = project_dir / "scripts" / "start_o5_tp2_cctl_service.sh"
    else:
        entry = project_dir / "scripts" / "start_o5_cctl_service.sh"

    os.execvpe("bash", ["bash", str(entry)], environment)


if __name__ == "__main__":
    main()
