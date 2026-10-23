"""按一个 DeploymentProfile 启动统一双工 FC Demo。

调用者只选择服务端 Profile。脚本根据 Profile 启动单卡或 O5 TP2 backend，
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
                    if (
                        isinstance(profile, O5FcDeploymentProfile)
                        and profile.deployment_mode == "tp2"
                    )
                    else f"{profile.model_family}-single"
                )
            ),
        }
    )
    if (
        isinstance(profile, O5FcDeploymentProfile)
        and profile.deployment_mode == "tp2"
    ):
        if profile.backbone_dir is None:
            raise RuntimeError("O5 TP2 Profile 缺少 backbone_dir")
        environment["BACKBONE_DIR"] = profile.backbone_dir
        # TP2 实时服务的长 prefill 切到 padded_bmm（按专家分组，不复制权重）：
        # - 每个 Unit 几十 token 的小 prefill（< O5_PREFILL_EAGER_MIN_TOKENS）保持 batched_mm；
        # - 工具返回（1500+ token）切到 padded_bmm。cjp 在 A100 TP2 上实测 1536 token
        #   2284ms → 544ms。显式设置的环境变量优先，出问题可退回 batched_mm。
        # 离线评测副本池不走这里。见 modeling/o5/bounded_prefill.py、prefill_moe.py。
        environment.setdefault("O5_PREFILL_EAGER_MIN_TOKENS", "32")
        environment.setdefault("O5_PREFILL_EXPERTS_IMPLEMENTATION", "padded_bmm")
        print(
            "[start_fc_service] prefill experts="
            f"{environment['O5_PREFILL_EXPERTS_IMPLEMENTATION']} "
            f"min_tokens={environment['O5_PREFILL_EAGER_MIN_TOKENS']}",
            flush=True,
        )
        entry = project_dir / "scripts" / "start_o5_tp2_cctl_service.sh"
    else:
        entry = project_dir / "scripts" / "start_o5_cctl_service.sh"

    os.execvpe("bash", ["bash", str(entry)], environment)


if __name__ == "__main__":
    main()
