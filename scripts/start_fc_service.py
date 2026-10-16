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
        # 实时服务每个 Unit 都有几十 token 的小 prefill；有界 prefill 默认会把它们也切到逐专家循环的
        # eager MoE（~470ms/Unit），超出 1 秒/Unit 的实时预算、语音一卡一卡。TP2 专家已切半，256 token
        # 以下用 batched_mm 的瞬时显存有限，开场长 system prefill 仍走 eager。显式设置的值优先；
        # 离线评测副本池不走这里，不受影响。见 modeling/o5/bounded_prefill.py。
        environment.setdefault("O5_PREFILL_EAGER_MIN_TOKENS", "256")
        # 开场长 system prefill 也不切 eager：TP2 下 batched_mm 按 512 token 分块，瞬时显存每块 ~8 GiB，
        # 建会话快约 2.8 倍（2×A100 实测 5021 token 22.1→7.7 秒，峰值显存 44→56 GiB，输出逐 Unit 一致）。
        environment.setdefault("O5_PREFILL_EXPERTS_IMPLEMENTATION", "batched_mm")
        entry = project_dir / "scripts" / "start_o5_tp2_cctl_service.sh"
    else:
        entry = project_dir / "scripts" / "start_o5_cctl_service.sh"

    os.execvpe("bash", ["bash", str(entry)], environment)


if __name__ == "__main__":
    main()
