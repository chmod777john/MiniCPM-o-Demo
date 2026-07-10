# tools/o5opt — 验证脚本

## 数值等价 gate（换权重后重跑这两个即可确认无回归）
- `ab_omni_full_equiv.py` — pure-eager omni vs 全优化栈，真实 omni duplex 路径，贪心，逐 unit 比首 token argmax + listen/speak 决策 + 文本轨迹。
- `ab_omni_tf_equiv.py` — teacher-forced 逐 token（把 pure-eager 的 token 强制喂进优化栈，上下文逐位相同），比每一步 argmax。**黄金标准**；对分歧报 top1-top2 gap（gap 小 => 良性近似平手，非 bug）。

在 iter1200 上的结论：决策/首token argmax 全一致；teacher-forced 逐 token argmax 47/48，唯一分歧 gap=0.125（近似平手，两个都合法）。

## 跑法
```bash
cd <repo-root>            # WORKTREE = 仓库根（自带 MiniCPMO45/ + scripts/ + assets/）
export PYTHONPATH=$PWD WORKTREE=$PWD
source <venv>/bin/activate
export MODEL_PATH=<你的基础模型路径> PT_PATH=<你的 .pt 或留空>
export OUT_DIR=/tmp/o5eq ATTN_IMPLEMENTATION=sdpa N_UNITS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tools/o5opt/ab_omni_full_equiv.py   # 端到端决策/argmax
python tools/o5opt/ab_omni_tf_equiv.py     # teacher-forced 逐 token（决定性）
```
判据：`decision_identical=True`、`argmax_agree` 接近满、分歧处 `ref_top1_top2_gap` 很小（< 重排噪声 ~1.0）。

注：这两个脚本用 `minimal_o5_unified_model_duplex.load_o5_model()`（空初始化 + load_state_dict，适配"代码目录+.pt"打包）；正常 HF checkpoint 可改用 from_pretrained。
