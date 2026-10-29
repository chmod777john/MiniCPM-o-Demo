"""人评专项 Demo（human_eval_1029_rejection）：交互复制自 rejection demo，模型端点只接人评网关。"""

from __future__ import annotations

from demo_gallery import demo


@demo(
    demo_id="human_eval_1029_rejection",
    title="Human Eval 1029 · Rejection",
    summary="人评专项：merge0930 / merge1018 两个模型的拒识行为对比；交互与线上 rejection demo 一致",
    order=20,
    styles=("human_eval_1029_rejection.css",),
)
class RejectionDemo:
    """纯前端 Demo：不需要 router，也不需要自己的 defaults。"""
