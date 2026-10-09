"""拒识 Demo：think 里识别到的说话人建卡 / 高亮。无工具、无后端接口。"""

from __future__ import annotations

from demo_gallery import demo


@demo(
    demo_id="rejection",
    title="MiniCPM-o · Rejection",
    summary="多人语音环境下只回应已登记的用户；think 中识别到的说话人建卡并高亮",
    order=20,
    styles=("rejection.css",),
)
class RejectionDemo:
    """纯前端 Demo：不需要 router，也不需要自己的 defaults。"""
