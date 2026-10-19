"""FC Duplex 流式喂入时的 canvas_v6 位置(user_video 帧的 3D M-RoPE)。

训练侧(Megatron-LM-huawei ``examples/multimodal/minicpmv/train.py`` 的
``_compute_canvas_v6_position_ids``)拿到整条序列后一次算出 ``[T, H, W]`` 三路位置;
推理侧的 FC 是逐 Unit 增量 feed,永远拿不到整条序列。本模块按**同一套规则**逐块发出
位置,跨 feed 保留游标,发出的位置拼起来与训练侧对整条序列的结果逐位相同
(由 ``tests/test_o5_canvas_stream_positions.py`` 对拍)。

规则(只覆盖 user_video 用到的情形:一帧 = 一个 ``<image>…</image>``,不切片):

- 文本 token(含结构 token):三路相等,每个 token 游标 +1;
- 一帧图像组 ``<image> + N 个视觉 token + </image>``,设 LLM 网格 ``h × w``、``base`` = 进入时的游标:

  ============== ======== ================= =================
  token          T        H                 W
  ============== ======== ================= =================
  ``<image>``    base     max(base-1, 0)    max(base-1, 0)
  第 i 行 j 列    base     base + i          base + j
  ``</image>``   base     base + h          base + w
  ============== ======== ================= =================

  之后游标 = ``base + max(h, w) + 1``。

- 训练侧会把「中间只隔着结构 token / 换行」的相邻帧合并成一段视频统一编号;duplex 里两帧
  之间总隔着 ``</user_video_slot>``、音频 slot、AI slot 等普通 token,不会合并,所以这里
  每帧独立成组,与训练一致。

除 T/H/W 外,还输出第 0 路 ``text``:等于该 token 在 KV cache 里的槽位,只给骨干构造
causal mask 用,不进入 RoPE。``CanvasStreamPositions.block`` 返回 ``[4, 1, L]``
(text, T, H, W),Qwen3.5 骨干原生接受这一形状。
"""

from __future__ import annotations

import math

import torch


def compute_llm_grid(n_tokens: int, vit_grid_height: int, vit_grid_width: int) -> tuple[int, int]:
    """由 ViT patch 网格与视觉 token 数反推 LLM 侧的二维网格。

    与训练侧 ``_compute_llm_grid`` 同一规则:先按「每边下采样同一个整数倍」反推,不成再按纵横比
    估算。O5 口径(resize 到 56 的倍数、patch 14、merger 4 倍)下必然命中第一条。

    参数:
        n_tokens: 该帧在 LLM 序列里的视觉 token 数。
        vit_grid_height: ViT 网格高(resize 后高 / patch_size)。
        vit_grid_width: ViT 网格宽。

    返回:
        ``(llm_h, llm_w)``,满足 ``llm_h × llm_w == n_tokens``。

    异常:
        ValueError: 推不出网格。训练侧遇到这种情况会静默退回一维位置;推理侧直接报错,
            因为那意味着视觉 token 数或 resize 尺寸已经和训练口径对不上。
    """

    if n_tokens <= 0 or vit_grid_height <= 0 or vit_grid_width <= 0:
        raise ValueError(f"非法的视觉网格:n_tokens={n_tokens}, vit={vit_grid_height}x{vit_grid_width}")
    vit_total = vit_grid_height * vit_grid_width
    if vit_total == n_tokens:
        return vit_grid_height, vit_grid_width
    factor = int(round(math.sqrt(vit_total / n_tokens))) if vit_total > n_tokens else 0
    if factor > 0 and vit_grid_height % factor == 0 and vit_grid_width % factor == 0:
        height, width = vit_grid_height // factor, vit_grid_width // factor
        if height * width == n_tokens:
            return height, width
    ratio = vit_grid_height / vit_grid_width
    width = max(1, int(round(math.sqrt(n_tokens / ratio))))
    height = n_tokens // width
    if height * width == n_tokens and height > 0:
        return height, width
    raise ValueError(
        f"推不出 LLM 网格:n_tokens={n_tokens}, vit={vit_grid_height}x{vit_grid_width};"
        "视觉 token 数与 resize 尺寸不满足训练口径"
    )


class CanvasStreamPositions:
    """跨多次 feed 持续推进的 canvas_v6 位置游标。

    ``slot`` 是 KV cache 槽位(text 路),``cursor`` 是 T/H/W 共用游标。没有图像时两者相等,
    此时发出的位置与普通一维位置逐位相同。
    """

    def __init__(self) -> None:
        self.slot = 0
        self.cursor = 0

    def reset(self) -> None:
        self.slot = 0
        self.cursor = 0

    def state(self) -> tuple[int, int]:
        return self.slot, self.cursor

    def text(self, length: int, device=None) -> torch.Tensor:
        """``length`` 个文本 token 的位置,形状 ``[4, 1, length]``。"""

        out = torch.empty(4, 1, length, dtype=torch.long, device=device)
        out[0, 0] = torch.arange(self.slot, self.slot + length, device=device)
        out[1:, 0] = torch.arange(self.cursor, self.cursor + length, device=device)
        self.slot += length
        self.cursor += length
        return out

    def image(self, n_tokens: int, llm_grid: tuple[int, int], device=None) -> torch.Tensor:
        """一帧图像组 ``<image> + n_tokens + </image>`` 的位置,形状 ``[4, 1, n_tokens + 2]``。"""

        height, width = int(llm_grid[0]), int(llm_grid[1])
        if height * width != n_tokens:
            raise ValueError(f"LLM 网格 {height}x{width} 与视觉 token 数 {n_tokens} 不符")
        total = n_tokens + 2
        base = self.cursor
        halo = max(base - 1, 0)
        out = torch.empty(4, 1, total, dtype=torch.long, device=device)
        out[0, 0] = torch.arange(self.slot, self.slot + total, device=device)
        out[1, 0] = base
        out[2, 0, 0] = halo
        out[3, 0, 0] = halo
        rows = torch.arange(height, device=device).view(-1, 1).expand(height, width).reshape(-1)
        cols = torch.arange(width, device=device).view(1, -1).expand(height, width).reshape(-1)
        out[2, 0, 1:1 + n_tokens] = rows + base
        out[3, 0, 1:1 + n_tokens] = cols + base
        out[2, 0, total - 1] = base + height
        out[3, 0, total - 1] = base + width
        self.slot += total
        self.cursor = base + max(height, width) + 1
        return out


__all__ = ["CanvasStreamPositions", "compute_llm_grid"]
