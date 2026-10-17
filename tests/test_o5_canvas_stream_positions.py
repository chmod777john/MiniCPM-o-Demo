"""canvas_v6 位置:推理侧逐块发出的位置,拼起来必须与训练侧对整条序列的结果逐位相同。

基准 ``_training_canvas_v6`` 移植自训练侧 Megatron-LM-huawei
``examples/multimodal/minicpmv/train.py`` 的 ``_compute_canvas_v6_position_ids``(commit
8cdaadb),只保留单条序列、无切片这一 user_video 实际会走的情形,其余逻辑(halo、
``</image>`` 的 H/W、游标 ``base + max(h, w) + 1``、相邻帧「紧邻」才合并)逐行保留。

两份输入:
1. SDK Guidebook case 4.01 真实 tokenize 出的整条序列(448×448,每帧 64 = 8×8);
2. 合成的非正方形帧(640×360 → 每帧 66 = 6×11),专门防「行列写反」。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from modeling.o5.canvas_stream import CanvasStreamPositions, compute_llm_grid

FIXTURE = Path(__file__).parent / "fixtures" / "sdk_case_4_01_user_video_tokens.json"


# ---------------------------------------------------------------- 训练侧基准(移植)


def _training_llm_grid(n_tokens: int, tgt_h: int, tgt_w: int) -> tuple[int, int]:
    if n_tokens <= 0 or tgt_h <= 0 or tgt_w <= 0:
        return 0, 0
    vit_total = tgt_h * tgt_w
    if vit_total == n_tokens:
        return tgt_h, tgt_w
    if vit_total < n_tokens:
        return 0, 0
    factor = int(round(math.sqrt(vit_total / n_tokens)))
    if factor > 0 and tgt_h % factor == 0 and tgt_w % factor == 0:
        h, w = tgt_h // factor, tgt_w // factor
        if h * w == n_tokens:
            return h, w
    ratio = tgt_h / tgt_w
    w = max(1, int(round(math.sqrt(n_tokens / ratio))))
    h = n_tokens // w
    if h * w == n_tokens and h > 0:
        return h, w
    return 0, 0


def _training_canvas_v6(ids: list[int], spans: list[tuple[int, int]], tgt_sizes: list[tuple[int, int]],
                        special: dict) -> torch.Tensor:
    """单条序列、无切片的 canvas_v6,返回 ``[3, len]``(T, H, W)。"""

    s_end = len(ids)
    pos3d = torch.arange(s_end).unsqueeze(0).expand(3, -1).clone()
    structural = {special["image_start"], special["image_end"], special["slice_start"], special["slice_end"]}
    groups = [{"thumbnail": (bs, be, gi), "slices": []} for gi, (bs, be) in enumerate(spans)]

    def group_end_pos(group):
        end = group["thumbnail"][1]
        while end < s_end and ids[end] in structural:
            end += 1
        return min(end, s_end)

    def is_tight_gap(prev_group, next_group):
        for pos in range(group_end_pos(prev_group), next_group["thumbnail"][0] - 1):
            if ids[pos] not in structural:
                return False
        return True

    merged, gi = [], 0
    while gi < len(groups):
        frames = [groups[gi]]
        nxt = gi + 1
        while nxt < len(groups) and is_tight_gap(groups[nxt - 1], groups[nxt]):
            frames.append(groups[nxt])
            nxt += 1
        merged.append(frames)
        gi = nxt
    # duplex 里两帧之间总隔着普通 token:训练侧不会把它们合并成一段视频
    assert all(len(frames) == 1 for frames in merged), "duplex 帧不应被训练侧合并"

    pos = 0
    cursor = 0

    def assign_1d(start, end):
        nonlocal pos
        length = end - start
        if length <= 0:
            return
        pos3d[:, start:end] = torch.arange(length) + pos
        pos += length

    for (frame,) in merged:
        t_bs, t_be, t_gi = frame["thumbnail"]
        group_start = t_bs - 1
        group_end = group_end_pos(frame)
        assign_1d(cursor, group_start)
        base = pos
        halo = max(base - 1, 0)
        n_thumb = t_be - t_bs
        llm_h, llm_w = _training_llm_grid(n_thumb, *tgt_sizes[t_gi])
        canvas_h, canvas_w = max(llm_h, 1), max(llm_w, 1)
        pos3d[:, group_start:group_end] = base
        pos3d[1, group_start] = halo
        pos3d[2, group_start] = halo
        if t_be < group_end:
            pos3d[1, t_be] = base + canvas_h
            pos3d[2, t_be] = base + canvas_w
        assert llm_h * llm_w == n_thumb
        h_idx = torch.arange(llm_h).view(-1, 1).expand(-1, llm_w).reshape(-1)
        w_idx = torch.arange(llm_w).view(1, -1).expand(llm_h, -1).reshape(-1)
        pos3d[0, t_bs:t_be] = base
        pos3d[1, t_bs:t_be] = h_idx + base
        pos3d[2, t_bs:t_be] = w_idx + base
        pos = base + max(canvas_h, canvas_w) + 1
        cursor = group_end
    assign_1d(cursor, s_end)
    return pos3d


# ---------------------------------------------------------------- 推理侧:按 FC 的喂法逐块发出


def _stream_positions(ids, spans, tgt_sizes, chunk=7) -> torch.Tensor:
    """模拟 FC:文本按任意长度分块喂,每帧作为一个整组喂。返回拼接后的 ``[4, len]``。"""

    stream = CanvasStreamPositions()
    blocks = []
    cursor = 0
    for (bs, be), (th, tw) in zip(spans, tgt_sizes):
        start = bs - 1                       # <image>
        while cursor < start:                # 文本块:长度任意,检验分块不影响结果
            step = min(chunk, start - cursor)
            blocks.append(stream.text(step)[:, 0])
            cursor += step
        n_tokens = be - bs
        blocks.append(stream.image(n_tokens, compute_llm_grid(n_tokens, th, tw))[:, 0])
        cursor = be + 1                      # 跳过 </image>
    while cursor < len(ids):
        step = min(chunk, len(ids) - cursor)
        blocks.append(stream.text(step)[:, 0])
        cursor += step
    return torch.cat(blocks, dim=1)


def _load_fixture():
    data = json.loads(FIXTURE.read_text())
    return data["input_ids"], [tuple(s) for s in data["image_spans"]], data


def test_sdk_case_4_01_positions_match_training_canvas_v6() -> None:
    ids, spans, data = _load_fixture()
    assert len(spans) == 12 and data["image_token_count_per_frame"] == 64
    tgt = [(32, 32)] * len(spans)            # 448×448 → ViT 32×32 → merger 4× → 8×8
    expected = _training_canvas_v6(ids, spans, tgt, data["special"])
    got = _stream_positions(ids, spans, tgt)
    assert got.shape == (4, len(ids))
    assert torch.equal(got[1:], expected)
    assert torch.equal(got[0], torch.arange(len(ids)))     # text 路 = KV 槽位


@pytest.mark.parametrize("chunk", [1, 3, 64, 10_000])
def test_text_chunking_does_not_change_positions(chunk: int) -> None:
    ids, spans, data = _load_fixture()
    tgt = [(32, 32)] * len(spans)
    expected = _training_canvas_v6(ids, spans, tgt, data["special"])
    assert torch.equal(_stream_positions(ids, spans, tgt, chunk=chunk)[1:], expected)


def test_non_square_frames_match_training_canvas_v6() -> None:
    """640×360 → resize 616×336 → ViT 24×44 → LLM 6×11 = 66。行列写反会在这里失败。"""

    ids, spans, data = _load_fixture()
    sp = data["special"]
    synth, synth_spans = [], []
    i = 0
    for bs, be in spans:                     # 把每帧的 64 个 placeholder 换成 66 个
        synth.extend(ids[i:bs])
        new_bs = len(synth)
        synth.extend([sp["image_placeholder"]] * 66)
        synth_spans.append((new_bs, len(synth)))
        i = be
    synth.extend(ids[i:])
    tgt = [(24, 44)] * len(synth_spans)
    expected = _training_canvas_v6(synth, synth_spans, tgt, sp)
    got = _stream_positions(synth, synth_spans, tgt)
    assert torch.equal(got[1:], expected)
    # 抽查第一帧:H 取行号、W 取列号
    bs, be = synth_spans[0]
    base = int(got[1, bs])
    assert got[2, bs:be].tolist()[:12] == [base] * 11 + [base + 1]
    assert got[3, bs:be].tolist()[:12] == list(range(base, base + 11)) + [base]
    assert int(got[2, be]) == base + 6 and int(got[3, be]) == base + 11


def test_without_images_positions_equal_plain_1d() -> None:
    """纯音频:三路都等于一维位置,与改动前逐位相同。"""

    stream = CanvasStreamPositions()
    got = torch.cat([stream.text(5)[:, 0], stream.text(1)[:, 0], stream.text(9)[:, 0]], dim=1)
    assert torch.equal(got, torch.arange(15).unsqueeze(0).expand(4, -1))


@pytest.mark.parametrize(("size", "vit", "grid"), [
    ((448, 448), (32, 32), (8, 8)),
    ((640, 360), (24, 44), (6, 11)),
    ((360, 640), (44, 24), (11, 6)),
    ((640, 480), (28, 36), (7, 9)),
    ((852, 384), (20, 48), (5, 12)),
])
def test_llm_grid_for_common_resolutions(size, vit, grid) -> None:
    n = grid[0] * grid[1]
    assert compute_llm_grid(n, *vit) == grid == _training_llm_grid(n, *vit)


def test_llm_grid_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="推不出 LLM 网格"):
        compute_llm_grid(61, 24, 44)
