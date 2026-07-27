import torch
from torch import nn


class DownsampleMLP(nn.Module):
    def __init__(self, hidden_size, llm_embed_dim, merge_kernel_size=(2, 2)):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size

        self.hidden_size = (
            hidden_size
            * self.merge_kernel_size[0]
            * self.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_embed_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.pre_norm(x).view(-1, self.hidden_size))
        return x


class Merger(nn.Module):
    def __init__(self, hidden_size, llm_embed_dim, merge_kernel_size=(2, 2), times=1):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.times = times
        self.mlp = nn.ModuleList([DownsampleMLP(hidden_size, llm_embed_dim if i==times-1 else hidden_size, merge_kernel_size) for i in range(times)])


    def forward(self, hidden_states: torch.Tensor, tgt_sizes: torch.IntTensor,) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size

        start = 0
        processed_features = []
        for batch_idx in range(len(tgt_sizes)):
            h, w = tgt_sizes[batch_idx]
            h = int(h.item() if hasattr(h, "item") else h)
            w = int(w.item() if hasattr(w, "item") else w)
            if h % m1 != 0 or w % m2 != 0:
                raise ValueError(
                    f"height={h}, width={w} must be divisible by merge size ({m1}, {m2})"
                )
            num_patches = h * w

            hidden_dim = hidden_states.shape[-1]
            h_new, w_new = h // m1, w // m2
            _hidden_state = (
                hidden_states[0, start : start + num_patches, :]
                .view(h_new, m1, w_new, m2, hidden_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(h_new * w_new, m1 * m2 * hidden_dim)
            )
            _hidden_state = self.mlp[0](_hidden_state)

            if self.times > 1:
                for i in range(1, self.times):
                    if h % m1 != 0 or w % m2 != 0:
                        raise ValueError(
                            f"height={h}, width={w} must be divisible by merge size ({m1}, {m2})"
                        )
                    h = h // 2
                    w = w // 2

                    hidden_dim = _hidden_state.shape[-1]
                    h_new, w_new = h // m1, w // m2
                    _hidden_state = (
                        _hidden_state.view(h_new, m1, w_new, m2, hidden_dim)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(h_new * w_new, m1 * m2 * hidden_dim)
                    )
                    _hidden_state = self.mlp[i](_hidden_state)

            start += num_patches
            processed_features.append(_hidden_state)

        return processed_features
