from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from .modeling_navit_siglip_fast import SiglipAttention
from .modeling_navit_siglip_fast import SiglipFlashAttention2
from .modeling_navit_siglip_fast import SiglipMLP


SUPPORTED_MODEL_TYPE = "uhd_mlp_insert_window_attention_ViTmlp_4_4"


def get_vit_insert_merger(model_type, hidden_size, intermediate_size, vpm, insert_layer_id):
    """Return the only ViT insert merger used by job 93032.

    93032 was launched with:
    ``--model_type uhd_mlp_insert_window_attention_ViTmlp_4_4 --insert_layer_id 6``.
    The unused experimental merger variants from the training tree are intentionally
    not carried in this release repo.
    """
    if model_type != SUPPORTED_MODEL_TYPE:
        raise NotImplementedError(f"Unsupported model_type for MiniCPM-o 4.6: {model_type}")

    return ViTWindowAttentionMerger(vpm, insert_layer_id)


class ViTWindowAttentionMerger(nn.Module):
    def __init__(self, vpm, insert_layer_id):
        super().__init__()
        self.window_kernel_size = (2, 2)
        self.embed_dim = vpm.config.hidden_size
        self._use_flash_attention_2 = vpm.config._attn_implementation == "flash_attention_2"

        self.self_attn = SiglipAttention(vpm.config) if not self._use_flash_attention_2 else SiglipFlashAttention2(vpm.config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=vpm.config.layer_norm_eps)
        self.mlp = SiglipMLP(vpm.config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=vpm.config.layer_norm_eps)

        hidden_4x = self.embed_dim * self.window_kernel_size[0] * self.window_kernel_size[1]
        intermediate_4x = vpm.config.intermediate_size * self.window_kernel_size[0] * self.window_kernel_size[1]

        self.pre_norm = nn.LayerNorm(hidden_4x, eps=1e-6)
        self.linear_1 = nn.Linear(hidden_4x, intermediate_4x, bias=True)
        self.act = ACT2FN["gelu_pytorch_tanh"]
        self.linear_2 = nn.Linear(intermediate_4x, self.embed_dim, bias=True)

        self._init_weight(vpm, insert_layer_id)

    def _init_weight(self, vpm, insert_layer_id):
        copy_block = vpm.encoder.layers[insert_layer_id]

        with torch.no_grad():
            for target_module, src_module in [
                (self.self_attn, copy_block.self_attn),
                (self.layer_norm1, copy_block.layer_norm1),
                (self.layer_norm2, copy_block.layer_norm2),
                (self.mlp, copy_block.mlp),
            ]:
                target_state = target_module.state_dict()
                src_state = src_module.state_dict()
                for key, value in src_state.items():
                    if key in target_state and value.shape == target_state[key].shape:
                        target_state[key].copy_(value)
                target_module.load_state_dict(target_state)

            fc1_old = copy_block.mlp.fc1
            fc2_old = copy_block.mlp.fc2

            hidden = fc1_old.weight.shape[1]
            intermediate = fc1_old.weight.shape[0]

            w_fc1_new = torch.zeros(intermediate * 4, hidden * 4, device=fc1_old.weight.device)
            for i in range(4):
                w_fc1_new[i * intermediate : (i + 1) * intermediate, i * hidden : (i + 1) * hidden] = fc1_old.weight.data
            self.linear_1.weight.copy_(w_fc1_new)
            self.linear_1.bias.copy_(fc1_old.bias.data.repeat(4))

            self.linear_2.weight.copy_(torch.cat([fc2_old.weight.data] * 4, dim=1) / 4.0)
            self.linear_2.bias.copy_(fc2_old.bias.data)

            self.pre_norm.weight.copy_(copy_block.layer_norm2.weight.data.repeat(4))
            self.pre_norm.bias.copy_(copy_block.layer_norm2.bias.data.repeat(4))

    def get_window_index(self, tgt_sizes):
        window_h, window_w = self.window_kernel_size
        max_seqlens = window_h * window_w

        window_index_list = []
        cu_seqlens = [0]
        token_offset = 0
        device = tgt_sizes.device

        for height, width in tgt_sizes:
            height = int(height.item() if hasattr(height, "item") else height)
            width = int(width.item() if hasattr(width, "item") else width)
            if height % window_h != 0 or width % window_w != 0:
                raise ValueError(f"height={height}, width={width} must be divisible by window size ({window_h}, {window_w})")

            index = torch.arange(height * width, device=device).reshape(height, width)
            num_windows_h = height // window_h
            num_windows_w = width // window_w
            num_windows = num_windows_h * num_windows_w

            index = index.reshape(num_windows_h, window_h, num_windows_w, window_w)
            index = index.permute(0, 2, 1, 3).reshape(num_windows, window_h * window_w)
            window_index_list.append(index.reshape(-1) + token_offset)

            cu_this = torch.arange(1, num_windows + 1, device=device) * max_seqlens + cu_seqlens[-1]
            cu_seqlens.extend(cu_this.tolist())
            token_offset += height * width

        window_index = torch.cat(window_index_list)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        return window_index, cu_seqlens, max_seqlens

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.IntTensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlens: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        all_pixel_values = []
        batch_size, _ = tgt_sizes.shape
        for batch_idx in range(batch_size):
            hidden_state = hidden_states[0, cu_seqlens[batch_idx] : cu_seqlens[batch_idx + 1], :].unsqueeze(0)
            tgt_size = tgt_sizes[batch_idx].unsqueeze(0)

            window_index, window_cu_seqlens, window_max_seqlens = self.get_window_index(tgt_size)
            hidden_state = hidden_state[:, window_index, :]

            attn_kwargs = {
                "hidden_states": hidden_state,
                "attention_mask": attention_mask,
                "cu_seqlens": window_cu_seqlens,
                "max_seqlens": window_max_seqlens,
            }
            if self._use_flash_attention_2:
                attn_kwargs["tgt_sizes"] = tgt_size
            hidden_state, _ = self.self_attn(**attn_kwargs)

            all_pixel_values.append(hidden_state[:, torch.argsort(window_index), :])

        hidden_states = torch.concat(all_pixel_values, dim=1)
        hidden_states = residual + hidden_states

        batch_size, _ = tgt_sizes.shape
        all_pixel_values = []
        new_tgt_sizes = torch.zeros_like(tgt_sizes, dtype=tgt_sizes.dtype, device=tgt_sizes.device)

        merge_h, merge_w = self.window_kernel_size
        for batch_idx in range(batch_size):
            height, width = tgt_sizes[batch_idx]
            height = int(height.item() if hasattr(height, "item") else height)
            width = int(width.item() if hasattr(width, "item") else width)
            if height % merge_h != 0 or width % merge_w != 0:
                raise ValueError(f"height={height}, width={width} must be divisible by merge size ({merge_h}, {merge_w})")

            patch = hidden_states[0, cu_seqlens[batch_idx] : cu_seqlens[batch_idx + 1], :].squeeze(0)
            hidden_dim = patch.shape[-1]
            new_height, new_width = height // merge_h, width // merge_w
            patch_5d = (
                patch.view(new_height, merge_h, new_width, merge_w, hidden_dim)
                .permute(0, 2, 1, 3, 4)
            )
            hidden_state = patch_5d.reshape(new_height * new_width, merge_h * merge_w * hidden_dim)
            residual = patch_5d.reshape(new_height * new_width, merge_h * merge_w, hidden_dim).mean(dim=1)

            hidden_state = self.pre_norm(hidden_state)
            hidden_state = self.linear_1(hidden_state)
            hidden_state = self.act(hidden_state)
            hidden_state = self.linear_2(hidden_state)

            all_pixel_values.append(hidden_state + residual)
            new_tgt_sizes[batch_idx, :2] = torch.tensor(
                [new_height, new_width],
                device=new_tgt_sizes.device,
                dtype=new_tgt_sizes.dtype,
            )

        new_hidden_states = torch.concat(all_pixel_values, dim=0).unsqueeze(0)
        new_cu_seqlens = F.pad(
            torch.cumsum(new_tgt_sizes[:, 0] * new_tgt_sizes[:, 1], dim=0, dtype=torch.int32),
            (1, 0),
        )
        if max_seqlens % 4 != 0:
            raise ValueError(f"max_seqlens={max_seqlens} must be divisible by 4")
        new_max_seqlens = max_seqlens // 4

        return new_hidden_states, new_tgt_sizes, attention_mask, new_cu_seqlens, new_max_seqlens
