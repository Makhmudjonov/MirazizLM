import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(
    head_dim,
    max_seq_len,
    theta=10000.0
):

    freq = 1.0 / (
        theta **
        (
            torch.arange(
                0,
                head_dim,
                2
            ).float() / head_dim
        )
    )

    positions = torch.arange(
        max_seq_len
    ).float()

    freqs = torch.outer(
        positions,
        freq
    )

    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):

    # x:
    # B, H, T, D

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    cos = cos[:x.shape[-2]].to(
        device=x.device,
        dtype=x.dtype
    )

    sin = sin[:x.shape[-2]].to(
        device=x.device,
        dtype=x.dtype
    )

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated_even = (
        x_even * cos -
        x_odd * sin
    )

    rotated_odd = (
        x_even * sin +
        x_odd * cos
    )

    output = torch.empty_like(x)

    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd

    return output


class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = (
            config.dim // config.n_heads
        )

        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(
            config.dim,
            config.dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.dim,
            config.dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.dim,
            config.dim,
            bias=False
        )

        self.out_proj = nn.Linear(
            config.dim,
            config.dim,
            bias=False
        )

        cos, sin = precompute_rope(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False
        )

        self.dropout = config.dropout

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin
        )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True
        )

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            B,
            T,
            C
        )

        return self.out_proj(output)