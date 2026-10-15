import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        norm = x.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(norm + self.eps)

        return self.weight * x


class SwiGLU(nn.Module):

    def __init__(self, dim):

        super().__init__()

        hidden_dim = int(8 * dim / 3)

        # 64 ga yaxlitlaymiz
        hidden_dim = 64 * (
            (hidden_dim + 63) // 64
        )

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False
        )

        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

    def forward(self, x):

        return self.w2(
            F.silu(self.w1(x)) *
            self.w3(x)
        )