import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MirazizConfig
from .layers import RMSNorm, SwiGLU
from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.attn_norm = RMSNorm(
            config.dim
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ffn_norm = RMSNorm(
            config.dim
        )

        self.feed_forward = SwiGLU(
            config.dim
        )

    def forward(self, x):

        x = x + self.attention(
            self.attn_norm(x)
        )

        x = x + self.feed_forward(
            self.ffn_norm(x)
        )

        return x


class MirazizLM(nn.Module):

    def __init__(
        self,
        config: MirazizConfig
    ):

        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.dim
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(
            config.dim
        )

        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        targets=None
    ):

        x = self.token_embedding(
            input_ids
        )

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                targets.view(-1)
            )

        return logits, loss