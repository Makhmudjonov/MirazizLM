from dataclasses import dataclass


@dataclass
class MirazizConfig:
    vocab_size: int = 16000

    # Model
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6

    # Context
    max_seq_len: int = 512

    # Regularization
    dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0

    # Token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2