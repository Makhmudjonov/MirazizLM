import sys
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import MirazizConfig
from model.transformer import MirazizLM
from training.dataset import TokenDataset


# =============================
# CONFIG
# =============================

BATCH_SIZE = 8

MAX_STEPS = 3000

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.1

LOG_INTERVAL = 10

EVAL_INTERVAL = 200

EVAL_STEPS = 20

SAVE_INTERVAL = 500


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("MirazizLM PRE-TRAINING")
print("=" * 60)

print("Device:", device)

if device == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =============================
# MODEL
# =============================

config = MirazizConfig()

model = MirazizLM(config)

model = model.to(device)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {parameters:,}"
)


# =============================
# DATASET
# =============================

train_dataset = TokenDataset(
    ROOT / "data/tokenized/train.bin",
    block_size=config.max_seq_len
)

val_dataset = TokenDataset(
    ROOT / "data/tokenized/val.bin",
    block_size=config.max_seq_len
)


# =============================
# OPTIMIZER
# =============================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =============================
# MIXED PRECISION
# =============================

use_amp = device == "cuda"

dtype = torch.bfloat16

print(
    "Mixed precision:",
    use_amp
)

print(
    "Training dtype:",
    dtype
)


# =============================
# EVALUATION
# =============================

@torch.no_grad()
def evaluate():

    model.eval()

    losses = []

    for _ in range(EVAL_STEPS):

        x, y = val_dataset.get_batch(
            BATCH_SIZE,
            device
        )

        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=use_amp
        ):

            _, loss = model(
                x,
                y
            )

        losses.append(
            loss.item()
        )

    model.train()

    return sum(losses) / len(losses)


# =============================
# TRAIN
# =============================

model.train()

start_time = time.time()

for step in range(1, MAX_STEPS + 1):

    x, y = train_dataset.get_batch(
        BATCH_SIZE,
        device
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=use_amp
    ):

        _, loss = model(
            x,
            y
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    if step % LOG_INTERVAL == 0:

        elapsed = (
            time.time() - start_time
        )

        tokens_seen = (
            step
            * BATCH_SIZE
            * config.max_seq_len
        )

        tokens_per_sec = (
            tokens_seen / elapsed
        )

        print(
            f"step {step:5d} | "
            f"loss {loss.item():.4f} | "
            f"{tokens_per_sec:,.0f} tok/s"
        )

    if step % EVAL_INTERVAL == 0:

        val_loss = evaluate()

        print(
            f">>> VALIDATION "
            f"step {step}: "
            f"{val_loss:.4f}"
        )

    if step % SAVE_INTERVAL == 0:

        checkpoint = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.__dict__,
        }

        path = (
            ROOT
            / "checkpoints"
            / f"mirazizlm_step_{step}.pt"
        )

        torch.save(
            checkpoint,
            path
        )

        print(
            f">>> CHECKPOINT: {path}"
        )


# =============================
# FINAL SAVE
# =============================

final_path = (
    ROOT
    / "checkpoints"
    / "mirazizlm_v0.pt"
)

torch.save(
    {
        "step": MAX_STEPS,
        "model": model.state_dict(),
        "config": config.__dict__,
    },
    final_path
)

print()
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    "Model:",
    final_path
)