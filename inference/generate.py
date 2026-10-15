import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import MirazizConfig
from model.transformer import MirazizLM


# ==============================
# SETTINGS
# ==============================

CHECKPOINT = ROOT / "checkpoints" / "mirazizlm_v0.pt"
TOKENIZER_FILE = ROOT / "tokenizer" / "tokenizer.json"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 50

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# LOAD TOKENIZER
# ==============================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)

eos_id = tokenizer.token_to_id("<|eos|>")


# ==============================
# LOAD MODEL
# ==============================

print("Checkpoint yuklanmoqda...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

config = MirazizConfig(
    **checkpoint["config"]
)

model = MirazizLM(config)

model.load_state_dict(
    checkpoint["model"]
)

model = model.to(device)
model.eval()

print(
    f"MirazizLM loaded | "
    f"step={checkpoint.get('step', '?')} | "
    f"device={device}"
)


# ==============================
# GENERATION
# ==============================

@torch.no_grad()
def generate(prompt):

    ids = tokenizer.encode(prompt).ids

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(MAX_NEW_TOKENS):

        # Context limit
        context = input_ids[
            :,
            -config.max_seq_len:
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device == "cuda")
        ):

            logits, _ = model(context)

        logits = logits[:, -1, :]

        # Temperature
        logits = logits / TEMPERATURE

        # Top-K
        if TOP_K is not None:

            values, _ = torch.topk(
                logits,
                min(TOP_K, logits.size(-1))
            )

            cutoff = values[:, [-1]]

            logits = torch.where(
                logits < cutoff,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits
            )

        probs = F.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        token_id = next_token.item()

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        if token_id == eos_id:
            break

    return tokenizer.decode(
        input_ids[0].tolist()
    )


# ==============================
# CHAT LOOP
# ==============================

print()
print("=" * 60)
print("MirazizLM V0")
print("exit - chiqish")
print("=" * 60)

while True:

    prompt = input("\nPrompt: ").strip()

    if prompt.lower() in {
        "exit",
        "quit"
    }:
        break

    if not prompt:
        continue

    result = generate(prompt)

    print()
    print("MirazizLM:")
    print(result)