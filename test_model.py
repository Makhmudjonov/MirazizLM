import torch

from model.config import MirazizConfig
from model.transformer import MirazizLM


config = MirazizConfig()

model = MirazizLM(config)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=" * 50)
print("MirazizLM V0")
print("=" * 50)

print(f"Parameters: {parameters:,}")
print(f"Trainable:  {trainable:,}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

model = model.to(device)

# Fake tokenlar
x = torch.randint(
    0,
    config.vocab_size,
    (2, 128),
    device=device
)

targets = torch.randint(
    0,
    config.vocab_size,
    (2, 128),
    device=device
)

print("Input:", x.shape)

logits, loss = model(
    x,
    targets
)

print("Logits:", logits.shape)
print("Loss:", loss.item())

loss.backward()

print("Backward: OK")

if device == "cuda":

    print(
        "VRAM:",
        round(
            torch.cuda.memory_allocated()
            / 1024**2,
            2
        ),
        "MB"
    )

print("=" * 50)
print("MODEL TEST PASSED")
print("=" * 50)