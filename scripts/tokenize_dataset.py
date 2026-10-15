import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "clean" / "uzbek_clean.jsonl"
TOKENIZER_FILE = ROOT / "tokenizer" / "tokenizer.json"
OUTPUT_DIR = ROOT / "data" / "tokenized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.bin"
VAL_FILE = OUTPUT_DIR / "val.bin"

VAL_RATIO = 0.02

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

eos_id = tokenizer.token_to_id("<|eos|>")

print("=" * 60)
print("MirazizLM Dataset Tokenizer")
print("=" * 60)

documents = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = item.get("text", "").strip()

        if text:
            documents.append(text)

print(f"Documents: {len(documents):,}")

split_index = int(
    len(documents) * (1 - VAL_RATIO)
)

train_documents = documents[:split_index]
val_documents = documents[split_index:]


def encode_documents(docs, output_file):

    total_tokens = 0

    with output_file.open("wb") as f:

        for i, text in enumerate(docs):

            ids = tokenizer.encode(text).ids

            # Har bir document oxiriga EOS
            ids.append(eos_id)

            # vocab 16k bo'lgani uchun uint16 yetadi
            array = np.asarray(
                ids,
                dtype=np.uint16
            )

            array.tofile(f)

            total_tokens += len(ids)

            if (i + 1) % 1000 == 0:
                print(
                    f"{i + 1:,} documents | "
                    f"{total_tokens:,} tokens"
                )

    return total_tokens


print()
print("TRAIN tokenization...")
train_tokens = encode_documents(
    train_documents,
    TRAIN_FILE
)

print()
print("VALIDATION tokenization...")
val_tokens = encode_documents(
    val_documents,
    VAL_FILE
)

print()
print("=" * 60)
print("TOKENIZATION FINISHED")
print("=" * 60)

print(f"Train documents: {len(train_documents):,}")
print(f"Val documents:   {len(val_documents):,}")

print(f"Train tokens: {train_tokens:,}")
print(f"Val tokens:   {val_tokens:,}")

print(
    f"Train size: "
    f"{TRAIN_FILE.stat().st_size / 1024 / 1024:.2f} MB"
)

print(
    f"Val size: "
    f"{VAL_FILE.stat().st_size / 1024 / 1024:.2f} MB"
)