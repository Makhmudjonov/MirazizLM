from datasets import load_dataset
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "raw" / "uz"
OUTPUT.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT / "uzbek.jsonl"

MAX_DOCUMENTS = 20_000


def clean_text(text):
    if not text:
        return None

    text = text.strip()

    # Juda kichik parchalarni tashlaymiz
    if len(text) < 200:
        return None

    return text


print("Uzbek dataset yuklanmoqda...")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.uz",
    split="train",
    streaming=True,
)

saved = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as f:

    for item in dataset:

        text = clean_text(item.get("text"))

        if not text:
            continue

        record = {
            "text": text,
            "language": "uz",
            "source": "wikipedia"
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

        saved += 1

        if saved % 1000 == 0:
            print(f"{saved:,} document saqlandi")

        if saved >= MAX_DOCUMENTS:
            break


print()
print("TAYYOR!")
print(f"Documents: {saved:,}")
print(f"File: {OUTPUT_FILE}")