from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "clean" / "uzbek_clean.jsonl"
OUTPUT_DIR = ROOT / "tokenizer"
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 16_000

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
]


def text_iterator():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = item.get("text", "").strip()

            if text:
                yield text


def main():
    print("=" * 60)
    print("MirazizLM Tokenizer Training")
    print("=" * 60)

    tokenizer = Tokenizer(
        BPE(
            unk_token="<|unk|>"
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    print(f"Vocabulary target: {VOCAB_SIZE:,}")
    print("Training boshlandi...")

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
    )

    tokenizer.save(str(OUTPUT_FILE))

    print()
    print("=" * 60)
    print("TOKENIZER READY")
    print("=" * 60)

    print(f"Vocabulary: {tokenizer.get_vocab_size():,}")
    print(f"File: {OUTPUT_FILE}")

    test_sentences = [
        "Assalomu alaykum!",
        "O'zbekiston poytaxti Toshkent shahri.",
        "Sun'iy intellekt dasturlashda yordam beradi.",
        "Men Python va Django dasturlashni o'rganaman.",
    ]

    print()
    print("TEST:")
    print("-" * 60)

    for text in test_sentences:
        encoded = tokenizer.encode(text)

        print()
        print("Text:")
        print(text)

        print("Tokens:")
        print(encoded.tokens)

        print("IDs:")
        print(encoded.ids)

        print("Decoded:")
        print(tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    main()