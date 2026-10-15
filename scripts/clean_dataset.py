import json
import re
import unicodedata
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "raw" / "uz" / "uzbek.jsonl"
OUTPUT_DIR = ROOT / "data" / "clean"
OUTPUT_FILE = OUTPUT_DIR / "uzbek_clean.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Non-breaking space -> normal space
    text = text.replace("\u00a0", " ")

    # Zero-width belgilar
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Windows newline
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Space/tablarni normallashtirish
    text = re.sub(r"[ \t]+", " ", text)

    # Har bir qatordagi ortiqcha space
    lines = [line.strip() for line in text.split("\n")]

    # Juda ko'p bo'sh qatorlarni kamaytirish
    cleaned_lines = []
    previous_empty = False

    for line in lines:
        if not line:
            if not previous_empty:
                cleaned_lines.append("")
            previous_empty = True
        else:
            cleaned_lines.append(line)
            previous_empty = False

    text = "\n".join(cleaned_lines)

    return text.strip()


def is_good_document(text: str) -> bool:

    # Juda qisqa
    if len(text) < 200:
        return False

    # Harflar yetarli bo'lishi kerak
    letters = sum(c.isalpha() for c in text)

    if letters / max(len(text), 1) < 0.50:
        return False

    # Haddan tashqari URL
    urls = re.findall(r"https?://\S+|www\.\S+", text)

    if len(urls) > 20:
        return False

    # Juda uzun bir xil belgilar
    if re.search(r"(.)\1{20,}", text):
        return False

    return True


def document_hash(text: str) -> str:

    normalized = re.sub(r"\s+", " ", text).strip().lower()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def main():

    print("=" * 60)
    print("MirazizLM Dataset Cleaner")
    print("=" * 60)

    total = 0
    saved = 0
    duplicates = 0
    rejected = 0

    seen_hashes = set()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as input_file, OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in input_file:

            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue

            text = item.get("text", "")

            text = normalize_text(text)

            if not is_good_document(text):
                rejected += 1
                continue

            text_hash = document_hash(text)

            if text_hash in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(text_hash)

            result = {
                "text": text,
                "language": item.get("language", "uz"),
                "source": item.get("source", "unknown")
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            saved += 1

            if total % 1000 == 0:
                print(
                    f"Processed: {total:,} | "
                    f"Saved: {saved:,} | "
                    f"Rejected: {rejected:,} | "
                    f"Duplicates: {duplicates:,}"
                )

    print()
    print("=" * 60)
    print("CLEANING FINISHED")
    print("=" * 60)

    print(f"Total:      {total:,}")
    print(f"Saved:      {saved:,}")
    print(f"Rejected:   {rejected:,}")
    print(f"Duplicates: {duplicates:,}")

    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024

    print(f"Size:       {size_mb:.2f} MB")
    print(f"Output:     {OUTPUT_FILE}")


if __name__ == "__main__":
    main()