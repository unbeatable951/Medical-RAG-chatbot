import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/medquad.jsonl")
OUTPUT_FILE = Path("data/processed/medquad_clean.jsonl")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate detection.
    """

    if not text:
        return ""

    return " ".join(text.lower().split()).strip()


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_medquad():

    print("=" * 70)
    print("MEDQUAD DEDUPLICATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    seen = set()

    total_records = 0
    unique_records = 0
    duplicate_records = 0

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in input_file:

            line = line.strip()

            if not line:
                continue

            total_records += 1

            record = json.loads(line)

            question = normalize_text(
                record.get("question", "")
            )

            text = normalize_text(
                record.get("text", "")
            )

            # ----------------------------------------------
            # Question + text combination
            # ----------------------------------------------

            content_key = (
                question,
                text
            )

            # ----------------------------------------------
            # Duplicate
            # ----------------------------------------------

            if content_key in seen:

                duplicate_records += 1
                continue

            # ----------------------------------------------
            # Unique record
            # ----------------------------------------------

            seen.add(content_key)

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            unique_records += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("DEDUPLICATION RESULTS")
    print("=" * 70)

    print(f"\nOriginal records   : {total_records}")
    print(f"Unique records     : {unique_records}")
    print(f"Duplicates removed : {duplicate_records}")

    print(f"\nOutput:")
    print(f"  {OUTPUT_FILE}")

    print("\n" + "=" * 70)

    if duplicate_records == 0:
        print("ℹ️ No duplicates found.")
    else:
        print(
            f"✅ Removed {duplicate_records} "
            f"duplicate records."
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    deduplicate_medquad()