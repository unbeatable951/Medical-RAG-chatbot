import json
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MEDQUAD_FILE = PROCESSED_DIR / "medquad_clean.jsonl"
MEDLINEPLUS_FILE = PROCESSED_DIR / "medlineplus.jsonl"
CDC_FILE = PROCESSED_DIR / "cdc.jsonl"

OUTPUT_FILE = PROCESSED_DIR / "combined_medical.jsonl"


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(file_path):

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                records.append(record)

            except json.JSONDecodeError as e:

                print(
                    f"WARNING: Invalid JSON in "
                    f"{file_path.name} "
                    f"at line {line_number}: {e}"
                )

    return records


# ============================================================
# VALIDATE RECORD
# ============================================================

def is_valid_record(record):

    required_fields = [
        "id",
        "source",
        "title",
        "text"
    ]

    for field in required_fields:

        if field not in record:
            return False

        if not str(record[field]).strip():
            return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COMBINING MEDICAL DATASETS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    input_files = [
        MEDQUAD_FILE,
        MEDLINEPLUS_FILE,
        CDC_FILE
    ]

    print("\nChecking input files...")

    for file_path in input_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        print(
            f"✓ {file_path.name}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    medquad = load_jsonl(
        MEDQUAD_FILE
    )

    medlineplus = load_jsonl(
        MEDLINEPLUS_FILE
    )

    cdc = load_jsonl(
        CDC_FILE
    )

    print(
        f"MedQuAD      : {len(medquad)} records"
    )

    print(
        f"MedlinePlus  : {len(medlineplus)} records"
    )

    print(
        f"CDC          : {len(cdc)} records"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_records = (
        medquad
        + medlineplus
        + cdc
    )

    print(
        f"\nRecords before validation: "
        f"{len(all_records)}"
    )

    # --------------------------------------------------------
    # Validate records
    # --------------------------------------------------------

    valid_records = []
    invalid_records = []

    for record in all_records:

        if is_valid_record(record):

            valid_records.append(record)

        else:

            invalid_records.append(record)

    print(
        f"Valid records             : "
        f"{len(valid_records)}"
    )

    print(
        f"Invalid records           : "
        f"{len(invalid_records)}"
    )

    # --------------------------------------------------------
    # Global ID deduplication
    # --------------------------------------------------------

    seen_ids = set()

    unique_records = []
    duplicate_id_count = 0

    for record in valid_records:

        record_id = record["id"]

        if record_id in seen_ids:

            duplicate_id_count += 1

            continue

        seen_ids.add(record_id)

        unique_records.append(record)

    # --------------------------------------------------------
    # Q&A / content deduplication
    # --------------------------------------------------------

    seen_content = set()

    final_records = []
    duplicate_content_count = 0

    for record in unique_records:

        question = record.get(
            "question",
            ""
        ).strip().lower()

        text = record.get(
            "text",
            ""
        ).strip().lower()

        # For MedlinePlus and CDC, question may not exist.
        # Therefore use title + text when question is empty.

        if question:

            content_key = (
                question,
                text
            )

        else:

            content_key = (
                record.get(
                    "title",
                    ""
                ).strip().lower(),
                text
            )

        if content_key in seen_content:

            duplicate_content_count += 1

            continue

        seen_content.add(content_key)

        final_records.append(record)

    # --------------------------------------------------------
    # Source statistics
    # --------------------------------------------------------

    source_counts = Counter(
        record.get(
            "source",
            "unknown"
        )
        for record in final_records
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for record in final_records:

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("COMBINATION RESULTS")
    print("=" * 70)

    print(
        f"Original records       : "
        f"{len(all_records)}"
    )

    print(
        f"Invalid records        : "
        f"{len(invalid_records)}"
    )

    print(
        f"Duplicate IDs removed  : "
        f"{duplicate_id_count}"
    )

    print(
        f"Duplicate content      : "
        f"{duplicate_content_count}"
    )

    print(
        f"Final records          : "
        f"{len(final_records)}"
    )

    print("\nRecords by source:")

    for source, count in source_counts.items():

        print(
            f"  {source:<15}: {count}"
        )

    print(
        f"\nOutput:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)

    if invalid_records == []:
        print(
            "✅ ALL RECORDS PASSED STRUCTURAL VALIDATION"
        )

    if duplicate_id_count == 0:
        print(
            "✅ NO DUPLICATE IDs"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()