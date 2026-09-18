import json
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/medquad.jsonl")


# ============================================================
# VALIDATION
# ============================================================

def validate_medquad(file_path):
    print("=" * 70)
    print("MEDQUAD DATASET VALIDATION")
    print("=" * 70)

    if not file_path.exists():
        print(f"\n❌ File not found: {file_path}")
        return

    total_records = 0
    valid_records = 0
    invalid_records = 0

    missing_fields = Counter()

    duplicate_ids = []
    duplicate_qa = []

    seen_ids = set()
    seen_qa = set()

    empty_questions = 0
    empty_text = 0
    empty_titles = 0

    required_fields = {
        "id",
        "source",
        "title",
        "question",
        "text",
        "url",
        "metadata"
    }

    # --------------------------------------------------------
    # Read JSONL
    # --------------------------------------------------------

    with open(file_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            total_records += 1

            # ------------------------------------------------
            # JSON validation
            # ------------------------------------------------

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:
                invalid_records += 1

                print(
                    f"❌ Invalid JSON at line {line_number}: {e}"
                )

                continue

            # ------------------------------------------------
            # Check required fields
            # ------------------------------------------------

            missing = required_fields - record.keys()

            if missing:
                invalid_records += 1

                for field in missing:
                    missing_fields[field] += 1

                # Don't print thousands of identical errors
                if invalid_records <= 10:
                    print(
                        f"⚠️ Line {line_number}: "
                        f"Missing fields: {', '.join(sorted(missing))}"
                    )

                continue

            # ------------------------------------------------
            # Check question
            # ------------------------------------------------

            question = str(record["question"]).strip()

            if not question:
                empty_questions += 1
                invalid_records += 1

                if empty_questions <= 10:
                    print(
                        f"⚠️ Line {line_number}: Empty question"
                    )

                continue

            # ------------------------------------------------
            # Check text
            # ------------------------------------------------

            text = str(record["text"]).strip()

            if not text:
                empty_text += 1
                invalid_records += 1

                if empty_text <= 10:
                    print(
                        f"⚠️ Line {line_number}: Empty text"
                    )

                continue

            # ------------------------------------------------
            # Check title
            # ------------------------------------------------

            title = str(record["title"]).strip()

            if not title:
                empty_titles += 1
                invalid_records += 1

                if empty_titles <= 10:
                    print(
                        f"⚠️ Line {line_number}: Empty title"
                    )

                continue

            # ------------------------------------------------
            # Duplicate ID
            # ------------------------------------------------

            record_id = str(record["id"]).strip()

            if record_id in seen_ids:
                duplicate_ids.append(record_id)

            else:
                seen_ids.add(record_id)

            # ------------------------------------------------
            # Duplicate question-text pair
            # ------------------------------------------------

            qa_pair = (
                question.lower(),
                text.lower()
            )

            if qa_pair in seen_qa:
                duplicate_qa.append(record_id)

            else:
                seen_qa.add(qa_pair)

            # ------------------------------------------------
            # Valid record
            # ------------------------------------------------

            valid_records += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nTotal records       : {total_records}")
    print(f"Valid records       : {valid_records}")
    print(f"Invalid records     : {invalid_records}")

    print(f"\nEmpty questions     : {empty_questions}")
    print(f"Empty text         : {empty_text}")
    print(f"Empty titles       : {empty_titles}")

    print(f"\nDuplicate IDs       : {len(duplicate_ids)}")
    print(f"Duplicate Q&A pairs : {len(duplicate_qa)}")

    # --------------------------------------------------------
    # Missing fields
    # --------------------------------------------------------

    if missing_fields:
        print("\nMissing fields:")

        for field, count in missing_fields.items():
            print(f"  - {field}: {count}")

    else:
        print("\nMissing fields      : None")

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    if duplicate_ids:
        print("\nDuplicate IDs (first 10):")

        for record_id in duplicate_ids[:10]:
            print(f"  - {record_id}")

    # --------------------------------------------------------
    # Duplicate Q&A
    # --------------------------------------------------------

    if duplicate_qa:
        print("\nDuplicate Q&A pairs (first 10):")

        for record_id in duplicate_qa[:10]:
            print(f"  - {record_id}")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)

    if invalid_records == 0:
        print("✅ DATASET STRUCTURE LOOKS GOOD")
    else:
        print("⚠️ DATASET HAS VALIDATION ISSUES")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    validate_medquad(INPUT_FILE)