import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MEDLINEPLUS_PATH = Path("data/processed/medlineplus.jsonl")
MEDQUAD_PATH = Path("data/processed/medquad.jsonl")

OUTPUT_PATH = Path("data/processed/all_documents.jsonl")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Remove HTML and normalize whitespace."""

    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# MEDLINEPLUS
# ============================================================

def load_medlineplus():
    documents = []

    with open(MEDLINEPLUS_PATH, "r", encoding="utf-8") as f:

        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            document = {
                "id": data["id"],
                "source": "medlineplus",
                "title": data["title"],
                "question": "",
                "text": clean_text(data["text"]),
                "url": data["url"],
                "metadata": data.get("metadata", {})
            }

            documents.append(document)

    return documents


# ============================================================
# MEDQUAD
# ============================================================

def load_medquad():
    documents = []

    with open(MEDQUAD_PATH, "r", encoding="utf-8") as f:

        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            document = {
                "id": data["id"],
                "source": "medquad",
                "title": data["title"],
                "question": data["question"],
                "text": clean_text(data["text"]),
                "url": data["url"],
                "metadata": data.get("metadata", {})
            }

            documents.append(document)

    return documents


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING UNIFIED MEDICAL DOCUMENT DATASET")
    print("=" * 70)

    print("\nLoading MedlinePlus...")
    medlineplus = load_medlineplus()
    print(f"MedlinePlus documents: {len(medlineplus)}")

    print("\nLoading MedQuAD...")
    medquad = load_medquad()
    print(f"MedQuAD documents: {len(medquad)}")

    # Combine
    documents = medlineplus + medquad

    print("\nTotal documents:")
    print(len(documents))

    # Check duplicate IDs
    ids = [doc["id"] for doc in documents]
    duplicate_ids = len(ids) - len(set(ids))

    print(f"Duplicate IDs: {duplicate_ids}")

    if duplicate_ids > 0:
        raise ValueError("Duplicate document IDs found!")

    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        for document in documents:
            f.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

    # Statistics
    medlineplus_count = sum(
        1 for doc in documents
        if doc["source"] == "medlineplus"
    )

    medquad_count = sum(
        1 for doc in documents
        if doc["source"] == "medquad"
    )

    print("\n" + "=" * 70)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"MedlinePlus: {medlineplus_count}")
    print(f"MedQuAD:     {medquad_count}")
    print(f"Total:       {len(documents)}")


if __name__ == "__main__":
    main()