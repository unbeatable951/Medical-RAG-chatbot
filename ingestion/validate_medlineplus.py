import json
from collections import Counter
from pathlib import Path


FILE = Path("data/processed/medlineplus.jsonl")


def validate():

    documents = []

    with open(FILE, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    print("=" * 60)
    print("MEDLINEPLUS DATA QUALITY REPORT")
    print("=" * 60)

    total = len(documents)

    print(f"\nTotal documents: {total}")

    # --------------------------------------------------
    # Missing fields
    # --------------------------------------------------

    missing_title = sum(
        not doc.get("title", "").strip()
        for doc in documents
    )

    missing_text = sum(
        not doc.get("text", "").strip()
        for doc in documents
    )

    missing_url = sum(
        not doc.get("url", "").strip()
        for doc in documents
    )

    print("\nMissing fields:")
    print(f"  Title: {missing_title}")
    print(f"  Text:  {missing_text}")
    print(f"  URL:   {missing_url}")

    # --------------------------------------------------
    # Duplicate titles
    # --------------------------------------------------

    titles = [
        doc["title"].strip().lower()
        for doc in documents
        if doc.get("title")
    ]

    title_counts = Counter(titles)

    duplicates = {
        title: count
        for title, count in title_counts.items()
        if count > 1
    }

    print(f"\nDuplicate titles: {len(duplicates)}")

    if duplicates:
        print("\nExamples:")

        for title, count in list(duplicates.items())[:10]:
            print(f"  {title}: {count}")

    # --------------------------------------------------
    # Text length
    # --------------------------------------------------

    lengths = [
        len(doc.get("text", "").split())
        for doc in documents
    ]

    if lengths:

        print("\nText statistics:")

        print(f"  Minimum words: {min(lengths)}")
        print(f"  Maximum words: {max(lengths)}")
        print(
            f"  Average words: {sum(lengths) / len(lengths):.2f}"
        )

        sorted_lengths = sorted(lengths)

        median = sorted_lengths[len(sorted_lengths) // 2]

        print(f"  Median words:  {median}")

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    sources = Counter(
        doc.get("source", "")
        for doc in documents
    )

    print("\nSources:")

    for source, count in sources.items():
        print(f"  {source}: {count}")

    # --------------------------------------------------
    # Groups
    # --------------------------------------------------

    groups = Counter()

    for doc in documents:

        for group in doc.get("metadata", {}).get("groups", []):
            groups[group] += 1

    print("\nTop groups:")

    for group, count in groups.most_common(15):
        print(f"  {group}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    validate()