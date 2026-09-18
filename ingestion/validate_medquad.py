import json
from collections import Counter
from statistics import mean, median


# ============================================================
# CONFIGURATION
# ============================================================

FILE = "data/processed/medquad.jsonl"


# ============================================================
# LOAD DATA
# ============================================================

documents = []

with open(FILE, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Invalid JSON at line {line_number}: {e}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 60)
print("MEDQUAD DATA QUALITY REPORT")
print("=" * 60)

print(f"\nTotal documents: {len(documents)}")


# ============================================================
# MISSING FIELDS
# ============================================================

required_fields = [
    "id",
    "title",
    "question",
    "text",
    "url",
]

missing = {field: 0 for field in required_fields}

for doc in documents:
    for field in required_fields:
        value = doc.get(field)

        if value is None or str(value).strip() == "":
            missing[field] += 1


print("\nMissing fields:")

for field in required_fields:
    print(f"  {field.capitalize():<10}: {missing[field]}")


# ============================================================
# DUPLICATE IDs
# ============================================================

ids = [
    doc.get("id")
    for doc in documents
    if doc.get("id")
]

id_counts = Counter(ids)

duplicate_ids = {
    key: count
    for key, count in id_counts.items()
    if count > 1
}


print(f"\nDuplicate IDs: {len(duplicate_ids)}")

if duplicate_ids:
    print("\nExamples:")

    for key, count in list(duplicate_ids.items())[:10]:
        print(f"  {key}: {count}")


# ============================================================
# DUPLICATE QUESTIONS
# ============================================================

questions = [
    doc.get("question", "").strip().lower()
    for doc in documents
    if doc.get("question")
]

question_counts = Counter(questions)

duplicate_questions = {
    key: count
    for key, count in question_counts.items()
    if count > 1
}


print(f"\nDuplicate questions: {len(duplicate_questions)}")

if duplicate_questions:
    print("\nExamples:")

    for question, count in list(duplicate_questions.items())[:10]:
        print(f"  {question}: {count}")


# ============================================================
# EMPTY / VERY SHORT TEXT
# ============================================================

empty_text = []
short_text = []

for doc in documents:
    text = doc.get("text", "").strip()

    word_count = len(text.split())

    if word_count == 0:
        empty_text.append(doc)

    elif word_count < 20:
        short_text.append(doc)


print(f"\nEmpty documents: {len(empty_text)}")
print(f"Very short documents (<20 words): {len(short_text)}")


if empty_text:
    print("\nEmpty document examples:")

    for doc in empty_text[:5]:
        print(f"  ID: {doc.get('id')}")
        print(f"  TITLE: {doc.get('title')}")
        print(f"  QUESTION: {doc.get('question')}")
        print()


if short_text:
    print("\nShort document examples:")

    for doc in short_text[:5]:
        text = doc.get("text", "")

        print(f"  ID: {doc.get('id')}")
        print(f"  TITLE: {doc.get('title')}")
        print(f"  WORDS: {len(text.split())}")
        print(f"  TEXT: {text[:200]}")
        print()


# ============================================================
# TEXT STATISTICS
# ============================================================

word_counts = [
    len(doc.get("text", "").split())
    for doc in documents
    if doc.get("text", "").strip()
]


if word_counts:

    print("\nText statistics:")

    print(f"  Minimum words: {min(word_counts)}")
    print(f"  Maximum words: {max(word_counts)}")
    print(f"  Average words: {mean(word_counts):.2f}")
    print(f"  Median words:  {median(word_counts):.0f}")


# ============================================================
# QUESTION LENGTH STATISTICS
# ============================================================

question_lengths = [
    len(doc.get("question", "").split())
    for doc in documents
    if doc.get("question", "").strip()
]


if question_lengths:

    print("\nQuestion statistics:")

    print(f"  Minimum words: {min(question_lengths)}")
    print(f"  Maximum words: {max(question_lengths)}")
    print(f"  Average words: {mean(question_lengths):.2f}")
    print(f"  Median words:  {median(question_lengths):.0f}")


# ============================================================
# QTYPE DISTRIBUTION
# ============================================================

qtypes = []

for doc in documents:

    metadata = doc.get("metadata", {})

    qtype = metadata.get("qtype")

    if qtype:
        qtypes.append(qtype)


qtype_counts = Counter(qtypes)


print("\nQuestion types:")

for qtype, count in qtype_counts.most_common():
    print(f"  {qtype}: {count}")


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

sources = []

for doc in documents:

    metadata = doc.get("metadata", {})

    source_name = metadata.get("source_name")

    if source_name:
        sources.append(source_name)


source_counts = Counter(sources)


print("\nSource distribution:")

for source, count in source_counts.most_common():
    print(f"  {source}: {count}")


# ============================================================
# TITLE DISTRIBUTION
# ============================================================

titles = [
    doc.get("title")
    for doc in documents
    if doc.get("title")
]

unique_titles = set(titles)


print("\nTitle statistics:")

print(f"  Total titles:  {len(titles)}")
print(f"  Unique titles: {len(unique_titles)}")


# ============================================================
# URL VALIDATION
# ============================================================

invalid_urls = []

for doc in documents:

    url = doc.get("url", "")

    if not url.startswith(("http://", "https://")):
        invalid_urls.append(doc)


print(f"\nInvalid URLs: {len(invalid_urls)}")


if invalid_urls:

    print("\nInvalid URL examples:")

    for doc in invalid_urls[:5]:

        print(f"  ID: {doc.get('id')}")
        print(f"  URL: {doc.get('url')}")
        print()


# ============================================================
# METADATA VALIDATION
# ============================================================

missing_metadata = 0

for doc in documents:

    metadata = doc.get("metadata")

    if not isinstance(metadata, dict) or not metadata:
        missing_metadata += 1


print(f"Documents with missing metadata: {missing_metadata}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

problems = (
    sum(missing.values())
    + len(duplicate_ids)
    + len(empty_text)
    + len(invalid_urls)
    + missing_metadata
)

if problems == 0:

    print("No critical data-quality problems found.")

else:

    print(f"Potential issues found: {problems}")

print("=" * 60)