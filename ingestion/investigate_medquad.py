import json
from collections import defaultdict


FILE = "data/processed/medquad.jsonl"


documents = []

with open(FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            documents.append(json.loads(line))


# ============================================================
# 1. VERY SHORT ANSWERS
# ============================================================

print("=" * 70)
print("VERY SHORT MEDQUAD ANSWERS")
print("=" * 70)

short_docs = []

for doc in documents:
    text = doc.get("text", "").strip()
    words = len(text.split())

    if words <= 5:
        short_docs.append(doc)


print(f"\nDocuments with <= 5 words: {len(short_docs)}")


for doc in short_docs[:30]:

    print("\n" + "-" * 70)
    print(f"ID       : {doc.get('id')}")
    print(f"TITLE    : {doc.get('title')}")
    print(f"QUESTION : {doc.get('question')}")
    print(f"WORDS    : {len(doc.get('text', '').split())}")
    print(f"ANSWER   : {doc.get('text')}")


# ============================================================
# 2. DUPLICATE QUESTIONS
# ============================================================

question_groups = defaultdict(list)

for doc in documents:

    question = doc.get("question", "").strip().lower()

    question_groups[question].append(doc)


duplicate_questions = {
    question: docs
    for question, docs in question_groups.items()
    if len(docs) > 1
}


print("\n\n" + "=" * 70)
print("DUPLICATE QUESTION INVESTIGATION")
print("=" * 70)

print(f"\nDuplicate question groups: {len(duplicate_questions)}")


# ============================================================
# 3. SAME QUESTION / SAME ANSWER
# ============================================================

same_answer = 0
different_answer = 0


for question, docs in duplicate_questions.items():

    answers = set(
        doc.get("text", "").strip()
        for doc in docs
    )

    if len(answers) == 1:
        same_answer += 1
    else:
        different_answer += 1


print(f"\nSame question + same answer   : {same_answer}")
print(f"Same question + different answer: {different_answer}")


# ============================================================
# 4. EXAMPLES OF DIFFERENT ANSWERS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE QUESTIONS WITH DIFFERENT ANSWERS")
print("=" * 70)


shown = 0

for question, docs in duplicate_questions.items():

    answers = set(
        doc.get("text", "").strip()
        for doc in docs
    )

    if len(answers) > 1:

        print("\n" + "-" * 70)
        print(f"QUESTION: {question}")

        for doc in docs:

            print(f"\nID: {doc.get('id')}")
            print(f"TITLE: {doc.get('title')}")
            print(f"SOURCE: {doc.get('metadata', {}).get('source_name')}")
            print(f"URL: {doc.get('url')}")
            print(f"ANSWER: {doc.get('text')[:500]}")

        shown += 1

        if shown >= 10:
            break


# ============================================================
# 5. SAME QUESTION + DIFFERENT SOURCES
# ============================================================

print("\n" + "=" * 70)
print("SOURCE OVERLAP")
print("=" * 70)


source_overlap_examples = 0

for question, docs in duplicate_questions.items():

    sources = set(
        doc.get("metadata", {}).get("source_name")
        for doc in docs
    )

    if len(sources) > 1:

        print("\n" + "-" * 70)
        print(f"QUESTION: {question}")
        print(f"SOURCES : {', '.join(str(s) for s in sources)}")

        for doc in docs:
            print(
                f"  {doc.get('id')} -> "
                f"{doc.get('metadata', {}).get('source_name')}"
            )

        source_overlap_examples += 1

        if source_overlap_examples >= 10:
            break


print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)