import json
from collections import defaultdict


FILE = "data/processed/medlineplus.jsonl"


documents_by_title = defaultdict(list)


with open(FILE, "r", encoding="utf-8") as f:

    for line in f:

        doc = json.loads(line)

        title = doc["title"].strip().lower()

        documents_by_title[title].append(doc)


duplicates = {
    title: docs
    for title, docs in documents_by_title.items()
    if len(docs) > 1
}


print(f"Duplicate titles: {len(duplicates)}")


for title, docs in list(duplicates.items())[:10]:

    print("\n" + "=" * 80)
    print("TITLE:", title)

    for doc in docs:

        print("\nID:", doc["id"])
        print("URL:", doc["url"])
        print("TEXT LENGTH:", len(doc["text"].split()))

        print(
            "GROUPS:",
            doc["metadata"].get("groups", [])
        )

        print(
            "ORGANIZATIONS:",
            doc["metadata"].get("organizations", [])
        )