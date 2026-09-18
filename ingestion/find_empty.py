import json

FILE = "data/processed/medlineplus.jsonl"

with open(FILE, "r", encoding="utf-8") as f:

    for line in f:

        doc = json.loads(line)

        if not doc["text"].strip():

            print("Empty document found")
            print("=" * 60)
            print("ID:", doc["id"])
            print("TITLE:", doc["title"])
            print("URL:", doc["url"])
            print("METADATA:", doc["metadata"])