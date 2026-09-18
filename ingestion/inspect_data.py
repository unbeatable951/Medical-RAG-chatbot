import json

FILE = "data/processed/medlineplus.jsonl"


with open(FILE, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        document = json.loads(line)

        print("=" * 80)
        print("ID:", document["id"])
        print("SOURCE:", document["source"])
        print("TITLE:", document["title"])
        print("URL:", document["url"])

        print("\nTEXT:")
        print(document["text"][:1000])

        print("\nMETADATA:")
        print(document["metadata"])

        if i == 2:
            break