import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/processed/all_documents.jsonl")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ============================================================
# CHUNKER
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        ""
    ],
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CHUNKING MEDICAL DOCUMENTS")
    print("=" * 70)

    documents = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    print(f"\nOriginal documents: {len(documents)}")

    chunks = []

    for document in documents:

        text = document["text"].strip()

        if not text:
            continue

        document_chunks = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(document_chunks):

            chunk = {
                "chunk_id": f"{document['id']}_chunk_{chunk_index}",
                "document_id": document["id"],
                "source": document["source"],
                "title": document["title"],
                "question": document.get("question", ""),
                "text": chunk_text,
                "url": document["url"],
                "metadata": document.get("metadata", {}),
                "chunk_index": chunk_index,
                "total_chunks": len(document_chunks)
            }

            chunks.append(chunk)

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    medlineplus_chunks = sum(
        1 for c in chunks
        if c["source"] == "medlineplus"
    )

    medquad_chunks = sum(
        1 for c in chunks
        if c["source"] == "medquad"
    )

    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    print(f"\nOutput: {OUTPUT_PATH}")

    print(f"Original documents : {len(documents)}")
    print(f"Total chunks       : {len(chunks)}")

    print(f"\nMedlinePlus chunks : {medlineplus_chunks}")
    print(f"MedQuAD chunks     : {medquad_chunks}")

    print(
        f"\nAverage chunks/document: "
        f"{len(chunks) / len(documents):.2f}"
    )

    print("\nChunk size:")
    print(f"  Target  : {CHUNK_SIZE}")
    print(f"  Overlap : {CHUNK_OVERLAP}")

    # Show examples
    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for chunk in chunks[:3]:

        print(f"\nChunk ID : {chunk['chunk_id']}")
        print(f"Source   : {chunk['source']}")
        print(f"Title    : {chunk['title']}")
        print(f"Index    : {chunk['chunk_index']}")
        print(f"Text     : {chunk['text'][:500]}...")


if __name__ == "__main__":
    main()