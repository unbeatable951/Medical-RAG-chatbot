import json
import os
import time
import pickle
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -------------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------------
CHUNKS_FILE = os.path.join("data", "processed", "chunks.jsonl")
VECTOR_STORE_DIR = os.path.join("data", "vector_store")
FAISS_INDEX_NAME = "medical_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500  # Process in batches for progress tracking and memory safety

def main():
    print("=" * 70)
    print("BUILDING FAISS VECTOR STORE")
    print("=" * 70)

    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}. Run chunk_documents.py first!")

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # 1. Load Chunks into LangChain Document Objects
    print(f"\n[1/3] Loading chunks from {CHUNKS_FILE}...")
    documents = []
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Store text content along with chunk metadata
            doc = Document(
                page_content=item["text"],
                metadata={
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "title": item["title"],
                    "chunk_index": item["chunk_index"]
                }
            )
            documents.append(doc)

    total_docs = len(documents)
    print(f"Loaded {total_docs} document chunks successfully.")

    # 2. Initialize Embedding Model
    print(f"\n[2/3] Initializing HuggingFace Embeddings ({EMBEDDING_MODEL_NAME})...")
    # Setting model_kwargs to use CPU or GPU automatically
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU is available
        encode_kwargs={'normalize_embeddings': True}
    )

    # 3. Batch Embed & Index into FAISS
    print(f"\n[3/3] Generating embeddings & building FAISS index in batches of {BATCH_SIZE}...")
    start_time = time.time()
    
    # Process first batch to initialize FAISS index
    first_batch = documents[:BATCH_SIZE]
    print(f" -> Processing batch 1/{int(np.ceil(total_docs / BATCH_SIZE))} (chunks 0 to {len(first_batch)})...")
    vector_store = FAISS.from_documents(first_batch, embeddings)

    # Process remaining batches
    for i in range(BATCH_SIZE, total_docs, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = int(np.ceil(total_docs / BATCH_SIZE))
        print(f" -> Processing batch {batch_num}/{total_batches} (chunks {i} to {i + len(batch)})...")
        
        vector_store.add_documents(batch)

    elapsed_time = time.time() - start_time
    print(f"\nEmbedding & Indexing complete in {elapsed_time:.2f} seconds!")

    # 4. Save Vector Index to Disk
    save_path = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)
    vector_store.save_local(save_path)
    
    print("\n" + "=" * 70)
    print("SAVED VECTOR STORE ARTIFACTS")
    print("=" * 70)
    print(f"FAISS Index location : {save_path}")
    print(f"Total Vectors Indexed: {total_docs}")
    print("=" * 70)

if __name__ == "__main__":
    main()