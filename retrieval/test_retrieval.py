import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_DIR = os.path.join("data", "vector_store", "medical_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("=" * 70)
    print("TESTING FAISS RETRIEVAL")
    print("=" * 70)

    # 1. Load Embedding Model
    print("\n[1/2] Loading HuggingFace Embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Load FAISS Vector Store
    print(f"\n[2/2] Loading FAISS index from {VECTOR_STORE_DIR}...")
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("Vector index loaded successfully!")

    # 3. Test Queries
    test_queries = [
        "What are the symptoms of Type 2 Diabetes?",
        "How is hypertension treated?",
        "What causes abdominal pain after eating?"
    ]

    for query in test_queries:
        print("\n" + "-" * 70)
        print(f"QUERY: '{query}'")
        print("-" * 70)

        start_time = time.time()
        # Retrieve top 3 relevant chunks with L2 distance scores
        results = vector_store.similarity_search_with_score(query, k=3)
        latency = (time.time() - start_time) * 1000

        print(f"Retrieval latency: {latency:.2f} ms\n")

        for idx, (doc, score) in enumerate(results, 1):
            print(f"--- Result #{idx} (Score/Distance: {score:.4f}) ---")
            print(f"Title : {doc.metadata.get('title', 'N/A')}")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Text  : {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    main()