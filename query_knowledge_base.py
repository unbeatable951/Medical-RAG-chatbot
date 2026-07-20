#!/usr/bin/env python3
"""
query_knowledge_base.py  (bonus / optional)

Quick sanity-check tool for the FAISS index built by prepare_knowledge_base.py.
This performs retrieval only (no LLM call) so you can verify the knowledge
base is working before wiring it into a chatbot.

Usage
-----
    python query_knowledge_base.py "what are the symptoms of type 2 diabetes"
    python query_knowledge_base.py "hypertension treatment" --top-k 3
"""

import argparse
import pickle

import faiss
from sentence_transformers import SentenceTransformer

import config


def main():
    parser = argparse.ArgumentParser(description="Query the medical RAG FAISS index.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL_NAME)
    args = parser.parse_args()

    print(f"Loading index from {config.FAISS_INDEX_PATH} ...")
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer(args.embedding_model)
    query_vec = model.encode([args.query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(query_vec, args.top_k)

    print(f"\nTop {args.top_k} results for: {args.query!r}\n" + "=" * 60)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        chunk = chunks[idx]
        print(f"\n[{rank}] score={score:.4f} source={chunk['source']} "
              f"meta={chunk['metadata']}")
        print(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))


if __name__ == "__main__":
    main()
