# 🩺 End-to-End Medical RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot built from trusted medical knowledge bases (**MedQuAD** & **MedlinePlus**). The pipeline leverages **FAISS** for fast local vector retrieval, **Groq** for high-performance LLM generation, and an interactive **Streamlit** chat interface with explicit source citation tracking.

---

## 🏗️ Architecture & Pipeline Overview

1. **Data Ingestion & Preprocessing:** Ingests medical document corpora, cleaning HTML/XML markup and chunking text semantically into 37,569 indexed passages.
2. **Dense Vector Indexing:** Generates 384-dimensional dense embeddings using `sentence-transformers/all-MiniLM-L6-v2` and persists a local FAISS vector store.
3. **Retrieval Chain:** Queries the local FAISS index with sub-100ms latency to fetch top-k relevant context passages.
4. **Context-Grounded Generation:** Formulates structured system prompts and streams medical answers using Groq-hosted open LLMs (Llama-3 / GPT-OSS models).
5. **Interactive Interface:** Renders responses alongside transparent, expandable citation accordions displaying similarity scores and source metadata.

---

## 📂 Project Structure

* **`app.py`**: Streamlit interactive web interface
* **`config.py`**: Centralized configuration and path management
* **`requirements.txt`**: Python dependency specifications
* **`.gitignore`**: Git exclusions for environment and cache files
* **`ingestion/`**: Data processing pipeline
  * **`chunk_documents.py`**: Cleans and chunks raw medical XML/JSON files
  * **`build_vector_store.py`**: Generates embeddings and saves FAISS index
* **`retrieval/`**: Retrieval modules
  * **`test_retrieval.py`**: Terminal sanity-check script for similarity search
* **`generation/`**: LLM generation modules
  * **`rag_chain.py`**: Medical RAG pipeline with Groq client & fallback models
* **`data/vector_store/`**: Persisted local FAISS index and metadata
  * **`medical_index/`**: Contains `index.faiss` and `index.pkl`

---

## 🛠️ Tech Stack & Key Technologies

* **Language & Frameworks:** Python 3.9+, Streamlit
* **Orchestration:** LangChain (`langchain-community`, `langchain-huggingface`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace Hub)
* **Inference Engine:** Groq API (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-20b`)
* **Environment & Config:** `python-dotenv`

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone [https://github.com/unbeatable951/Medical-RAG-chatbot.git](https://github.com/unbeatable951/Medical-RAG-chatbot.git)
cd Medical-RAG-chatbot

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## Configure Environment Variables

Create a .env file in the project root folder to store your Groq **API** key:

Code snippet
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
Note: Get a free **API** key at Groq Console.

## Verify Pipeline Components

Run a terminal test to confirm vector retrieval and generation pipeline execution before launching the UI:

Bash python generation/rag_chain.py

 ## Launch the Web Application Start the Streamlit interface locally:

Bash streamlit run app.py
 Open your browser at [http://localhost:**8501**](http://localhost:**8501**)
 to query the chatbot.
