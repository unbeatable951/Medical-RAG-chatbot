"""
config.py
Central configuration for the Medical RAG chatbot project.
Edit the constants below to change scope/scale — no other files need touching.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MEDQUAD_DIR = os.path.join(RAW_DIR, "MedQuAD")
MEDLINEPLUS_DIR = os.path.join(RAW_DIR, "MedlinePlus")
WHO_DIR = os.path.join(RAW_DIR, "WHO")
CDC_DIR = os.path.join(RAW_DIR, "CDC")

ALL_RAW_DIRS = [MEDQUAD_DIR, MEDLINEPLUS_DIR, WHO_DIR, CDC_DIR]

# ---------------------------------------------------------------------------
# Dataset source configuration
# ---------------------------------------------------------------------------

# MedQuAD — cloned from the official GitHub repo (CC BY 4.0)
MEDQUAD_REPO_URL = "https://github.com/abachaa/MedQuAD.git"

# MedlinePlus — official XML files page (NLM ODbL-style open license)
MEDLINEPLUS_XML_INDEX_URL = "https://medlineplus.gov/xml.html"
MEDLINEPLUS_FALLBACK_TOPICS_URL = (
    "https://medlineplus.gov/xml/mplus_topics_compressed.xml"
)

# WHO IRIS (DSpace 7) REST API — CC BY-NC-SA 3.0 IGO
WHO_IRIS_SEARCH_API = "https://iris.who.int/server/api/discover/search/objects"
WHO_SEARCH_QUERIES = [
    "diabetes",
    "hypertension",
    "tuberculosis",
    "immunization",
    "malaria",
    "noncommunicable diseases",
    "mental health",
    "antimicrobial resistance",
]
WHO_MAX_DOCS_PER_QUERY = 5

# CDC — no unified bulk-text API for "Health Topics A-Z" narrative pages.
# We combine: (a) the official CDC Open Data (Socrata) API for structured
# datasets, and (b) a curated list of CDC "Health Topics" landing pages
# fetched as HTML (public, robots.txt-permitting, low request volume).
CDC_OPEN_DATA_API = "https://data.cdc.gov/resource"
CDC_OPEN_DATA_DATASETS = {
    # dataset_name: Socrata resource id (4x4 code)
    "chronic_disease_indicators": "g4ie-h725",
    "flu_vaccination_coverage": "vh55-3he6",
}
CDC_HEALTH_TOPIC_PAGES = [
    "https://www.cdc.gov/diabetes/about/index.html",
    "https://www.cdc.gov/high-blood-pressure/about/index.html",
    "https://www.cdc.gov/asthma/about/index.html",
    "https://www.cdc.gov/flu/index.html",
    "https://www.cdc.gov/heart-disease/about/index.html",
    "https://www.cdc.gov/cancer/index.html",
    "https://www.cdc.gov/mental-health/about/index.html",
    "https://www.cdc.gov/vaccines/index.html",
]

# ---------------------------------------------------------------------------
# Chunking / embedding / FAISS settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 120       # characters of overlap between consecutive chunks
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "medical_index.faiss")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_metadata.pkl")

# HTTP settings
REQUEST_TIMEOUT = 30
USER_AGENT = "medical-rag-chatbot-educational-project/1.0 (contact: set-your-email@example.com)"
REQUEST_DELAY_SECONDS = 0.4  # politeness delay between requests to public APIs


def ensure_directories():
    for d in ALL_RAW_DIRS + [PROCESSED_DIR, VECTOR_STORE_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
