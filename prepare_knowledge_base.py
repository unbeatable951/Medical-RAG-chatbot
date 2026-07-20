#!/usr/bin/env python3
"""
prepare_knowledge_base.py

Loads all downloaded raw datasets, cleans and deduplicates them, splits them
into overlapping chunks, embeds the chunks with a sentence-transformer model,
and builds a FAISS vector index for retrieval.

Pipeline
--------
1. Load        -> dataset-specific parsers turn raw files into Document dicts
                   {"text": ..., "source": ..., "metadata": {...}}
2. Clean       -> strip HTML/XML noise, normalize whitespace
3. Deduplicate -> drop exact and near-duplicate documents
4. Chunk       -> recursive character splitter (paragraph-aware, with overlap)
5. Embed       -> sentence-transformers (all-MiniLM-L6-v2, local, no API key)
6. Index       -> FAISS (cosine similarity via normalized inner product)
7. Save        -> data/vector_store/medical_index.faiss + chunk_metadata.pkl

Usage
-----
    python prepare_knowledge_base.py                  # full rebuild
    python prepare_knowledge_base.py --only medquad medlineplus
    python prepare_knowledge_base.py --chunk-size 500 --chunk-overlap 80
"""

import argparse
import hashlib
import html
import json
import logging
import os
import pickle
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.ensure_directories()
LOG_FILE = os.path.join(config.LOG_DIR, "prepare_knowledge_base.log")

logger = logging.getLogger("prepare_knowledge_base")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def doc_hash(text: str) -> str:
    normalized = re.sub(r"\W+", "", text.lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dataset loaders -> each yields dicts: {"text", "source", "metadata"}
# ---------------------------------------------------------------------------
def load_medquad():
    """Parse MedQuAD's per-collection XML QA files."""
    docs = []
    root_dir = config.MEDQUAD_DIR
    if not os.path.isdir(root_dir):
        logger.warning(f"MedQuAD directory not found: {root_dir}")
        return docs

    xml_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".xml"):
                xml_files.append(os.path.join(dirpath, fn))

    logger.info(f"MedQuAD: found {len(xml_files)} XML files")
    for path in tqdm(xml_files, desc="Loading MedQuAD"):
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            focus = root.findtext("Focus", default="")
            for qa_pair in root.findall(".//QAPair"):
                question = qa_pair.findtext("Question", default="")
                answer = qa_pair.findtext("Answer", default="")
                q_type = qa_pair.find("Question")
                qtype_attr = q_type.attrib.get("qtype", "") if q_type is not None else ""
                if not answer or not answer.strip():
                    continue
                text = f"Q: {question.strip()}\nA: {answer.strip()}"
                docs.append({
                    "text": clean_text(text),
                    "source": "MedQuAD",
                    "metadata": {
                        "focus": focus,
                        "question_type": qtype_attr,
                        "file": os.path.relpath(path, root_dir),
                    },
                })
        except ET.ParseError as exc:
            logger.warning(f"Failed to parse {path}: {exc}")
    logger.info(f"MedQuAD: loaded {len(docs)} QA documents")
    return docs


def load_medlineplus():
    """Parse the MedlinePlus compressed health topic XML file."""
    docs = []
    path = os.path.join(config.MEDLINEPLUS_DIR, "mplus_topics_compressed.xml")
    if not os.path.isfile(path):
        logger.warning(f"MedlinePlus file not found: {path}")
        return docs
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        topics = root.findall(".//health-topic")
        logger.info(f"MedlinePlus: found {len(topics)} health topics")
        for topic in tqdm(topics, desc="Loading MedlinePlus"):
            title = topic.attrib.get("title", "")
            summary_el = topic.find("full-summary")
            summary = "".join(summary_el.itertext()) if summary_el is not None else ""
            if not summary.strip():
                continue
            docs.append({
                "text": clean_text(f"{title}. {summary}"),
                "source": "MedlinePlus",
                "metadata": {"title": title, "url": topic.attrib.get("url", "")},
            })
    except ET.ParseError as exc:
        logger.error(f"Failed to parse MedlinePlus XML: {exc}")
    logger.info(f"MedlinePlus: loaded {len(docs)} documents")
    return docs


def load_pmc():
    """Extract title/abstract/body text from PMC OA JATS XML packages (.tar.gz)."""
    import tarfile

    docs = []
    pmc_dir = config.PMC_DIR
    if not os.path.isdir(pmc_dir):
        logger.warning(f"PMC directory not found: {pmc_dir}")
        return docs

    tar_files = [f for f in os.listdir(pmc_dir) if f.endswith(".tar.gz")]
    logger.info(f"PMC OA: found {len(tar_files)} article packages")
    for fn in tqdm(tar_files, desc="Loading PMC OA"):
        path = os.path.join(pmc_dir, fn)
        try:
            with tarfile.open(path, "r:gz") as tar:
                xml_member = next(
                    (m for m in tar.getmembers() if m.name.endswith(".nxml")), None
                )
                if xml_member is None:
                    continue
                f = tar.extractfile(xml_member)
                if f is None:
                    continue
                root = ET.fromstring(f.read())

                title_el = root.find(".//article-title")
                title = "".join(title_el.itertext()) if title_el is not None else ""

                abstract_parts = [
                    "".join(p.itertext()) for p in root.findall(".//abstract//p")
                ]
                body_parts = [
                    "".join(p.itertext()) for p in root.findall(".//body//p")
                ]

                full_text = " ".join([title] + abstract_parts + body_parts)
                cleaned = clean_text(full_text)
                if len(cleaned) < 200:
                    continue
                docs.append({
                    "text": cleaned,
                    "source": "PMC_OA",
                    "metadata": {"title": title.strip(), "file": fn},
                })
        except Exception as exc:
            logger.warning(f"Failed to parse PMC package {fn}: {exc}")
    logger.info(f"PMC OA: loaded {len(docs)} documents")
    return docs


def load_who():
    """Extract text from downloaded WHO PDF publications."""
    docs = []
    who_dir = config.WHO_DIR
    if not os.path.isdir(who_dir):
        logger.warning(f"WHO directory not found: {who_dir}")
        return docs

    pdf_files = [f for f in os.listdir(who_dir) if f.lower().endswith(".pdf")]
    logger.info(f"WHO: found {len(pdf_files)} PDF publications")
    if not pdf_files:
        return docs

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf is not installed (pip install pypdf). Skipping WHO PDFs.")
        return docs

    for fn in tqdm(pdf_files, desc="Loading WHO PDFs"):
        path = os.path.join(who_dir, fn)
        try:
            reader = PdfReader(path)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            cleaned = clean_text(text)
            if len(cleaned) < 200:
                continue
            docs.append({
                "text": cleaned,
                "source": "WHO",
                "metadata": {"title": os.path.splitext(fn)[0].replace("_", " "), "file": fn},
            })
        except Exception as exc:
            logger.warning(f"Failed to extract text from {fn}: {exc}")
    logger.info(f"WHO: loaded {len(docs)} documents")
    return docs


def load_cdc():
    """Parse CDC health-topic HTML pages and the structured Socrata JSON datasets."""
    docs = []
    cdc_dir = config.CDC_DIR
    if not os.path.isdir(cdc_dir):
        logger.warning(f"CDC directory not found: {cdc_dir}")
        return docs

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 is not installed (pip install beautifulsoup4). Skipping CDC HTML.")
        BeautifulSoup = None

    files = os.listdir(cdc_dir)
    html_files = [f for f in files if f.endswith(".html")]
    json_files = [f for f in files if f.endswith(".json")]

    if BeautifulSoup:
        logger.info(f"CDC: found {len(html_files)} HTML topic pages")
        for fn in tqdm(html_files, desc="Loading CDC HTML"):
            path = os.path.join(cdc_dir, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                main = soup.find("main") or soup
                text = main.get_text(separator=" ")
                cleaned = clean_text(text)
                if len(cleaned) < 200:
                    continue
                title = soup.title.get_text().strip() if soup.title else fn
                docs.append({
                    "text": cleaned,
                    "source": "CDC",
                    "metadata": {"title": title, "file": fn},
                })
            except Exception as exc:
                logger.warning(f"Failed to parse {fn}: {exc}")

    logger.info(f"CDC: found {len(json_files)} structured Socrata datasets")
    for fn in json_files:
        path = os.path.join(cdc_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            # Summarize structured rows into short readable text snippets
            # rather than indexing raw JSON.
            for row in records[:500]:
                summary = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
                cleaned = clean_text(summary)
                if len(cleaned) < 40:
                    continue
                docs.append({
                    "text": cleaned,
                    "source": "CDC",
                    "metadata": {"dataset": fn},
                })
        except Exception as exc:
            logger.warning(f"Failed to parse {fn}: {exc}")

    logger.info(f"CDC: loaded {len(docs)} documents")
    return docs


LOADERS = {
    "medquad": load_medquad,
    "medlineplus": load_medlineplus,
    "pmc": load_pmc,
    "who": load_who,
    "cdc": load_cdc,
}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(docs):
    seen = set()
    unique = []
    for d in docs:
        h = doc_hash(d["text"])
        if h in seen:
            continue
        seen.add(h)
        unique.append(d)
    logger.info(f"Deduplication: {len(docs)} -> {len(unique)} documents")
    return unique


# ---------------------------------------------------------------------------
# Chunking (paragraph-aware recursive splitter, no external dependency)
# ---------------------------------------------------------------------------
def split_text(text: str, chunk_size: int, chunk_overlap: int):
    """Greedy paragraph/sentence packing splitter with character overlap."""
    if len(text) <= chunk_size:
        return [text]

    # Prefer splitting on paragraph/sentence boundaries where possible.
    units = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # start next chunk with overlap from the tail of the previous one
            overlap_text = current[-chunk_overlap:] if current else ""
            current = f"{overlap_text} {unit}".strip()
            # if a single unit is longer than chunk_size, hard-split it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - chunk_overlap:]
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs, chunk_size, chunk_overlap):
    chunks = []
    for d in tqdm(docs, desc="Chunking documents"):
        for i, piece in enumerate(split_text(d["text"], chunk_size, chunk_overlap)):
            piece = piece.strip()
            if len(piece) < 30:
                continue
            chunks.append({
                "text": piece,
                "source": d["source"],
                "metadata": {**d["metadata"], "chunk_index": i},
            })
    logger.info(f"Chunking: {len(docs)} documents -> {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Embedding + FAISS index
# ---------------------------------------------------------------------------
def build_index(chunks, model_name, batch_size):
    from sentence_transformers import SentenceTransformer
    import faiss

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks (batch size {batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def save_index(index, chunks):
    import faiss

    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved FAISS index -> {config.FAISS_INDEX_PATH}")
    logger.info(f"Saved chunk metadata -> {config.METADATA_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build the medical RAG knowledge base.")
    parser.add_argument("--only", nargs="+", choices=LOADERS.keys(),
                         help="Only process these datasets.")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE)
    args = parser.parse_args()

    config.ensure_directories()
    targets = args.only if args.only else list(LOADERS.keys())

    logger.info(f"Loading datasets: {targets}")
    all_docs = []
    for name in targets:
        try:
            all_docs.extend(LOADERS[name]())
        except Exception as exc:
            logger.error(f"UNEXPECTED ERROR loading {name}: {exc}")

    if not all_docs:
        logger.error(
            "No documents were loaded. Did you run download_datasets.py first? Aborting."
        )
        sys.exit(1)

    logger.info(f"Total raw documents loaded: {len(all_docs)}")
    all_docs = deduplicate(all_docs)
    chunks = chunk_documents(all_docs, args.chunk_size, args.chunk_overlap)

    if not chunks:
        logger.error("No chunks produced after cleaning/splitting. Aborting.")
        sys.exit(1)

    index, _ = build_index(chunks, args.embedding_model, args.batch_size)
    save_index(index, chunks)

    logger.info("=== Knowledge base build complete ===")
    logger.info(f"Documents: {len(all_docs)} | Chunks: {len(chunks)} | Vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
