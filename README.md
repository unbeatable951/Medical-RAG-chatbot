# Medical RAG Chatbot — Knowledge Base

A Retrieval-Augmented-Generation knowledge base built from trusted, publicly
available medical sources, with automated download and indexing scripts.

---

## 1. Dataset comparison

| Dataset | Trust | Preprocessing | Resume value | Diseases | Medicines | Symptoms | Lab tests | Prevention | License |
|---|---|---|---|---|---|---|---|---|---|
| **MedQuAD** | High (NIH-derived) | Easy (clean XML QA pairs) | Medium | Strong | Medium | Strong | Weak | Weak | CC BY 4.0 (fully open) |
| **MedlinePlus** | High (NLM/NIH) | Easy (structured XML) | Medium | Strong | Strong (drug pages) | Strong | Medium | Strong | NLM open/ODbL-style, free reuse w/ attribution |
| **PMC OA Subset** | High (peer-reviewed) | Hard (JATS XML, variable structure) | High (shows real IR/NLP engineering) | Strong (deep, technical) | Strong | Medium | Strong | Medium | Mixed CC licenses, filterable to commercial-use-allowed |
| **WHO Publications** | Very high (UN agency) | Medium (PDF extraction) | High | Strong (global health) | Medium | Medium | Weak | Very strong | CC BY-NC-SA 3.0 IGO (non-commercial) |
| **CDC Health Topics** | Very high (US federal) | Hard (no bulk text API; HTML scraping) | Medium | Strong (US public health) | Medium | Strong | Medium | Very strong | Public domain (US govt work), CDC Open Data Socrata API for structured data |
| NIH Health Topics | High | N/A | — | — | — | — | — | — | *(not selected separately — see rationale)* |

### Selection: **MedQuAD + MedlinePlus + PMC OA Subset + WHO Publications + CDC Health Topics**

**Why not a separate "NIH Health Topics" source too?** NIH does not publish one
unified health-topics corpus — individual NIH institutes (NCI, NIDDK, NIAID,
GARD, etc.) each run their own site, and MedlinePlus/MedQuAD are themselves
built by aggregating and normalizing content from 12 of those NIH sites. Adding
NIH separately would mostly re-scrape content already present in MedQuAD and
MedlinePlus. The five sources above give complementary strengths instead of
overlapping ones:

- **MedQuAD** → structured Q&A pairs (fastest to index, great for direct
  question matching)
- **MedlinePlus** → consumer-friendly topic summaries + drug pages (broad
  coverage, easy licensing)
- **PMC OA** → deep technical/clinical detail from peer-reviewed literature
  (the piece that makes this project resume-worthy — real scientific-XML
  parsing, license filtering, and IR)
- **WHO** → global health, prevention, and policy guidance in PDF form (adds
  PDF-extraction skills to the pipeline)
- **CDC** → US public-health guidance, prevention, and structured indicator
  datasets (adds HTML scraping + a public REST/Socrata API to the pipeline)

Together they cover diseases, medicines, symptoms, lab tests, and preventive
care, use only trusted government/international/peer-reviewed sources, and
are all free for educational/non-commercial use (see per-source license notes
below — WHO content specifically is CC BY-NC-SA, so **non-commercial use
only**).

---

## 2. Per-dataset details

### 2.1 MedQuAD
- **Official name:** Medical Question Answering Dataset (MedQuAD)
- **Download:** https://github.com/abachaa/MedQuAD (`git clone`)
- **Docs:** same repo README; paper: Ben Abacha & Demner-Fushman, *BMC
  Bioinformatics* 2019
- **License:** CC BY 4.0
- **Format:** XML (one file per question, grouped into per-source folders)
- **Files to download:** the entire repository (~47,457 QA pairs across 12
  NIH-website collections)
- **Storage folder:** `data/raw/MedQuAD/`
- **Approx. size:** ~15 MB
- **Preprocessing:** parse `<QAPair>` nodes for `Question`/`Answer` text +
  `Focus`/`qtype` metadata; drop empty answers; clean whitespace.

### 2.2 MedlinePlus
- **Official name:** MedlinePlus Compressed Health Topic XML
- **Download:** https://medlineplus.gov/xml.html (index page; the actual file
  name changes daily, e.g. `mplus_topics_compressed_2026-07-15.xml`)
- **Docs:** https://medlineplus.gov/about/developers/ and
  https://medlineplus.gov/xml.html
- **License:** Free to download/reuse with attribution to MedlinePlus.gov
  (NLM); see https://support.nlm.nih.gov/kbArticle/?pn=KA-04683
- **Format:** XML
- **Files to download:** the current "MedlinePlus Compressed Health Topic
  XML" file (all English health topics in one file)
- **Storage folder:** `data/raw/MedlinePlus/`
- **Approx. size:** ~30–40 MB
- **Preprocessing:** parse `<health-topic>` nodes, extract `title` +
  `full-summary`, strip embedded HTML tags in the summary.

### 2.3 PubMed Central Open Access Subset (PMC OA)
- **Official name:** PMC Open Access Subset
- **Download:** via NCBI E-utilities (`esearch`) + PMC OA Web Service
  (`oa.fcgi`) — see https://ftp.ncbi.nlm.nih.gov/pub/pmc/ for the raw FTP tree
- **Docs:** https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ and
  https://www.ncbi.nlm.nih.gov/pmc/tools/oai/
- **License:** Mixed (CC0, CC BY, CC BY-SA, CC BY-ND = commercial-use-allowed;
  CC BY-NC* = non-commercial only). The script logs each article's license;
  the search terms in `config.py` favor commercial-use-allowed review articles
  but does not currently filter noncommercial ones out — tighten the filter in
  `download_pmc()` if you need commercial-use-only.
- **Format:** JATS XML (`.nxml`) + media, packaged as `.tar.gz`
- **Files to download:** individual OA article packages returned by search
  (default: 8 topic queries × 15 articles = up to 120 articles; tune via
  `PMC_SEARCH_TERMS` / `PMC_MAX_ARTICLES_PER_TERM` in `config.py`)
- **Storage folder:** `data/raw/PMC_OA/`
- **Approx. size:** ~1–3 MB per article package → ~150–350 MB for the default
  pull
- **Preprocessing:** extract `.nxml` from each tarball, parse
  `article-title`, `abstract//p`, `body//p` with `xml.etree.ElementTree`,
  strip tags/refs, clean whitespace.

### 2.4 WHO Publications
- **Official name:** WHO IRIS (Institutional Repository for Information
  Sharing)
- **Download:** via the IRIS REST API — https://iris.who.int/server/api/discover/search/objects
- **Docs:** https://www.who.int/about/policies/publishing/open-access and
  https://iris.who.int
- **License:** CC BY-NC-SA 3.0 IGO — **non-commercial use only**, share-alike,
  attribution required
- **Format:** PDF
- **Files to download:** top N publications per topic query (default: 8
  topics × 5 docs = up to 40 PDFs; tune via `WHO_SEARCH_QUERIES` /
  `WHO_MAX_DOCS_PER_QUERY` in `config.py`)
- **Storage folder:** `data/raw/WHO/`
- **Approx. size:** ~1–5 MB per PDF → ~50–150 MB for the default pull
- **Preprocessing:** extract text per page with `pypdf`, join, clean
  whitespace; drop documents whose extracted text is too short (scanned
  images with no text layer).

### 2.5 CDC Health Topics
- **Official name:** CDC Health Topics + CDC Open Data (Socrata)
- **Download:**
  - Structured data: https://data.cdc.gov/resource/{dataset-id}.json (Socrata
    Open Data API, no key required for light use)
  - Narrative pages: curated list of `cdc.gov` Health Topic landing pages in
    `config.CDC_HEALTH_TOPIC_PAGES` (no unified bulk-text API exists for
    these — see note below)
- **Docs:** https://data.cdc.gov and https://dev.socrata.com/
- **License:** U.S. Government work — public domain in the US (17 U.S.C. §105);
  CDC still asks for source attribution as a courtesy
- **Format:** JSON (structured datasets) + HTML (topic pages)
- **Files to download:** the datasets/pages listed in
  `CDC_OPEN_DATA_DATASETS` and `CDC_HEALTH_TOPIC_PAGES` in `config.py`
- **Storage folder:** `data/raw/CDC/`
- **Approx. size:** a few MB (small demo set; scale up by adding more
  dataset IDs / URLs)
- **Preprocessing:** JSON rows are flattened into short `key: value` text
  snippets; HTML pages are parsed with BeautifulSoup, `<script>/<style>/
  <nav>/<footer>` stripped, and the `<main>` content's text extracted.
- **Important caveat:** CDC does **not** provide an official bulk-download
  API for the narrative "Health Topics A-Z" text itself. The script performs
  a small, polite, rate-limited fetch of a curated URL list. For production
  use, review CDC's site terms, expand the curated URL list deliberately, and
  consider requesting data directly from CDC where higher volume is needed.

---

## 3. Project structure

```
medical-rag-chatbot/
│
├── README.md
├── requirements.txt
├── config.py                     # all paths & dataset settings in one place
├── download_datasets.py          # Step 1: fetch raw data
├── prepare_knowledge_base.py     # Step 2: clean, chunk, embed, index
├── query_knowledge_base.py       # optional: sanity-check retrieval
│
├── data/
│   ├── raw/
│   │   ├── MedQuAD/               # cloned git repo (XML QA pairs)
│   │   ├── MedlinePlus/           # mplus_topics_compressed.xml
│   │   ├── PMC_OA/                # PMCxxxxxxx.tar.gz packages
│   │   ├── WHO/                   # publication PDFs
│   │   └── CDC/                   # topic .html pages + Socrata .json
│   │
│   ├── processed/                 # (reserved for intermediate exports)
│   └── vector_store/
│       ├── medical_index.faiss    # FAISS vector index
│       └── chunk_metadata.pkl     # chunk text + source metadata
│
└── logs/
    ├── download.log
    └── prepare_knowledge_base.log
```

---

## 4. Setup & usage

```bash
# 1. Create an environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download all datasets (safe to re-run — already-downloaded files are skipped)
python download_datasets.py

#    Optional: only some datasets, or a smaller demo pull
python download_datasets.py --only medquad medlineplus
python download_datasets.py --pmc-per-term 5 --who-per-query 2

# 3. Build the knowledge base (clean -> dedupe -> chunk -> embed -> FAISS index)
python prepare_knowledge_base.py

# 4. Sanity-check retrieval
python query_knowledge_base.py "what are the early symptoms of diabetes"
```

Requirements: Python 3.9+, ~2 GB free disk space for the default-sized pull,
`git` installed (for the MedQuAD clone), and outbound internet access to
`github.com`, `medlineplus.gov`, `eutils.ncbi.nlm.nih.gov`,
`ftp.ncbi.nlm.nih.gov`, `iris.who.int`, `data.cdc.gov`, and `cdc.gov`.

## 5. Notes on scale & production hardening

- The default query/term limits in `config.py` are tuned for a fast demo
  build (a few hundred MB, minutes to run). Raise
  `PMC_MAX_ARTICLES_PER_TERM`, `WHO_MAX_DOCS_PER_QUERY`, and the CDC page
  list for a larger production corpus.
- `IndexFlatIP` (exact search) is used for simplicity and correctness at
  demo scale. For a corpus of 1M+ chunks, switch to `faiss.IndexIVFFlat` or
  `IndexHNSWFlat` for faster approximate search — the surrounding code
  (`build_index`/`save_index`) does not need to change beyond the index type.
- No API keys are required anywhere in this pipeline; the embedding model
  runs locally via `sentence-transformers`.
- This project is for **educational/demo purposes**. It is not a medical
  device and must not be used to provide diagnosis or treatment advice
  without appropriate clinical review, disclaimers, and — for WHO content —
  respecting the CC BY-NC-SA non-commercial restriction.
