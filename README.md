# Medical RAG Chatbot

A Retrieval-Augmented-Generation chatbot built from trusted, publicly
available medical sources, with automated download/indexing scripts and a
Streamlit chat UI backed by Groq.

Sources: **MedQuAD**, **MedlinePlus**, **WHO Publications**, **CDC Health
Topics**. (PMC Open Access was intentionally excluded — NCBI is mid-migration
off its legacy FTP/OA-Web-Service APIs as of Aug 2026, and the four sources
above are plenty for a solid demo knowledge base.)

## Project structure

```
medical-rag-chatbot/
├── README.md
├── requirements.txt
├── config.py                     # all paths & dataset settings in one place
├── download_datasets.py          # Step 1: fetch raw data
├── prepare_knowledge_base.py     # Step 2: clean, chunk, embed, index
├── query_knowledge_base.py       # Step 3: sanity-check retrieval (no LLM)
├── app.py                        # Step 4: Streamlit chat UI
│
├── data/
│   ├── raw/
│   │   ├── MedQuAD/               # cloned git repo (XML QA pairs)
│   │   ├── MedlinePlus/           # mplus_topics_compressed.xml
│   │   ├── WHO/                   # publication PDFs
│   │   └── CDC/                   # topic .html pages + Socrata .json
│   └── vector_store/
│       ├── medical_index.faiss    # FAISS vector index
│       └── chunk_metadata.pkl     # chunk text + source metadata
│
└── logs/
    ├── download.log
    └── prepare_knowledge_base.log
```

## Setup & usage

```bash
# 1. Create an environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the datasets (safe to re-run — already-downloaded files are skipped)
python download_datasets.py

#    Optional: only some datasets, or a smaller demo pull
python download_datasets.py --only medquad medlineplus
python download_datasets.py --who-per-query 2

# 3. Check what actually landed before spending time on the index build
python verify_downloads.py

# 4. Build the knowledge base (clean -> dedupe -> chunk -> embed -> FAISS index)
python prepare_knowledge_base.py

# 5. Sanity-check retrieval (no LLM call yet — confirms the index itself is good)
python query_knowledge_base.py "what are the early symptoms of diabetes"

# 6. Run the chatbot (only after step 5 looks right)
export GROQ_API_KEY=your_key_here      # Windows PowerShell: $env:GROQ_API_KEY="your_key_here"
streamlit run app.py
```

Requirements: Python 3.9+, ~1 GB free disk space for the default-sized pull,
`git` installed (for the MedQuAD clone), and outbound internet access to
`github.com`, `medlineplus.gov`, `iris.who.int`, `data.cdc.gov`, and `cdc.gov`.

Get a free Groq API key at https://console.groq.com/keys.

## Recommended build order (don't skip steps)

Building in this order — and testing each step from the terminal before
moving to the next — is what catches bugs early instead of discovering them
inside the Streamlit UI:

1. `config.py` — defines every path/setting; nothing else needs touching to
   change scope.
2. `download_datasets.py` — start with `--only medquad` (it's a plain git
   clone and the most reliable source) to prove the rest of the pipeline
   works before dealing with any flakier API.
3. `verify_downloads.py` — confirms files actually landed before you burn
   time on embeddings.
4. `prepare_knowledge_base.py` — build the FAISS index.
5. `query_knowledge_base.py` — retrieval-only sanity check. **Don't skip
   this.** If results here don't look right, the bug is in retrieval, not
   the chat UI — much easier to debug without Streamlit in the way.
6. `app.py` — the chat UI, last, since it depends on everything above.

## Per-dataset details

### MedQuAD
- **Download:** `git clone https://github.com/abachaa/MedQuAD.git` (~47,457
  QA pairs across 12 NIH-website collections)
- **License:** CC BY 4.0
- **Format:** XML — one file per question
- **Preprocessing:** parse `<QAPair>` nodes for `Question`/`Answer` text +
  `Focus`/`qtype` metadata; drop empty answers.

### MedlinePlus
- **Download:** https://medlineplus.gov/xml.html — the actual filename
  changes daily (e.g. `mplus_topics_compressed_2026-07-15.xml`), so
  `download_datasets.py` scrapes the index page first to resolve today's
  real link before falling back to a static URL.
- **License:** Free to reuse with attribution to MedlinePlus.gov (NLM)
- **Preprocessing:** parse `<health-topic>` nodes, extract `title` +
  `full-summary`, strip embedded HTML.

### WHO Publications
- **Download:** WHO IRIS REST API — `https://iris.who.int/server/api/discover/search/objects`
- **License:** CC BY-NC-SA 3.0 IGO — **non-commercial use only**
- **Format:** PDF
- **Preprocessing:** extract text per page with `pypdf`; drop documents
  under 200 characters of extracted text (usually scanned images with no
  real text layer).

### CDC Health Topics
- **Download:** CDC Open Data (Socrata) API for structured datasets +
  a curated list of `cdc.gov` Health Topic landing pages (no unified
  bulk-text API exists for the narrative pages).
- **License:** U.S. Government work — public domain in the US
- **Preprocessing:** JSON rows flattened into short text snippets; HTML
  parsed with BeautifulSoup, boilerplate stripped, `<main>` content kept.

## Notes on scale & production hardening

- The default query/document limits in `config.py` are tuned for a fast
  demo build. Raise `WHO_MAX_DOCS_PER_QUERY` and the CDC page list for a
  larger production corpus.
- `IndexFlatIP` (exact search) is used for simplicity/correctness at demo
  scale. For 1M+ chunks, switch to `faiss.IndexIVFFlat` or `IndexHNSWFlat`.
- No API keys are required for the data pipeline; only `app.py` needs
  `GROQ_API_KEY`. The embedding model runs locally via `sentence-transformers`.
- **This project is for educational/demo purposes.** It is not a medical
  device and must not be used to provide diagnosis or treatment advice
  without appropriate clinical review — and WHO content specifically is
  CC BY-NC-SA, so **non-commercial use only** if you keep that source.
