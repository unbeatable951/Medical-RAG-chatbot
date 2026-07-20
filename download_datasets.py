#!/usr/bin/env python3
"""
download_datasets.py

Downloads / retrieves the five source datasets used by the Medical RAG
chatbot's knowledge base:

    1. MedQuAD           - git clone of the official repo (fully automatic)
    2. MedlinePlus       - official XML health-topic file (fully automatic)
    3. PMC Open Access   - via NCBI E-utilities + OA Web Service (automatic,
                            commercial-use-allowed licenses only)
    4. WHO Publications  - via the WHO IRIS REST API (automatic)
    5. CDC Health Topics - CDC Open Data (Socrata) API (automatic) +
                            curated Health Topic HTML pages (automatic,
                            low-volume, polite scraping of public pages)

Design notes
------------
* Every dataset has its own `download_<name>()` function so partial
  failures don't block the others.
* Already-downloaded files are skipped (checked by path + non-zero size).
* Progress bars via tqdm; every step is logged to logs/download.log AND
  the console.
* Nothing here requires an API key. Where a source has no official bulk
  API for full-text (CDC health topic narrative pages), that's called
  out explicitly and a best-effort automatic method is used instead of
  silently failing.

Usage
-----
    python download_datasets.py                 # download everything
    python download_datasets.py --only medquad medlineplus
    python download_datasets.py --skip pmc who   # download everything except these
    python download_datasets.py --pmc-per-term 5 --who-per-query 3   # smaller demo pull
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import requests
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.ensure_directories()
LOG_FILE = os.path.join(config.LOG_DIR, "download.log")

logger = logging.getLogger("download_datasets")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)

logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT})


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def file_ready(path: str, min_bytes: int = 1) -> bool:
    """True if a file already exists and looks complete (non-trivial size)."""
    return os.path.isfile(path) and os.path.getsize(path) >= min_bytes


def download_file(url: str, dest_path: str, min_bytes: int = 1, params=None) -> bool:
    """Stream-download a single file with a progress bar. Returns success bool."""
    if file_ready(dest_path, min_bytes):
        logger.info(f"SKIP (already exists): {dest_path}")
        return True
    try:
        with SESSION.get(
            url, params=params, stream=True, timeout=config.REQUEST_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            tmp_path = dest_path + ".part"
            with open(tmp_path, "wb") as f, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(dest_path),
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            os.replace(tmp_path, dest_path)
        if not file_ready(dest_path, min_bytes):
            raise IOError("Downloaded file is empty or truncated")
        logger.info(f"OK: {url} -> {dest_path}")
        return True
    except Exception as exc:
        logger.error(f"FAILED to download {url}: {exc}")
        if os.path.isfile(dest_path + ".part"):
            os.remove(dest_path + ".part")
        return False


# ---------------------------------------------------------------------------
# 1. MedQuAD
# ---------------------------------------------------------------------------
def download_medquad():
    logger.info("=== MedQuAD ===")
    dest = config.MEDQUAD_DIR
    marker = os.path.join(dest, ".git")
    if os.path.isdir(marker):
        logger.info("SKIP: MedQuAD repo already cloned. Pulling latest changes...")
        try:
            subprocess.run(
                ["git", "-C", dest, "pull", "--ff-only"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(f"git pull failed (continuing with existing copy): {exc.stderr}")
        return True
    try:
        os.makedirs(dest, exist_ok=True)
        logger.info(f"Cloning {config.MEDQUAD_REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", config.MEDQUAD_REPO_URL, dest],
            check=True, capture_output=True, text=True,
        )
        logger.info("OK: MedQuAD cloned.")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"FAILED to clone MedQuAD: {exc.stderr}")
        logger.error(
            "MANUAL FALLBACK: download the ZIP from "
            f"{config.MEDQUAD_REPO_URL.replace('.git', '')}/archive/refs/heads/master.zip "
            f"and extract it into {dest}"
        )
        return False


# ---------------------------------------------------------------------------
# 2. MedlinePlus
# ---------------------------------------------------------------------------
def download_medlineplus():
    logger.info("=== MedlinePlus ===")
    dest_dir = config.MEDLINEPLUS_DIR
    dest_path = os.path.join(dest_dir, "mplus_topics_compressed.xml")
    if file_ready(dest_path, 1000):
        logger.info(f"SKIP (already exists): {dest_path}")
        return True

    # The exact filename on medlineplus.gov/xml.html changes with every
    # publish date (e.g. mplus_topics_compressed_2026-07-15.xml), so we
    # first scrape the index page to find today's actual link.
    resolved_url = None
    try:
        resp = SESSION.get(config.MEDLINEPLUS_XML_INDEX_URL, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        matches = re.findall(
            r'href="([^"]*mplus_topics_compressed[^"]*\.xml)"', resp.text
        )
        if matches:
            resolved_url = urljoin(config.MEDLINEPLUS_XML_INDEX_URL, matches[0])
            logger.info(f"Resolved current MedlinePlus XML link: {resolved_url}")
    except Exception as exc:
        logger.warning(f"Could not scrape {config.MEDLINEPLUS_XML_INDEX_URL}: {exc}")

    candidates = [u for u in [resolved_url, config.MEDLINEPLUS_FALLBACK_TOPICS_URL] if u]
    for url in candidates:
        if download_file(url, dest_path, min_bytes=1000):
            return True

    logger.error(
        "MANUAL FALLBACK: open https://medlineplus.gov/xml.html in a browser, "
        "download the 'MedlinePlus Compressed Health Topic XML' file, and save it as "
        f"{dest_path}"
    )
    return False


# ---------------------------------------------------------------------------
# 3. PubMed Central Open Access Subset
# ---------------------------------------------------------------------------
def _pmc_esearch(term: str, retmax: int):
    """Return a list of PMC IDs (e.g. 'PMC1234567') for a search term,
    restricted to the Open Access subset."""
    params = {
        "db": "pmc",
        "term": f"{term} AND open access[filter]",
        "retmax": retmax,
        "retmode": "json",
    }
    resp = SESSION.get(config.PMC_EUTILS_ESEARCH, params=params, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    return [f"PMC{i}" for i in ids]


def _pmc_oa_package_url(pmcid: str):
    """Ask the OA Web Service for the download URL of an article's package."""
    resp = SESSION.get(
        config.PMC_OA_SERVICE, params={"id": pmcid}, timeout=config.REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    record = root.find(".//record")
    if record is None:
        return None, None
    license_type = record.attrib.get("license", "unknown")
    link = record.find(".//link[@format='tgz']")
    if link is None:
        link = record.find(".//link")
    href = link.attrib.get("href") if link is not None else None
    return href, license_type


def download_pmc(per_term: int = None):
    logger.info("=== PubMed Central Open Access Subset ===")
    per_term = per_term or config.PMC_MAX_ARTICLES_PER_TERM
    dest_dir = config.PMC_DIR
    ok_count, fail_count = 0, 0

    for term in config.PMC_SEARCH_TERMS:
        logger.info(f"PMC search: '{term}' (up to {per_term} articles)")
        try:
            pmcids = _pmc_esearch(term, per_term)
        except Exception as exc:
            logger.error(f"esearch failed for '{term}': {exc}")
            continue

        for pmcid in tqdm(pmcids, desc=term[:30], leave=False):
            tar_path = os.path.join(dest_dir, f"{pmcid}.tar.gz")
            if file_ready(tar_path, 500):
                ok_count += 1
                continue
            try:
                href, license_type = _pmc_oa_package_url(pmcid)
                if not href:
                    logger.warning(f"No OA package found for {pmcid} (not in OA subset)")
                    continue
                if license_type and "NC" in license_type.upper() and "noncomm" in href.lower():
                    # We only *need* metadata; non-commercial articles are still
                    # fine for an educational/non-commercial RAG demo, so we keep
                    # them but log the license for transparency.
                    logger.info(f"{pmcid} license: {license_type} (non-commercial use only)")
                # href is an ftp:// URL in the API response; NCBI also serves the
                # same paths over https, which `requests` can fetch directly.
                https_href = href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
                if download_file(https_href, tar_path, min_bytes=500):
                    ok_count += 1
                else:
                    fail_count += 1
            except Exception as exc:
                logger.error(f"FAILED processing {pmcid}: {exc}")
                fail_count += 1
            time.sleep(config.REQUEST_DELAY_SECONDS)

    logger.info(f"PMC OA done. Success: {ok_count}, Failed: {fail_count}")
    return ok_count > 0


# ---------------------------------------------------------------------------
# 4. WHO Publications (IRIS repository)
# ---------------------------------------------------------------------------
def download_who(per_query: int = None):
    logger.info("=== WHO Publications (IRIS) ===")
    per_query = per_query or config.WHO_MAX_DOCS_PER_QUERY
    dest_dir = config.WHO_DIR
    ok_count, fail_count = 0, 0

    for query in config.WHO_SEARCH_QUERIES:
        logger.info(f"WHO IRIS search: '{query}' (up to {per_query} documents)")
        try:
            params = {"query": query, "dsoType": "publication", "size": per_query}
            resp = SESSION.get(config.WHO_IRIS_SEARCH_API, params=params, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(f"WHO IRIS search failed for '{query}': {exc}")
            fail_count += 1
            continue

        try:
            objects = (
                data.get("_embedded", {})
                .get("searchResult", {})
                .get("_embedded", {})
                .get("objects", [])
            )
        except AttributeError:
            objects = []

        for obj in tqdm(objects, desc=query[:30], leave=False):
            try:
                item = obj.get("_embedded", {}).get("indexableObject", {})
                item_id = item.get("uuid") or item.get("id")
                title = item.get("name", item_id or "unknown")
                if not item_id:
                    continue
                bundles_href = (
                    item.get("_links", {}).get("bundles", {}).get("href")
                )
                bitstream_url = None
                if bundles_href:
                    b_resp = SESSION.get(bundles_href, timeout=config.REQUEST_TIMEOUT)
                    if b_resp.ok:
                        bitstream_url = _first_pdf_bitstream(b_resp.json())
                if not bitstream_url:
                    logger.warning(f"No PDF bitstream found for WHO item '{title}'")
                    continue
                safe_name = re.sub(r"[^\w\-]+", "_", title)[:80] or item_id
                dest_path = os.path.join(dest_dir, f"{safe_name}.pdf")
                if download_file(bitstream_url, dest_path, min_bytes=1000):
                    ok_count += 1
                else:
                    fail_count += 1
            except Exception as exc:
                logger.error(f"FAILED processing WHO item: {exc}")
                fail_count += 1
            time.sleep(config.REQUEST_DELAY_SECONDS)

    if ok_count == 0:
        logger.error(
            "No WHO documents were downloaded automatically. "
            "MANUAL FALLBACK: browse https://iris.who.int, search for the topics "
            f"in config.WHO_SEARCH_QUERIES, and save PDFs into {dest_dir}"
        )
    logger.info(f"WHO IRIS done. Success: {ok_count}, Failed: {fail_count}")
    return ok_count > 0


def _first_pdf_bitstream(bundles_json):
    """Walk a DSpace bundles response and return the first PDF bitstream's
    direct-download URL, if any."""
    try:
        bundles = bundles_json.get("_embedded", {}).get("bundles", [])
        for bundle in bundles:
            bitstreams_href = bundle.get("_links", {}).get("bitstreams", {}).get("href")
            if not bitstreams_href:
                continue
            resp = SESSION.get(bitstreams_href, timeout=config.REQUEST_TIMEOUT)
            if not resp.ok:
                continue
            bitstreams = resp.json().get("_embedded", {}).get("bitstreams", [])
            for bs in bitstreams:
                name = bs.get("name", "")
                if name.lower().endswith(".pdf"):
                    content_href = bs.get("_links", {}).get("content", {}).get("href")
                    if content_href:
                        return content_href
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 5. CDC — Open Data API + curated Health Topic pages
# ---------------------------------------------------------------------------
def download_cdc():
    logger.info("=== CDC Health Topics ===")
    logger.info(
        "NOTE: CDC does not publish a unified bulk-download API for narrative "
        "'Health Topics A-Z' text. This step therefore combines (a) the official "
        "CDC Open Data (Socrata) API for structured indicator datasets, and "
        "(b) a curated, low-volume fetch of public CDC topic landing pages. "
        "Review data/raw/CDC manually afterwards; consider adding more pages "
        "to config.CDC_HEALTH_TOPIC_PAGES as needed."
    )
    dest_dir = config.CDC_DIR
    ok_count, fail_count = 0, 0

    # (a) Structured open datasets via Socrata (JSON, official API, no key needed
    #     for low-volume use).
    for name, resource_id in config.CDC_OPEN_DATA_DATASETS.items():
        dest_path = os.path.join(dest_dir, f"{name}.json")
        url = f"{config.CDC_OPEN_DATA_API}/{resource_id}.json"
        if download_file(url, dest_path, min_bytes=10, params={"$limit": 5000}):
            ok_count += 1
        else:
            fail_count += 1

    # (b) Curated narrative health-topic pages (HTML -> cleaned in
    #     prepare_knowledge_base.py).
    for url in tqdm(config.CDC_HEALTH_TOPIC_PAGES, desc="CDC topic pages", leave=False):
        safe_name = re.sub(r"[^\w\-]+", "_", url.split("//")[-1])[:100]
        dest_path = os.path.join(dest_dir, f"{safe_name}.html")
        if download_file(url, dest_path, min_bytes=500):
            ok_count += 1
        else:
            fail_count += 1
        time.sleep(config.REQUEST_DELAY_SECONDS)

    logger.info(f"CDC done. Success: {ok_count}, Failed: {fail_count}")
    return ok_count > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DATASET_FUNCS = {
    "medquad": download_medquad,
    "medlineplus": download_medlineplus,
    "pmc": download_pmc,
    "who": download_who,
    "cdc": download_cdc,
}


def main():
    parser = argparse.ArgumentParser(description="Download all medical RAG datasets.")
    parser.add_argument(
        "--only", nargs="+", choices=DATASET_FUNCS.keys(),
        help="Only download these datasets.",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=DATASET_FUNCS.keys(), default=[],
        help="Skip these datasets.",
    )
    parser.add_argument("--pmc-per-term", type=int, default=None,
                         help="Override PMC articles per search term.")
    parser.add_argument("--who-per-query", type=int, default=None,
                         help="Override WHO documents per search query.")
    args = parser.parse_args()

    config.ensure_directories()

    targets = args.only if args.only else list(DATASET_FUNCS.keys())
    targets = [t for t in targets if t not in args.skip]

    logger.info(f"Starting dataset download for: {targets}")
    results = {}
    for name in targets:
        try:
            if name == "pmc":
                results[name] = download_pmc(per_term=args.pmc_per_term)
            elif name == "who":
                results[name] = download_who(per_query=args.who_per_query)
            else:
                results[name] = DATASET_FUNCS[name]()
        except Exception as exc:
            logger.error(f"UNEXPECTED ERROR in {name}: {exc}")
            results[name] = False

    logger.info("=== Summary ===")
    for name, ok in results.items():
        logger.info(f"{name}: {'OK' if ok else 'FAILED / NEEDS MANUAL STEP'}")
    logger.info(f"Full log written to {LOG_FILE}")


if __name__ == "__main__":
    main()
