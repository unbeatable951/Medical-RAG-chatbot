import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "CDC"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "cdc.jsonl"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PARSE CDC HTML
# ============================================================

def parse_html(html_file):

    with open(
        html_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        html = f.read()

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove unwanted elements
    for element in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header"
        ]
    ):
        element.decompose()

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = ""

    if soup.title:

        title = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )

    # --------------------------------------------------------
    # Main content
    # --------------------------------------------------------

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("body")
    )

    if main_content is None:
        return None

    text = clean_text(
        main_content.get_text(
            " ",
            strip=True
        )
    )

    if not text:
        return None

    # --------------------------------------------------------
    # Canonical URL
    # --------------------------------------------------------

    url = ""

    canonical = soup.find(
        "link",
        rel="canonical"
    )

    if canonical:

        url = canonical.get(
            "href",
            ""
        )

    # --------------------------------------------------------
    # Stable ID
    # --------------------------------------------------------

    file_id = html_file.stem

    record = {

        "id": f"cdc_{file_id}",

        "source": "cdc",

        "title": title,

        "text": text,

        "url": url,

        "metadata": {

            "source_name":
                "Centers for Disease Control and Prevention",

            "file_name":
                html_file.name,

            "content_type":
                "html"
        }
    }

    return record


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CDC DATA INGESTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"CDC directory not found: {RAW_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = [

        f

        for f in RAW_DIR.rglob("*")

        if f.is_file()
        and f.name != ".gitkeep"
    ]

    print(
        f"\nFound {len(files)} files."
    )

    total_records = 0
    skipped_files = 0
    failed_files = 0

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for index, file in enumerate(
            files,
            start=1
        ):

            try:

                suffix = file.suffix.lower()

                # =================================================
                # HTML
                # =================================================

                if suffix in [
                    ".html",
                    ".htm"
                ]:

                    record = parse_html(file)

                    if record is None:

                        skipped_files += 1

                        print(
                            f"Skipped empty HTML: "
                            f"{file.name}"
                        )

                        continue

                    output.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    total_records += 1

                # =================================================
                # JSON
                # =================================================

                elif suffix == ".json":

                    # IMPORTANT:
                    # The CDC influenza JSON contains
                    # 5,000 structured surveillance records.
                    #
                    # We are keeping it OUT of the main
                    # medical RAG knowledge base.

                    print(
                        f"Skipping structured CDC dataset: "
                        f"{file.name}"
                    )

                    skipped_files += 1

                # =================================================
                # Other files
                # =================================================

                else:

                    skipped_files += 1

            except Exception as e:

                failed_files += 1

                print()
                print(
                    f"ERROR: {file}"
                )

                print(
                    f"       {e}"
                )

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if index % 5 == 0:

                print(
                    f"Processed "
                    f"{index}/{len(files)} files..."
                )

    # ============================================================
    # RESULTS
    # ============================================================

    print()

    print("=" * 70)
    print("CDC INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"Files found      : {len(files)}"
    )

    print(
        f"Records created  : {total_records}"
    )

    print(
        f"Skipped files    : {skipped_files}"
    )

    print(
        f"Failed files     : {failed_files}"
    )

    print(
        f"Output           : {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()