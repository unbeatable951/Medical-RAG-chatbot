import json
from pathlib import Path
from lxml import etree


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "MedQuAD"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "medquad.jsonl"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(text):
    """
    Clean whitespace while preserving readable text.
    """

    if not text:
        return ""

    text = " ".join(text.split())

    return text.strip()


def get_element_text(element):
    """
    Extract all text contained inside an XML element.
    """

    if element is None:
        return ""

    text = "".join(element.itertext())

    return clean_text(text)


# ---------------------------------------------------------
# Parse one XML document
# ---------------------------------------------------------

def parse_document(xml_file):

    tree = etree.parse(str(xml_file))
    root = tree.getroot()

    document_id = root.get("id", "")
    source = root.get("source", "")
    url = root.get("url", "")

    # -----------------------------------------------------
    # Focus
    # -----------------------------------------------------

    focus_element = root.find("Focus")
    focus = get_element_text(focus_element)

    # Use a safe fallback for empty Focus
    title = focus if focus else "MedQuAD"

    records = []

    # -----------------------------------------------------
    # QAPairs
    # -----------------------------------------------------

    qa_pairs = root.find("QAPairs")

    if qa_pairs is None:
        return records

    for qa_pair in qa_pairs.findall("QAPair"):

        pid = qa_pair.get("pid", "")

        question_element = qa_pair.find("Question")
        answer_element = qa_pair.find("Answer")

        question = get_element_text(question_element)
        answer = get_element_text(answer_element)

        # -------------------------------------------------
        # qtype
        # -------------------------------------------------

        qtype = ""

        if question_element is not None:
            qtype = question_element.get("qtype", "")

        # -------------------------------------------------
        # Original QID
        # -------------------------------------------------

        original_qid = ""

        if question_element is not None:
            original_qid = question_element.get("qid", "")

        # -------------------------------------------------
        # Skip incomplete records
        # -------------------------------------------------

        if not question or not answer:
            continue

        # -------------------------------------------------
        # Record
        # -------------------------------------------------

        record = {
            "id": "",
            "source": "medquad",
            "title": title,
            "question": question,
            "text": answer,
            "url": url,
            "metadata": {
                "focus": focus,
                "qtype": qtype,
                "document_id": document_id,
                "pid": pid,
                "original_qid": original_qid,
                "source_name": source
            }
        }

        records.append(record)

    return records


# ---------------------------------------------------------
# Parse entire MedQuAD dataset
# ---------------------------------------------------------

def main():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"MedQuAD directory not found: {RAW_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    xml_files = list(RAW_DIR.rglob("*.xml"))

    print(f"Found {len(xml_files)} XML files.")

    total_records = 0
    failed_files = 0
    global_record_id = 0

    # -----------------------------------------------------
    # Write JSONL
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for index, xml_file in enumerate(
            xml_files,
            start=1
        ):

            try:

                records = parse_document(xml_file)

                for record in records:

                    # -------------------------------------
                    # Guaranteed globally unique ID
                    # -------------------------------------

                    global_record_id += 1

                    record["id"] = (
                        f"medquad_{global_record_id:08d}"
                    )

                    output.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    total_records += 1

            except Exception as e:

                failed_files += 1

                print(
                    f"ERROR: {xml_file}"
                )

                print(
                    f"       {e}"
                )

            if index % 500 == 0:

                print(
                    f"Processed "
                    f"{index}/{len(xml_files)} files..."
                )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()

    print("=" * 60)
    print("MEDQUAD INGESTION COMPLETE")
    print("=" * 60)

    print(f"XML files:       {len(xml_files)}")
    print(f"Q&A records:     {total_records}")
    print(f"Failed files:    {failed_files}")
    print(f"Output:          {OUTPUT_FILE}")


if __name__ == "__main__":
    main()