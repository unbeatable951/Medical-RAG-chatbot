import json
import warnings
import html as html_lib  # Added to unescape CDATA HTML entities
from pathlib import Path
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from lxml import etree

# Suppress spurious URL warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Dynamically find project root (medical-rag-chatbot directory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Point to exact XML location and output path
INPUT_FILE = BASE_DIR / "data" / "raw" / "MedlinePlus" / "mplus_topics_compressed.xml"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "medlineplus.jsonl"


def clean_html(text):
    """Unwrap link tags, format lists/paragraphs, and strip HTML markup into clean text."""
    if not text:
        return ""

    # Unescape HTML entities (e.g., &lt;p&gt; becomes <p>) from CDATA sections
    # so BeautifulSoup can parse them as actual HTML tags and strip them.
    text = html_lib.unescape(str(text))

    soup = BeautifulSoup(text, "html.parser")

    # Remove links but keep their visible text
    for tag in soup.find_all("a"):
        tag.unwrap()

    # Convert list items into readable bullet lines
    for tag in soup.find_all("li"):
        tag.insert_before("\n- ")
        tag.insert_after("\n")

    # Convert paragraphs into separate blocks
    for tag in soup.find_all("p"):
        tag.insert_after("\n")

    # Extract text and strip whitespace
    clean_text = soup.get_text(" ", strip=True)

    # Normalize whitespace line by line
    lines = []
    for line in clean_text.splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)

    return "\n".join(lines)


def get_text(element, xpath):
    """Return cleaned, plain text from the first matching element."""
    result = element.xpath(xpath)
    if not result:
        return ""

    if isinstance(result[0], str):
        raw_str = result[0]
    else:
        raw_str = etree.tostring(result[0], encoding="unicode")

    return clean_html(raw_str)


def get_text_list(element, xpath):
    """Return a list of cleaned plain text values."""
    results = element.xpath(xpath)
    values = []
    for result in results:
        if isinstance(result, str):
            raw_str = result
        else:
            raw_str = etree.tostring(result, encoding="unicode")

        cleaned = clean_html(raw_str)
        if cleaned:
            values.append(cleaned)
    return values


def get_sites(element):
    """Extract nested site element details for provenance."""
    sites = []
    for site in element.xpath("./site"):
        site_data = {
            "title": clean_html(site.get("title", "")),
            "url": site.get("url", ""),
            "organization": get_text(site, "./organization"),
            "information_category": get_text(site, "./information-category"),
        }
        if site_data["title"] or site_data["url"]:
            sites.append(site_data)
    return sites


def parse_medlineplus():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Source XML file not found at: {INPUT_FILE}")

    tree = etree.parse(str(INPUT_FILE))
    topics = tree.xpath("//health-topic")

    print(f"Found {len(topics)} health topics.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for index, topic in enumerate(topics):
            title = topic.get("title", "")
            url = topic.get("url", "")

            # Fallback to meta-desc if full-summary is missing
            summary = get_text(topic, "./full-summary")
            if not summary:
                summary = clean_html(topic.get("meta-desc", ""))

            also_called = get_text_list(topic, "./also-called")
            groups = get_text_list(topic, "./group")
            mesh_headings = get_text_list(topic, "./mesh-heading")
            organizations = get_text_list(
                topic, "./primary-institute | .//site/organization"
            )
            see_references = get_text_list(topic, "./see-reference")
            sites = get_sites(topic)

            document = {
                "id": f"medlineplus_{index:06d}",
                "source": "medlineplus",
                "title": clean_html(title),
                "text": summary,
                "url": url,
                "metadata": {
                    "also_called": also_called,
                    "groups": groups,
                    "mesh_headings": mesh_headings,
                    "organizations": list(set(organizations)),
                    "see_references": see_references,
                    "related_sites": sites,
                },
            }

            f.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(f"Saved processed data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parse_medlineplus()