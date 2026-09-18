from pathlib import Path
from lxml import etree

# Define project root first
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the path to your MedlinePlus XML file
FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MedlinePlus"
    / "mplus_topics_compressed.xml"
)

print("Reading:", FILE)

if not FILE.exists():
    raise FileNotFoundError(f"Cannot find XML file at: {FILE}")

tree = etree.parse(str(FILE))
topics = tree.xpath("//health-topic")

found = False
for topic in topics:
    if topic.get("title") == "Electrical Injuries":
        print("=" * 80)
        print("FOUND: Electrical Injuries")
        print("=" * 80)
        print(etree.tostring(topic, pretty_print=True, encoding="unicode"))
        found = True
        break

if not found:
    print("Could not find a health topic titled 'Electrical Injuries'.")