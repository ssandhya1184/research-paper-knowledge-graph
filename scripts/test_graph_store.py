
import json
from pathlib import Path

from research_assistant.storage.graph_store import KnowledgeGraphStore


files = list(Path("data/processed").glob("*_knowledge_resolved.json"))

if not files:
    raise FileNotFoundError(
        "No resolved knowledge JSON found in data/processed/"
    )

data = json.loads(files[0].read_text(encoding="utf-8"))

# Load the original paper metadata, if available.
paper_id = data.get("paper_id", files[0].name.split("_knowledge")[0])
metadata_file = Path("data/processed/papers.json")

title = paper_id

if metadata_file.exists():
    papers = json.loads(metadata_file.read_text(encoding="utf-8"))
    for paper in papers:
        if paper.get("paper_id") == paper_id:
            title = paper.get("title", paper_id)
            break

entities = data.get("entities", [])
relationships = data.get("relationships", [])
aliases = data.get("aliases", {})

store = KnowledgeGraphStore()

store.add_paper(paper_id, title)
store.add_entities(entities, paper_id)
store.add_relationships(relationships, entities, aliases)

store.print_summary()
store.save(f"{paper_id}_graph.json")

print("\nSample nodes:")
for node, attributes in list(store.graph.nodes(data=True))[:5]:
    print(node, attributes)

print("\nSample relationships:")
for source, target, attributes in list(
    store.graph.edges(data=True)
)[:5]:
    print(source, "->", target, attributes)
