
import json
from pathlib import Path

from research_assistant.agents.knowledge_extractor import extract_knowledge

files = [
    p for p in Path("data/processed").glob("*.json")
    if p.name != "papers.json"
]

if not files:
    raise FileNotFoundError("No extracted paper JSON files found.")

paper = json.loads(files[0].read_text(encoding="utf-8"))

result = extract_knowledge(paper["full_text"])

output_path = Path("data/processed") / (
    f"{paper['paper_id']}_knowledge.json"
)

output_path.write_text(
    result.model_dump_json(indent=2),
    encoding="utf-8",
)

print(f"Extracted knowledge saved to: {output_path}")
print(result.model_dump_json(indent=2))
