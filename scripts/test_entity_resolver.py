
import json
from pathlib import Path

from research_assistant.processing.entity_resolver import resolve_entities


files = list(Path("data/processed").glob("*_knowledge.json"))

if not files:
    raise FileNotFoundError(
        "No knowledge extraction JSON files found."
    )

data = json.loads(files[0].read_text(encoding="utf-8"))

# Combine all extracted entity categories.
entities = (
    data.get("methods", [])
    + data.get("datasets", [])
    + data.get("metrics", [])
    + data.get("concepts", [])
    + data.get("findings", [])
)

unique_entities, aliases = resolve_entities(entities)

# Update relationship endpoints to use canonical names.
for relationship in data.get("relationships", []):
    relationship["source"] = aliases.get(
        relationship["source"], relationship["source"]
    )
    relationship["target"] = aliases.get(
        relationship["target"], relationship["target"]
    )

data["entities"] = unique_entities
data["aliases"] = aliases

output_path = files[0].with_name(
    files[0].stem + "_resolved.json"
)
output_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Original entities: {len(entities)}")
print(f"Unique entities: {len(unique_entities)}")
print(f"Aliases resolved: {len(aliases)}")
print(f"Saved to: {output_path}")
