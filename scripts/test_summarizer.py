
import json
from pathlib import Path

from research_assistant.agents.summarizer import (
    summarize_paper,
    save_summary,
)

files = [
    p for p in Path("data/processed").glob("*.json")
    if p.name != "papers.json"
    and "_knowledge" not in p.name
    and "_summary" not in p.name
]

if not files:
    raise FileNotFoundError("No extracted paper JSON files found.")

paper = json.loads(files[0].read_text(encoding="utf-8"))

summary = summarize_paper(paper)
output_path = save_summary(summary)

print(summary["summary"])
print(f"\nSaved summary to: {output_path}")
