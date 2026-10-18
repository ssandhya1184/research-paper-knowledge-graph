
from pathlib import Path

from research_assistant.ingestion.pdf_processor import (
    extract_text,
    save_extracted_text,
)

papers_dir = Path("data/papers")
pdf_files = list(papers_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDFs found in data/papers/")

pdf_path = pdf_files[0]

result = extract_text(pdf_path)

print(f"Paper: {pdf_path.name}")
print(f"Pages extracted: {result['total_pages']}")
print("\nFirst 1000 characters:\n")
print(result["full_text"][:1000])

save_extracted_text(pdf_path)
