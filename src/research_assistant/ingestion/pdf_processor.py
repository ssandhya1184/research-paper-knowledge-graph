
import json
from pathlib import Path

import pymupdf  # PyMuPDF


PROCESSED_DIR = Path("data/processed")


def extract_text(pdf_path: str | Path) -> dict:
    """Extract text from a PDF, preserving page numbers."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

    return {
        "paper_id": pdf_path.stem,
        "pdf_path": str(pdf_path),
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(
            page["text"] for page in pages
        ),
    }


def save_extracted_text(pdf_path: str | Path) -> Path:
    """Extract PDF text and save it as a JSON file."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    extracted = extract_text(pdf_path)
    output_path = PROCESSED_DIR / f"{extracted['paper_id']}.json"

    output_path.write_text(
        json.dumps(extracted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Extracted {extracted['total_pages']} pages: {output_path}")

    return output_path
