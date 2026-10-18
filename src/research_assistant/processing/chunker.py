import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")


def chunk_paper(json_path: str | Path) -> list[dict]:
    """Split extracted paper text into overlapping chunks."""

    json_path = Path(json_path)

    paper = json.loads(json_path.read_text(encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(paper["full_text"])

    chunk_records = [
        {
            "paper_id": paper["paper_id"],
            "chunk_id": f"{paper['paper_id']}_chunk_{i}",
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CHUNKS_DIR / f"{paper['paper_id']}_chunks.json"
    output_path.write_text(
        json.dumps(chunk_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Paper: {paper['paper_id']}")
    print(f"Total chunks: {len(chunk_records)}")
    print(f"Saved to: {output_path}")

    return chunk_records