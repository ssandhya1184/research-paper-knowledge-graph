
import json
from pathlib import Path
from urllib.request import urlretrieve
import arxiv


DATA_DIR = Path("data")
PAPERS_DIR = DATA_DIR / "papers"
METADATA_DIR = DATA_DIR / "processed"


def fetch_papers(query: str = "cat:cs.LG", max_results: int = 5) -> list[dict]:
    """Search arXiv, download PDFs, and save paper metadata."""

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []

    for result in client.results(search):
        paper_id = result.get_short_id()
        pdf_path = PAPERS_DIR / f"{paper_id.replace('/', '_')}.pdf"

        print(f"Processing: {result.title}")

        if not pdf_path.exists():
            urlretrieve(result.pdf_url, pdf_path)

        paper = {
            "paper_id": paper_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": [author.name for author in result.authors],
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat(),
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "pdf_path": str(pdf_path),
        }

        papers.append(paper)

    metadata_path = METADATA_DIR / "papers.json"
    metadata_path.write_text(
        json.dumps(papers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDownloaded {len(papers)} papers.")
    print(f"Metadata saved to: {metadata_path}")

    return papers
