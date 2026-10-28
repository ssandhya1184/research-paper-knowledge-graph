
import json
from pathlib import Path
from urllib.request import urlretrieve

import arxiv


DATA_DIR = Path("data")
PAPERS_DIR = DATA_DIR / "papers"
METADATA_DIR = DATA_DIR / "processed"
METADATA_PATH = METADATA_DIR / "papers.json"


def search_arxiv(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Search arXiv and return metadata without downloading PDFs."""

    if not query or not query.strip():
        return []

    client = arxiv.Client()

    search = arxiv.Search(
        query=query.strip(),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for result in client.results(search):
        paper_id = result.get_short_id()

        papers.append({
            "paper_id": paper_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": [author.name for author in result.authors],
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat(),
            "categories": result.categories,
            "pdf_url": result.pdf_url,
        })

    return papers


def load_paper_metadata() -> list[dict]:
    """Load existing paper metadata, returning an empty list if absent."""

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        return []

    data = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    return data if isinstance(data, list) else []


def save_paper_metadata(new_papers: list[dict]) -> None:
    """Merge paper metadata by ID instead of overwriting the library."""

    existing_papers = load_paper_metadata()
    papers_by_id = {
        paper["paper_id"]: paper
        for paper in existing_papers
        if paper.get("paper_id")
    }

    for paper in new_papers:
        paper_id = paper.get("paper_id")

        if not paper_id:
            continue

        # Preserve existing fields, such as pdf_path, when updating metadata.
        merged = papers_by_id.get(paper_id, {}).copy()
        merged.update(paper)
        papers_by_id[paper_id] = merged

    METADATA_PATH.write_text(
        json.dumps(
            list(papers_by_id.values()),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def download_paper(paper: dict) -> dict:
    """Download one paper's PDF and save its metadata."""

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    paper_id = paper["paper_id"]
    pdf_path = PAPERS_DIR / f"{paper_id.replace('/', '_')}.pdf"

    if not pdf_path.exists():
        pdf_url = paper.get("pdf_url")

        if not pdf_url:
            raise ValueError(
                f"No PDF URL available for paper {paper_id}"
            )

        print(f"Downloading: {paper.get('title', paper_id)}")
        urlretrieve(pdf_url, pdf_path)

    downloaded_paper = {
        **paper,
        "pdf_path": str(pdf_path),
    }

    save_paper_metadata([downloaded_paper])

    return downloaded_paper


def fetch_papers(
    query: str = "cat:cs.LG",
    max_results: int = 5,
) -> list[dict]:
    """Search arXiv, download the matching PDFs, and save metadata.

    Kept compatible with the existing ingestion graph.
    """

    papers = search_arxiv(
        query=query,
        max_results=max_results,
    )

    downloaded_papers = []

    for paper in papers:
        try:
            downloaded = download_paper(paper)
            downloaded_papers.append(downloaded)
        except Exception as exc:
            print(
                f"Failed to download {paper.get('paper_id')}: {exc}"
            )

    print(f"\nDownloaded or found {len(downloaded_papers)} PDFs.")
    print(f"Metadata saved to: {METADATA_PATH}")

    return downloaded_papers

def get_ingestion_status(paper_id: str) -> str:
    """Return the processing status based on generated artifacts."""

    processed_path = METADATA_DIR / f"{paper_id}.json"
    graph_path = DATA_DIR / "graph" / f"{paper_id}_graph.json"
    pdf_path = PAPERS_DIR / f"{paper_id.replace('/', '_')}.pdf"

    if graph_path.exists() and processed_path.exists():
        return "Ingested"

    if processed_path.exists():
        return "Text extracted"

    if pdf_path.exists():
        return "PDF downloaded"

    return "Not ingested"
