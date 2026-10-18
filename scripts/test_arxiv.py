from research_assistant.ingestion.arxiv_client import fetch_papers

papers = fetch_papers(
    query="cat:cs.LG",
    max_results=5,
)

for paper in papers:
    print(f"\nTitle: {paper['title']}")
    print(f"Paper ID: {paper['paper_id']}")
    print(f"PDF: {paper['pdf_path']}")