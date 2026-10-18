
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from research_assistant.ingestion.arxiv_client import fetch_papers
from research_assistant.ingestion.pdf_processor import save_extracted_text
from research_assistant.processing.chunker import chunk_paper
from research_assistant.processing.entity_resolver import resolve_entities
from research_assistant.agents.knowledge_extractor import extract_knowledge
from research_assistant.storage.graph_store import KnowledgeGraphStore
from research_assistant.storage.vector_store import VectorStore


class IngestionState(TypedDict):
    query: str
    max_results: int
    papers: list[dict]
    processed_papers: list[str]
    total_chunks: int
    graph_paths: list[str]
    status: str


def fetch_node(state: IngestionState) -> dict:
    """Search arXiv and download papers."""

    papers = fetch_papers(
        query=state["query"],
        max_results=state["max_results"],
    )

    return {"papers": papers}


def process_node(state: IngestionState) -> dict:
    """Extract text, chunk papers, and index chunks."""

    vector_store = VectorStore()
    processed_papers = []
    total_chunks = 0

    for paper in state["papers"]:
        paper_id = paper["paper_id"]
        pdf_path = Path(paper["pdf_path"])

        extracted_path = save_extracted_text(pdf_path)
        chunks = chunk_paper(extracted_path)

        if not chunks:
            print(f"Skipping empty paper: {paper_id}")
            continue

        vector_store.add_chunks(chunks)

        processed_papers.append(paper_id)
        total_chunks += len(chunks)

    return {
        "processed_papers": processed_papers,
        "total_chunks": total_chunks,
    }


def graph_node(state: IngestionState) -> dict:
    """Extract knowledge and build a graph for each processed paper."""

    graph_paths = []

    for paper in state["papers"]:
        paper_id = paper["paper_id"]

        if paper_id not in state["processed_papers"]:
            continue

        extracted_path = Path("data/processed") / f"{paper_id}.json"
        if not extracted_path.exists():
            print(f"Missing extracted text for {paper_id}; skipping graph.")
            continue

        processed = json.loads(
            extracted_path.read_text(encoding="utf-8")
        )
        extraction = extract_knowledge(processed["full_text"])
        extraction_data = extraction.model_dump()

        # Combine all extracted entity categories.
        entities = []
        for category in (
            "methods",
            "datasets",
            "metrics",
            "concepts",
            "findings",
        ):
            entities.extend(extraction_data.get(category, []))

        unique_entities, aliases = resolve_entities(entities)

        # Save the raw structured extraction for inspection.
        extraction_path = (
            Path("data/processed") / f"{paper_id}_extraction.json"
        )
        extraction_path.write_text(
            json.dumps(
                extraction_data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        graph_store = KnowledgeGraphStore()
        graph_store.add_paper(
            paper_id=paper_id,
            title=paper.get("title", paper_id),
        )
        graph_store.add_entities(unique_entities, paper_id)
        graph_store.add_relationships(
            relationships=extraction_data.get("relationships", []),
            entities=unique_entities,
            aliases=aliases,
        )

        graph_path = graph_store.save(
            filename=f"{paper_id}_graph.json"
        )
        graph_store.print_summary()
        graph_paths.append(str(graph_path))

    return {"graph_paths": graph_paths}


def build_ingestion_graph():
    workflow = StateGraph(IngestionState)

    workflow.add_node("fetch", fetch_node)
    workflow.add_node("process", process_node)
    workflow.add_node("build_graph", graph_node)

    workflow.add_edge(START, "fetch")
    workflow.add_edge("fetch", "process")
    workflow.add_edge("process", "build_graph")
    workflow.add_edge("build_graph", END)

    return workflow.compile()


ingestion_graph = build_ingestion_graph()
