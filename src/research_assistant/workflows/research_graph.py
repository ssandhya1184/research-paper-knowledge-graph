
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from research_assistant.agents.qa_agent import ResearchQAAgent
import json
from pathlib import Path
from research_assistant.agents.knowledge_extractor import extract_knowledge
from research_assistant.agents.summarizer import (
    summarize_paper,
    save_summary,
)

class ResearchState(TypedDict):
    question: str
    paper_id: str
    request_type: str
    vector_context: list[str]
    sources: list[dict]
    graph_relationships: list[str]
    answer: str
    summary_path: str
    extraction_path: str


qa_agent = ResearchQAAgent()


def router_node(state: ResearchState) -> dict:
    """Classify the request using simple keyword rules."""

    question = state["question"].lower()

    if any(word in question for word in ["summarize", "summary"]):
        request_type = "summarize"
    elif any(
        word in question
        for word in ["extract entities", "extract knowledge", "identify methods"]
    ):
        request_type = "extract"
    else:
        request_type = "qa"

    print(f"Request type: {request_type}")
    return {"request_type": request_type}


def route_request(
    state: ResearchState,
) -> Literal["qa", "summarize", "extract"]:
    return state["request_type"]


def vector_retrieval_node(state: ResearchState) -> dict:
    results = qa_agent.vector_store.search(state["question"], top_k=3, paper_id=state["paper_id"],)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    vector_context = []
    sources = []

    for document, metadata in zip(documents, metadatas):
        paper_id = metadata.get("paper_id", "Unknown")
        chunk_index = metadata.get("chunk_index", "Unknown")

        vector_context.append(
            f"Paper ID: {paper_id}\n"
            f"Chunk: {chunk_index}\n"
            f"Content:\n{document}"
        )

        source = {
            "paper_id": paper_id,
            "chunk_index": chunk_index,
        }
        if source not in sources:
            sources.append(source)

    return {"vector_context": vector_context, "sources": sources}


def graph_retrieval_node(state: ResearchState) -> dict:
    paper_id = state["paper_id"]
    graph_path = Path("data/graph") / f"{paper_id}_graph.json"

    if not graph_path.exists():
        return {
            "graph_relationships": [],
            "answer": f"Knowledge graph not found for paper: {paper_id}",
        }

    agent = ResearchQAAgent(graph_path=graph_path)

    return {
        "graph_relationships": agent.retrieve_graph_context(
            state["question"]
        )
    }


def answer_generation_node(state: ResearchState) -> dict:
    vector_context = state["vector_context"]
    relationships = state["graph_relationships"]

    if not vector_context and not relationships:
        return {"answer": "No relevant text or graph context was found."}

    prompt = f"""
You are a scientific research assistant.
Answer using only the evidence below.

Rules:
- Do not invent facts or relationships.
- Say so if evidence is insufficient.
- Cite text using [Paper ID: <id>, Chunk: <index>].
- Treat graph relationships as supporting context, not independent proof.

RETRIEVED PAPER PASSAGES:
{chr(10).join(vector_context) or "No relevant passages found."}

KNOWLEDGE GRAPH RELATIONSHIPS:
{chr(10).join(relationships) or "No matching graph relationships found."}

QUESTION:
{state["question"]}
"""

    response = qa_agent.llm.invoke(prompt)
    return {"answer": response.content}


def summarize_node(state: ResearchState) -> dict:
    """Summarize the existing processed research paper."""

    paper_id = state["paper_id"]
    paper_path = Path("data/processed") / f"{paper_id}.json"

    if not paper_path.exists():
        return {
            "answer": f"Processed paper not found: {paper_path}"
        }

    paper = json.loads(paper_path.read_text(encoding="utf-8"))

    summary = summarize_paper(paper)
    output_path = save_summary(summary)

    return {
        "answer": summary["summary"],
        "summary_path": str(output_path),
    }


def extract_node(state: ResearchState) -> dict:
    """Extract research entities and relationships from a paper."""

    paper_id = state["paper_id"]
    paper_path = Path("data/processed") / f"{paper_id}.json"

    if not paper_path.exists():
        return {
            "answer": f"Processed paper not found: {paper_path}"
        }

    paper = json.loads(
        paper_path.read_text(encoding="utf-8")
    )

    extraction = extract_knowledge(paper["full_text"])

    # Convert the Pydantic model into a JSON-compatible dictionary.
    extraction_data = extraction.model_dump()

    output_path = (
        Path("data/processed")
        / f"{paper_id}_extraction.json"
    )
    output_path.write_text(
        json.dumps(extraction_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "answer": json.dumps(
            extraction_data,
            indent=2,
            ensure_ascii=False,
        ),
        "extraction_path": str(output_path),
    }


def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("router", router_node)
    workflow.add_node("vector_retrieval", vector_retrieval_node)
    workflow.add_node("graph_retrieval", graph_retrieval_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("extract", extract_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_request,
        {
            "qa": "vector_retrieval",
            "summarize": "summarize",
            "extract": "extract",
        },
    )

    workflow.add_edge("vector_retrieval", "graph_retrieval")
    workflow.add_edge("graph_retrieval", "answer_generation")

    workflow.add_edge("answer_generation", END)
    workflow.add_edge("summarize", END)
    workflow.add_edge("extract", END)

    return workflow.compile()


research_graph = build_research_graph()
