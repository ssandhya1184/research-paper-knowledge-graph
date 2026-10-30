
import json
import re
from pathlib import Path

import networkx as nx

from research_assistant.agents.knowledge_extractor import create_llm
from research_assistant.storage.vector_store import VectorStore


GRAPH_PATH = Path("data/graph/2610.22343v1_graph.json")


def normalize_text(text: str) -> str:
    """Normalize text for entity matching."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class ResearchQAAgent:
    """Answer research questions using vector search and graph context."""

    def __init__(self, graph_path: str | Path = GRAPH_PATH):
        self.vector_store = VectorStore()
        self.llm = create_llm()

        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {graph_path}")

        graph_data = json.loads(
            graph_path.read_text(encoding="utf-8")
        )
        self.graph = nx.node_link_graph(
            graph_data,
            directed=True,
            multigraph=True,
        )

    def retrieve_graph_context(
        self,
        question: str,
        max_nodes: int = 10,
    ) -> list[str]:
        """Find entities mentioned in the question and their relationships."""

        normalized_question = normalize_text(question)
        matched_nodes = []

        for node_id, attributes in self.graph.nodes(data=True):
            name = attributes.get("name", "")
            normalized_name = normalize_text(name)

            if (
                normalized_name
                and normalized_name in normalized_question
            ):
                matched_nodes.append(node_id)

        relationships = []

        for node_id in matched_nodes[:max_nodes]:
            name = self.graph.nodes[node_id].get("name", node_id)

            # Outgoing relationships
            for _, target, attrs in self.graph.out_edges(
                node_id, data=True
            ):
                target_name = self.graph.nodes[target].get(
                    "name", target
                )
                relation = attrs.get("relation", "related to")

                relationships.append(
                    f"{name} --[{relation}]--> {target_name}"
                )

            # Incoming relationships
            for source, _, attrs in self.graph.in_edges(
                node_id, data=True
            ):
                source_name = self.graph.nodes[source].get(
                    "name", source
                )
                relation = attrs.get("relation", "related to")

                relationships.append(
                    f"{source_name} --[{relation}]--> {name}"
                )

        # Remove duplicate relationship strings while preserving order.
        return list(dict.fromkeys(relationships))

    
def answer(
    self,
    question: str,
    top_k: int = 3,
    paper_id: str | None = None,
    max_distance: float = 0.55,
) -> dict:
    """Answer using only sufficiently similar chunks from the selected paper."""

    print(f"[QA] Answering question: '{question}' (top_k={top_k}, paper_id={paper_id})")
    results = self.vector_store.search(
        question,
        top_k=top_k,
        paper_id=paper_id,
    )
    print(f"[QA] Retrieved {len(results.get('documents', [[]])[0])} chunks from vector store.")

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    vector_context = []
    sources = []

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        if distance is None or distance > max_distance:
            continue

        source_paper_id = metadata.get("paper_id", "Unknown")

        vector_context.append(
            f"Paper ID: {source_paper_id}\n"
            f"Chunk: {metadata.get('chunk_index', 'Unknown')}\n"
            f"Content:\n{document}"
        )

        source = {
            "paper_id": source_paper_id,
            "chunk_index": metadata.get("chunk_index", "Unknown"),
            "distance": float(distance),
        }

        if source not in sources:
            sources.append(source)

    graph_relationships = self.retrieve_graph_context(question)

    if not vector_context and not graph_relationships:
        return {
            "question": question,
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the selected paper to answer this question."
            ),
            "sources": [],
            "graph_relationships": [],
        }

    prompt = f"""
    You are a scientific research assistant.

    Answer the question using the retrieved paper passages
    and knowledge graph relationships below.

    Rules:
    - Use only evidence supported by the provided context.
    - Do not invent facts or relationships.
    - If evidence is insufficient, say so.
    - Cite textual evidence using [Paper ID: <id>, Chunk: <index>].
    - Treat graph relationships as additional evidence, not proof
      of facts beyond the relationship recorded.
    - Avoid repetitive citations.

    RETRIEVED PAPER PASSAGES:
    {chr(10).join(vector_context) or "No relevant passages found."}

    KNOWLEDGE GRAPH RELATIONSHIPS:
    {chr(10).join(graph_relationships) or "No matching graph relationships found."}

    QUESTION:
    {question}
    """
    print("[QA] Sending prompt to Gemini...")
    response = self.llm.invoke(prompt)
    print("[QA] Gemini response received.")

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "graph_relationships": graph_relationships,
    }

