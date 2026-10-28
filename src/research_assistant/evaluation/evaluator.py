import json
from pathlib import Path

import networkx as nx

from research_assistant.evaluation.metrics import (
    evaluate_entity_extraction,
    evaluate_graph_connectivity,
)


def load_graph(graph_path: str | Path) -> nx.MultiDiGraph:
    """Load a saved NetworkX graph from node-link JSON."""
    with open(graph_path, "r", encoding="utf-8") as file:
        graph_data = json.load(file)

    # Detect the edge key used in the saved JSON.
    if "edges" in graph_data:
        edge_key = "edges"
    elif "links" in graph_data:
        edge_key = "links"
    else:
        raise ValueError(
            f"Could not identify the edge key in {graph_path}. "
            f"Available keys: {list(graph_data.keys())}"
        )

    return nx.node_link_graph(graph_data, edges=edge_key)



def evaluate_paper(
    graph_path: str | Path,
    ground_truth_path: str | Path,
) -> dict:
    """Evaluate entity extraction and connectivity for one paper."""
    graph = load_graph(graph_path)

    with open(ground_truth_path, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    # Ground-truth format: {"entities": ["Entity A", "Entity B"]}
    expected_entities = ground_truth["entities"]

    # Exclude paper nodes and compare research entity names.
    extracted_entities = [
        str(data.get("name", node_id.split("::", 1)[-1]))
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type", data.get("type", "")) != "paper"
        and not str(node_id).startswith("paper::")
    ]

    print("\n=== Extracted Entity Names ===")
    for entity in sorted(extracted_entities):
        print(f"- {entity}")

    print("\n=== Expected Entity Names ===")
    for entity in ground_truth["entities"]:
        print(f"- {entity}")
        
    return {
        "entity_extraction": evaluate_entity_extraction(
            expected_entities, extracted_entities
        ),
        "graph_connectivity": evaluate_graph_connectivity(graph),
    }


def print_report(report: dict) -> None:
    """Print a readable evaluation report."""
    print("\n=== Entity Extraction Evaluation ===")
    for metric, value in report["entity_extraction"].items():
        print(f"{metric}: {value}")

    print("\n=== Graph Connectivity Evaluation ===")
    for metric, value in report["graph_connectivity"].items():
        print(f"{metric}: {value}")
