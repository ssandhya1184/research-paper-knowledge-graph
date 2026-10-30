
import json
from pathlib import Path

import networkx as nx


from research_assistant.evaluation.metrics import (
    evaluate_entity_extraction,
    evaluate_graph_connectivity,
    aggregate_entity_metrics,
)


def load_graph(graph_path: str | Path) -> nx.MultiDiGraph:
    """Load a saved NetworkX graph from node-link JSON."""
    with open(graph_path, "r", encoding="utf-8") as file:
        graph_data = json.load(file)

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
    expected_entities: list[str],
) -> dict:
    """Evaluate entity extraction and connectivity for one paper."""
    graph = load_graph(graph_path)

    extracted_entities = [
        str(data.get("name", node_id.split("::", 1)[-1]))
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type", data.get("type", "")) != "paper"
        and not str(node_id).startswith("paper::")
    ]

    return {
        "entity_extraction": evaluate_entity_extraction(
            expected_entities, extracted_entities
        ),
        "graph_connectivity": evaluate_graph_connectivity(graph),
    }


def evaluate_all_papers(
    graph_dir: str | Path,
    ground_truth_path: str | Path,
) -> dict:
    """Evaluate all papers listed in multi-paper ground truth."""
    graph_dir = Path(graph_dir)

    with open(ground_truth_path, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    if "papers" not in ground_truth:
        raise ValueError(
            "Expected multi-paper ground truth with a 'papers' list."
        )

    results = {}

    for paper in ground_truth["papers"]:
        paper_id = paper["paper_id"]
        expected_entities = paper["entities"]

        graph_path = graph_dir / f"{paper_id}_graph.json"

        print(f"\n{'=' * 60}")
        print(f"Evaluating Paper: {paper_id}")
        print(f"{'=' * 60}")

        if not graph_path.exists():
            print(f"Graph file not found: {graph_path}")
            continue

        report = evaluate_paper(graph_path, expected_entities)
        results[paper_id] = report

        print_report(report, paper_id)

    if results:
        overall = aggregate_entity_metrics(results)

        print("\n" + "=" * 60)
        print("OVERALL ENTITY EXTRACTION EVALUATION")
        print("=" * 60)

        for metric, value in overall.items():
            print(f"{metric}: {value}")

        results["overall"] = {
            "entity_extraction": overall
        }
        
    return results


def print_report(report: dict, paper_id: str) -> None:
    """Print evaluation results for one paper."""
    print(f"\n=== Entity Extraction: {paper_id} ===")
    for metric, value in report["entity_extraction"].items():
        print(f"{metric}: {value}")

    print(f"\n=== Graph Connectivity: {paper_id} ===")
    for metric, value in report["graph_connectivity"].items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    evaluate_all_papers(
        graph_dir=project_root / "data" / "graph",
        ground_truth_path=(
            project_root
            / "data"
            / "processed"
            / "evaluation_ground_truth.json"
        ),
    )
