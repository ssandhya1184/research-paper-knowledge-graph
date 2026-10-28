from collections.abc import Iterable

import networkx as nx


def normalize_entity(name: str) -> str:
    """Normalize entity names for consistent comparisons."""
    return " ".join(name.casefold().strip().split())


def evaluate_entity_extraction(
    expected_entities: Iterable[str],
    extracted_entities: Iterable[str],
) -> dict:
    """
    Evaluate extracted entity names against ground-truth entity names.

    Exact normalized-name matching is used in this initial version.
    """
    expected = {normalize_entity(name) for name in expected_entities}
    extracted = {normalize_entity(name) for name in extracted_entities}

    true_positives = len(expected & extracted)
    false_positives = len(extracted - expected)
    false_negatives = len(expected - extracted)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "expected_entities": len(expected),
        "extracted_entities": len(extracted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
    }


def evaluate_graph_connectivity(graph: nx.Graph) -> dict:
    """
    Evaluate connectivity and structural properties of a knowledge graph.
    Connectivity is measured on the undirected version of the graph.
    """
    undirected_graph = graph.to_undirected()

    node_count = undirected_graph.number_of_nodes()
    edge_count = undirected_graph.number_of_edges()

    if node_count == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "connected_components": 0,
            "largest_component_ratio": 0.0,
            "isolated_nodes": 0,
            "density": 0.0,
        }

    components = list(nx.connected_components(undirected_graph))
    largest_component_size = max(len(component) for component in components)

    return {
        "nodes": node_count,
        "edges": edge_count,
        "connected_components": len(components),
        "largest_component_ratio": round(
            largest_component_size / node_count, 4
        ),
        "isolated_nodes": len(list(nx.isolates(undirected_graph))),
        "density": round(nx.density(undirected_graph), 4),
    }



def evaluate_retrieval_relevance(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 3,
) -> dict:
    """Calculate Precision@K and Recall@K for one query."""
    retrieved = retrieved_ids[:k]
    relevant = set(relevant_ids)
    retrieved_set = set(retrieved)

    hits = len(retrieved_set & relevant)

    precision_at_k = hits / k if k > 0 else 0.0
    recall_at_k = hits / len(relevant) if relevant else 0.0

    return {
        "k": k,
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant),
        "relevant_retrieved": hits,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
    }
