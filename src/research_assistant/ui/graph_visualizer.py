
import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network


def visualize_graph(
    graph_json: str | Path,
    output_html: str | Path = "data/graph/knowledge_graph.html",
) -> Path:
    """Create an interactive HTML visualization of a saved graph."""

    graph_json = Path(graph_json)
    output_html = Path(output_html)

    data = json.loads(graph_json.read_text(encoding="utf-8"))
    graph = nx.node_link_graph(data, directed=True, multigraph=True)

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
    )

    colors = {
        "Paper": "#F4A261",
        "Method": "#4C78A8",
        "Dataset": "#54A24B",
        "Metric": "#E45756",
        "Concept": "#B279A2",
        "Finding": "#72B7B2",
    }

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "Unknown")
        name = attrs.get("name", node_id)
        description = attrs.get("description", "")

        net.add_node(
            node_id,
            label=name,
            title=f"{node_type}: {name}\n{description}",
            group=node_type,
            color=colors.get(node_type, "#999999"),
        )

    for source, target, attrs in graph.edges(data=True):
        net.add_edge(
            source,
            target,
            label=attrs.get("relation", ""),
            title=attrs.get("relation", ""),
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_html), notebook=False)

    print(f"Visualization saved to: {output_html.resolve()}")
    return output_html


if __name__ == "__main__":
    visualize_graph("data/graph/2610.22343v1_graph.json")
