
import json
from pathlib import Path

import networkx as nx

graph_file = Path("data/graph/2610.22343v1_graph.json")

data = json.loads(graph_file.read_text(encoding="utf-8"))
graph = nx.node_link_graph(data, directed=True, multigraph=True)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Connectivity ignoring edge direction
undirected = graph.to_undirected()
components = list(nx.connected_components(undirected))

print(f"Connected components: {len(components)}")
print(f"Largest component size: {max(map(len, components), default=0)}")

isolated = list(nx.isolates(undirected))
print(f"Isolated nodes: {len(isolated)}")

for node in isolated[:10]:
    print("Isolated:", node, graph.nodes[node].get("name"))
