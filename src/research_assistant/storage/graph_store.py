
import json
import re
from pathlib import Path

import networkx as nx


GRAPH_DIR = Path("data/graph")


def normalize_name(name: str) -> str:
    """Create a stable node identifier."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return name.strip("_")


class KnowledgeGraphStore:
    """Build and persist a research paper knowledge graph."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_paper(self, paper_id: str, title: str = "") -> None:
        self.graph.add_node(
            f"paper::{paper_id}",
            name=title or paper_id,
            node_type="Paper",
            paper_id=paper_id,
        )

    def add_entities(self, entities: list[dict], paper_id: str) -> None:
        paper_node = f"paper::{paper_id}"
        print("entities->",entities)
        for entity in entities:
            name = entity["name"].strip()
            print("name:",name  )
            entity_type = entity["entity_type"]
            print("entity_type:",entity_type)
            node_id = f"{entity_type.lower()}::{normalize_name(name)}"
            print("node_id:",node_id)

            self.graph.add_node(
                node_id,
                name=name,
                node_type=entity_type,
                description=entity.get("description", ""),
            )

            # Link every extracted entity to its source paper.
            self.graph.add_edge(
                paper_node,
                node_id,
                relation="contains",
            )

    def add_relationships(
        self,
        relationships: list[dict],
        entities: list[dict],
        aliases: dict[str, str] | None = None,
    ) -> None:
        aliases = aliases or {}

        # Map names to node IDs. Type is included in each node ID.
        name_to_ids = {}

        for entity in entities:
            name = entity["name"].strip()
            canonical = aliases.get(name, name)
            node_id = (
                f"{entity['entity_type'].lower()}::"
                f"{normalize_name(canonical)}"
            )
            name_to_ids.setdefault(canonical, []).append(node_id)
            name_to_ids.setdefault(name, []).append(node_id)

        for relationship in relationships:
            source = aliases.get(
                relationship["source"],
                relationship["source"],
            )
            target = aliases.get(
                relationship["target"],
                relationship["target"],
            )

            source_ids = list(dict.fromkeys(name_to_ids.get(source, [])))
            target_ids = list(dict.fromkeys(name_to_ids.get(target, [])))

            # Avoid guessing when a name maps to multiple entity types.
            if len(source_ids) != 1 or len(target_ids) != 1:
                print(
                    f"Skipping ambiguous or unresolved relationship: "
                    f"{source} -> {target}"
                )
                continue

            self.graph.add_edge(
                source_ids[0],
                target_ids[0],
                relation=relationship["relation"],
            )

    def save(self, filename: str = "research_graph.json") -> Path:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        output_path = GRAPH_DIR / filename

        graph_data = nx.node_link_data(self.graph)

        output_path.write_text(
            json.dumps(graph_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Graph saved to: {output_path}")
        return output_path

    def print_summary(self) -> None:
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print(f"Is directed: {self.graph.is_directed()}")
