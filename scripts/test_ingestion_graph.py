
from research_assistant.workflows.ingestion_graph import ingestion_graph


result = ingestion_graph.invoke(
    {
        "query": "cat:cs.LG",
        "max_results": 1,
        "papers": [],
        "processed_papers": [],
        "total_chunks": 0,
        "graph_paths": [],
        "status": "",
    }
)

print("\nProcessed papers:", result["processed_papers"])
print("Total chunks:", result["total_chunks"])
print("Knowledge graphs:", result["graph_paths"])
