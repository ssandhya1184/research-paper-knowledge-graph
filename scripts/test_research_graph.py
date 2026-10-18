
from research_assistant.workflows.research_graph import research_graph

"""
questions = [
    "How does GeoState support long-horizon geometry forecasting?",
    "Summarize this paper",
    "Extract entities and research knowledge",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    result = research_graph.invoke(
        {
            "question": question,
            "request_type": "",
            "vector_context": [],
            "sources": [],
            "graph_relationships": [],
            "answer": "",
        }
    )

    print("Answer:", result["answer"])

    if result["sources"]:
        print("Sources:", result["sources"])

        """

result = research_graph.invoke(
    {
        "question": "What datasets are used in LIBERO-PeRM?",
        "paper_id": "2610.22372v1",
        "request_type": "",
        "vector_context": [],
        "sources": [],
        "graph_relationships": [],
        "answer": "",
        "summary_path": "",
        "extraction_path": "",
    }
)
print("\nRequest type:", result["request_type"])
print("\nAnswer:\n", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(source)

print("\nGraph relationships:")
for relationship in result["graph_relationships"]:
    print(relationship)

