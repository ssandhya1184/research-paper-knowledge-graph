from research_assistant.agents.qa_agent import ResearchQAAgent

paper_id = "2610.22372v1"
graph_path = f"data/graph/{paper_id}_graph.json"

agent = ResearchQAAgent(graph_path=graph_path)

question = "What datasets are used in LIBERO-PeRM?"

relationships = agent.retrieve_graph_context(question)

print("Paper ID:", paper_id)
print("Graph relationships:")
for relationship in relationships:
    print(relationship)