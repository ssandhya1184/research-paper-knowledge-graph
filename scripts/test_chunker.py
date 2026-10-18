
from research_assistant.agents.qa_agent import ResearchQAAgent

agent = ResearchQAAgent()

question = "How does GeoState support long-horizon geometry forecasting?"
result = agent.answer(question, top_k=3)

print("\nQuestion:")
print(result["question"])

print("\nAnswer:")
print(result["answer"])

print("\nText sources:")
for source in result["sources"]:
    print(source)

print("\nKnowledge graph relationships:")
for relationship in result["graph_relationships"]:
    print(relationship)
