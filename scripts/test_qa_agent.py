
from research_assistant.agents.qa_agent import ResearchQAAgent

agent = ResearchQAAgent()

question = "What problem does GeoState solve in 3D world modeling?"
result = agent.answer(question, top_k=3)

print("Question:", result["question"])
print("\nAnswer:\n", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(source)
