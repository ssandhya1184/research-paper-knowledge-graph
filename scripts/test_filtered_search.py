
from research_assistant.storage.vector_store import VectorStore

store = VectorStore()

result = store.search(
    query="What datasets are used in LIBERO-PeRM?",
    top_k=3,
    paper_id="2610.22372v1",
)

documents = result["documents"][0]
metadatas = result["metadatas"][0]

print(f"Retrieved chunks: {len(documents)}")

for metadata in metadatas:
    print(metadata)
    assert metadata["paper_id"] == "2610.22372v1"

print("Paper-specific retrieval passed!")
