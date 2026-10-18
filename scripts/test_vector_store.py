
import json
from pathlib import Path

from research_assistant.storage.vector_store import VectorStore


chunk_files = list(Path("data/chunks").glob("*_chunks.json"))

if not chunk_files:
    raise FileNotFoundError("No chunk files found in data/chunks/")

chunks = json.loads(
    chunk_files[0].read_text(encoding="utf-8")
)

store = VectorStore()

# Store the chunk embeddings
store.add_chunks(chunks)

print(f"Total chunks in database: {store.collection.count()}")

# Test semantic search
query = "How do world models predict the future?"
results = store.search(query, top_k=3)

print(f"\nQuery: {query}")

for i, text in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(f"\nResult {i + 1}")
    print(f"Paper ID: {metadata['paper_id']}")
    print(f"Distance: {distance:.4f}")
    print(text[:300])
