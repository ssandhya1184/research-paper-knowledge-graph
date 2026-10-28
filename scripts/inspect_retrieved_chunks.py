from pathlib import Path

from research_assistant.storage.vector_store import VectorStore

paper_id = "2610.22372v1"

queries = [
    "How does mixed training affect no-preference retention?",
    #"What are the challenges in zero-shot personalization?",
]

vector_store = VectorStore()

for query in queries:
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print("=" * 80)

    results = vector_store.search(
        query=query,
        top_k=5,
        paper_id=paper_id,
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for rank, (chunk_id, document, metadata) in enumerate(
        zip(ids, documents, metadatas), start=1
    ):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Metadata: {metadata}")
        print(f"Full text:\n{document}")
        for chunk_id, document in zip(
            results["ids"][0],
            results["documents"][0],
        )
            print(f"\nChunk ID: {chunk_id}")
            print(f"Full text:\n{document}")
