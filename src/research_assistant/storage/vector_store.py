
import chromadb
from pathlib import Path

from research_assistant.processing.embeddings import EmbeddingGenerator


DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "research_papers"


class VectorStore:
    """Store and search paper chunks in ChromaDB."""

    def __init__(self):
        DB_PATH.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = EmbeddingGenerator()

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "paper_id": chunk["paper_id"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in chunks
            ],
        )

        print(f"Stored {len(chunks)} chunks in ChromaDB.")

    
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        paper_id: str | None = None,
    ) -> dict:
        print(f"[SEARCH] Searching for: '{query}' (top_k={top_k}, paper_id={paper_id})")
        query_embedding = self.embedder.embed_texts([query])[0]
        print(f"[SEARCH] Query embedding generated for: '{query}'")

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        # Filter results by paper when a paper ID is provided.
        if paper_id:
            query_args["where"] = {"paper_id": paper_id}

        return self.collection.query(**query_args)


