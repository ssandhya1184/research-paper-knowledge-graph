import json
from pathlib import Path

from research_assistant.storage.vector_store import VectorStore
from research_assistant.evaluation.metrics import (
    evaluate_retrieval_relevance,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = (
    ROOT_DIR / "data" / "processed" / "retrieval_ground_truth.json"
)


def main():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    paper_id = ground_truth["paper_id"]
    vector_store = VectorStore()

    results = []

    for item in ground_truth["queries"]:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        print(f"\nPaper ID: {paper_id}")
        print(f"Query: {query}")
        print("Top K: 3")
        # Retrieve the top 3 chunks from the selected paper.
        retrieved = vector_store.search(
            query=query,
            top_k=3,
            paper_id=paper_id,
        )

        retrieved_ids = retrieved.get("ids", [[]])[0] 
        print("Retrieved chunk IDs:", retrieved.get("ids", [[]])[0])

        # Prevent placeholder IDs from producing misleading metrics.
        if any(chunk_id.startswith("REPLACE_WITH_") for chunk_id in relevant_ids):
            print(f"\nSkipping unlabeled query: {query}")
            print("Replace placeholder IDs with verified chunk IDs.")
            continue

        metrics = evaluate_retrieval_relevance(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=3,
        )

        results.append(metrics)

        print(f"\nQuery: {query}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Metrics: {metrics}")

    if results:
        average_precision = sum(
            result["precision_at_k"] for result in results
        ) / len(results)

        average_recall = sum(
            result["recall_at_k"] for result in results
        ) / len(results)

        print("\n=== Retrieval Evaluation Summary ===")
        print(f"Evaluated queries: {len(results)}")
        print(f"Mean Precision@3: {average_precision:.4f}")
        print(f"Mean Recall@3: {average_recall:.4f}")
    else:
        print("\nNo queries evaluated. Add verified relevant chunk IDs.")


if __name__ == "__main__":
    main()
