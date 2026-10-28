from pathlib import Path

from research_assistant.evaluation.evaluator import (
    evaluate_paper,
    print_report,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

graph_path = ROOT_DIR / "data" / "graph" / "2610.22372v1_graph.json"
ground_truth_path = (
    ROOT_DIR
    / "data"
    / "processed"
    / "evaluation_ground_truth.json"
)

if __name__ == "__main__":
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {ground_truth_path}"
        )

    report = evaluate_paper(graph_path, ground_truth_path)
    print_report(report)
