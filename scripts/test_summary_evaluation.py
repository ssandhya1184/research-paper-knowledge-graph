import json
from pathlib import Path

from research_assistant.evaluation.summary_evaluator import (
    evaluate_summary,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT_DIR / "data" / "processed" / "evaluation_reference_summary.json"
)
GENERATED_PATH = (
    ROOT_DIR / "data" / "processed" / "2610.22343v1_summary.json"
)


def main():
    with open(REFERENCE_PATH, "r", encoding="utf-8") as file:
        reference_data = json.load(file)

    with open(GENERATED_PATH, "r", encoding="utf-8") as file:
        generated_data = json.load(file)

    reference = reference_data["reference_summary"]
    generated = generated_data["summary"]

    metrics = evaluate_summary(generated, reference)

    print("\n=== Summary Quality Evaluation ===")
    print(f"Paper ID: {reference_data['paper_id']}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
