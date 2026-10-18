
import json
from pathlib import Path

from research_assistant.agents.knowledge_extractor import create_llm


def summarize_paper(paper: dict) -> dict:
    """Generate a structured research paper summary using Gemini."""

    llm = create_llm()

    prompt = f"""
    Summarize this scientific research paper for a researcher.

    Return these sections:
    1. Objective: What problem does the paper address?
    2. Methodology: What approach or model is proposed?
    3. Datasets: What datasets are used?
    4. Key Findings: What are the main results?
    5. Contributions: What is new about this work?
    6. Limitations: What limitations are stated in the paper?

    Use only information supported by the paper.
    Do not invent results or limitations.
    If a section is not supported by the text, state
    "Not specified in the provided text."

    Paper title: {paper.get("paper_id", "Unknown")}

    Paper text:
    {paper["full_text"][:20000]}
    """

    response = llm.invoke(prompt)

    return {
        "paper_id": paper["paper_id"],
        "summary": response.content,
    }


def save_summary(summary: dict) -> Path:
    """Persist a generated summary as JSON."""

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{summary["paper_id"]}_summary.json'
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path
