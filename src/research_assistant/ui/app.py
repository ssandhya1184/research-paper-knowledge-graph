
import json
from pathlib import Path

import gradio as gr

from research_assistant.workflows.research_graph import research_graph
from research_assistant.ui.graph_visualizer import visualize_graph


PROCESSED_DIR = Path("data/processed")
GRAPH_DIR = Path("data/graph")


def get_paper_ids():
    """List papers with processed text available."""
    return sorted(
        path.stem
        for path in PROCESSED_DIR.glob("*.json")
        if not path.stem.endswith(("_summary", "_extraction"))
        and path.stem != "papers"
        and (GRAPH_DIR / f"{path.stem}_graph.json").exists()
    )


def run_task(paper_id, task, question):
    if not paper_id:
        return "Please select a paper.", "", None

    if task == "Ask a question" and not question.strip():
        return "Please enter a question.", "", None

    request_text = {
        "Ask a question": question,
        "Summarize paper": "Summarize this paper",
        "Extract knowledge": "Extract entities and research knowledge",
    }[task]

    result = research_graph.invoke(
        {
            "question": request_text,
            "paper_id": paper_id,
            "request_type": "",
            "vector_context": [],
            "sources": [],
            "graph_relationships": [],
            "answer": "",
            "summary_path": "",
            "extraction_path": "",
        }
    )

    answer = result.get("answer", "No result returned.")

    sources = result.get("sources", [])
    source_text = "\n".join(
        f"- Paper: {source['paper_id']}, Chunk: {source['chunk_index']}"
        for source in sources
    )

    graph_path = GRAPH_DIR / f"{paper_id}_graph.json"
    graph_html = None

    if graph_path.exists():
        graph_html = str(
            visualize_graph(
                graph_path,
                GRAPH_DIR / f"{paper_id}_graph.html",
            )
        )

    return answer, source_text, graph_html


with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown("# Research Paper Assistant")
    gr.Markdown(
        "Ask questions, summarize papers, extract knowledge, "
        "and explore research knowledge graphs."
    )

    paper_dropdown = gr.Dropdown(
        choices=get_paper_ids(),
        label="Select research paper",
        value=None,
    )

    task_dropdown = gr.Radio(
        choices=[
            "Ask a question",
            "Summarize paper",
            "Extract knowledge",
        ],
        value="Ask a question",
        label="Task",
    )

    question_input = gr.Textbox(
        label="Your question",
        placeholder="e.g. What datasets are used in this paper?",
        lines=3,
    )

    run_button = gr.Button("Run", variant="primary")

    answer_output = gr.Markdown(label="Result")
    sources_output = gr.Textbox(label="Sources", lines=5)
    graph_output = gr.File(label="Knowledge graph (HTML)")

    run_button.click(
        fn=run_task,
        inputs=[paper_dropdown, task_dropdown, question_input],
        outputs=[answer_output, sources_output, graph_output],
    )


if __name__ == "__main__":
    demo.launch()
