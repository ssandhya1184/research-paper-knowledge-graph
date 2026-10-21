import base64
import json
from pathlib import Path

import gradio as gr

from research_assistant.workflows.research_graph import research_graph
from research_assistant.ui.graph_visualizer import visualize_graph


PROCESSED_DIR = Path("data/processed")
GRAPH_DIR = Path("data/graph")




def get_paper_choices():
    """Return readable paper titles with paper IDs as values."""

    metadata_path = PROCESSED_DIR / "papers.json"
    papers_by_id = {}

    if metadata_path.exists():
        papers = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        papers_by_id = {
            paper["paper_id"]: paper.get("title", "")
            for paper in papers
        }

    choices = []

    for graph_path in sorted(GRAPH_DIR.glob("*_graph.json")):
        paper_id = graph_path.name.removesuffix("_graph.json")

        processed_path = PROCESSED_DIR / f"{paper_id}.json"
        if not processed_path.exists():
            continue

        title = papers_by_id.get(paper_id, "")

        # Fall back to the title stored in the graph's Paper node.
        if not title:
            graph_data = json.loads(
                graph_path.read_text(encoding="utf-8")
            )

            for node in graph_data.get("nodes", []):
                if node.get("node_type") == "Paper":
                    title = node.get("name", "")
                    break

        title = title or paper_id
        choices.append((f"{title} ({paper_id})", paper_id))

    return choices




def run_task(paper_id, task, question):
    if not paper_id:
        return "Please select a paper.", "", ""

    if task == "Ask a question" and not question.strip():
        return "Please enter a question.", "", ""

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

    if task == "Ask a question":
        sources = result.get("sources", [])
        source_text = "\n".join(
            f"- Paper: {source['paper_id']}, Chunk: {source['chunk_index']}"
            for source in sources
        ) or "No sources returned."
    else:
        source_text = "Sources are available in the selected paper's processed data."

    graph_path = GRAPH_DIR / f"{paper_id}_graph.json"
    graph_html = ""

    if graph_path.exists():
        html_path = visualize_graph(
            graph_path,
            GRAPH_DIR / f"{paper_id}_graph.html",
        )

        html_content = html_path.read_text(encoding="utf-8")
        encoded_html = base64.b64encode(
            html_content.encode("utf-8")
        ).decode("utf-8")

        graph_html = f"""
        <iframe
            src="data:text/html;base64,{encoded_html}"
            width="100%"
            height="750"
            style="border: 1px solid #ddd; border-radius: 8px;"
        ></iframe>
        """

    return answer, source_text, graph_html


def update_question_visibility(task):
    return gr.update(
        visible=(task == "Ask a question")
    )


with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown("# Research Paper Assistant")
    gr.Markdown(
        "Ask questions, summarize papers, extract knowledge, "
        "and explore research knowledge graphs."
    )

    paper_dropdown = gr.Dropdown(
        choices=get_paper_choices(),
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
            visible=True,
        )

    task_dropdown.change(
        fn=update_question_visibility,
        inputs=task_dropdown,
        outputs=question_input,
    )

    

    run_button = gr.Button("Run", variant="primary")

    answer_output = gr.Markdown(label="Result")
    extraction_output = gr.JSON(label="Extracted Knowledge", visible=False)
    sources_output = gr.Textbox(label="Sources", lines=5)
    graph_output = gr.HTML(label="Knowledge graph")

    run_button.click(
        fn=run_task,
        inputs=[paper_dropdown, task_dropdown, question_input],
        outputs=[answer_output, sources_output, graph_output],
    )


if __name__ == "__main__":
    demo.launch()
