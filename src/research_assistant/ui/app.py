
import base64
import json
from pathlib import Path
import threading
import time
import gradio as gr
from langgraph.func import task
import networkx as nx

from research_assistant.evaluation.metrics import (
    evaluate_entity_extraction,
    evaluate_graph_connectivity,
    evaluate_retrieval_relevance,
)
from research_assistant.ui.graph_visualizer import visualize_graph
from research_assistant.workflows.research_graph import research_graph
from research_assistant.ingestion.arxiv_client import (
    search_arxiv,
    load_paper_metadata,
    get_ingestion_status,
)
from research_assistant.workflows.ingestion_graph import (
    ingest_selected_paper,
)

PROCESSED_DIR = Path("data/processed")
GRAPH_DIR = Path("data/graph")


# ---------------------------------------------------------
# Paper selection
# ---------------------------------------------------------


def search_papers_ui(query: str, max_results: int):
    """Search arXiv and prepare compact choices for the UI."""

    if not query or not query.strip():
        return [], gr.update(choices=[], value=None), "Enter a search query."

    try:
        papers = search_arxiv(
            query=query.strip(),
            max_results=int(max_results),
        )

        choices = [
            (
                f"{paper['title']} ({paper['paper_id']})",
                paper["paper_id"],
            )
            for paper in papers
        ]

        message = (
            f"Found {len(papers)} paper(s). Select a paper to preview it."
            if papers
            else "No papers found. Try different keywords."
        )

        return papers, gr.update(choices=choices, value=None), message

    except Exception as exc:
        return [], gr.update(choices=[], value=None), (
            f"Search failed: {exc}"
        )



def preview_paper(paper_id: str, search_results: list):
    """Show concise details for the selected arXiv paper."""

    if not paper_id or not search_results:
        return "Select a paper to view its details."

    paper = next(
        (
            item for item in search_results
            if item["paper_id"] == paper_id
        ),
        None,
    )

    if not paper:
        return "Paper details are unavailable."

    abstract = paper.get("abstract", "No abstract available.")
    words = abstract.split()
    concise_abstract = " ".join(words[:45])

    if len(words) > 45:
        concise_abstract += "..."

    return (
        f"**{paper['title']}**\n\n"
        f"`{paper_id}`\n\n"
        f"**Published:** {paper.get('published', 'Unknown')[:10]}\n\n"
        f"**Abstract preview:** {concise_abstract}"
    )



def ingest_paper_ui(paper_id: str, search_results: list):
    """Ingest the selected arXiv search result."""

    if not paper_id or not search_results:
        return (
            "Select a search result first.",
            gr.update(),
        )

    paper = next(
        (
            item for item in search_results
            if item["paper_id"] == paper_id
        ),
        None,
    )

    if not paper:
        return "Selected paper details could not be found.", gr.update()

    try:
        status = get_ingestion_status(paper_id)

        if status == "Ingested":
            return (
                f"Paper **{paper_id}** is already ingested.",
                gr.update(
                    choices=get_paper_choices(),
                    value=paper_id,
                ),
            )

        result = ingest_selected_paper(paper)

        return (
            f"**{result['status']}**\n\n"
            f"Paper: {result['title']}\n\n"
            f"Chunks indexed: {result['chunks']}",
            gr.update(
                choices=get_paper_choices(),
                value="",
            ),
        )

    except Exception as exc:
        return f"Ingestion failed: {exc}", gr.update()



def get_paper_choices():
    """Return readable paper titles with paper IDs as values."""

    metadata_path = PROCESSED_DIR / "papers.json"
    papers_by_id = {}

    if metadata_path.exists():
        papers = json.loads(metadata_path.read_text(encoding="utf-8"))
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

        if not title:
            graph_data = json.loads(
                graph_path.read_text(encoding="utf-8")
            )

            for node in graph_data.get("nodes", []):
                if node.get("node_type", "").lower() == "paper":
                    title = node.get("name", "")
                    break

        title = title or paper_id
        choices.append((f"{title} ({paper_id})", paper_id))

    return [("-- Select a paper --", "")] + choices


# ---------------------------------------------------------
# Extraction display helpers
# ---------------------------------------------------------

def load_extraction(paper_id, result):
    """Load the extraction JSON produced by the workflow."""

    extraction_path = result.get("extraction_path", "")
    candidates = []

    if extraction_path:
        candidates.append(Path(extraction_path))

    candidates.append(
        PROCESSED_DIR / f"{paper_id}_extraction.json"
    )

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict) and "extraction" in data:
                data = data["extraction"]

            return data

    return {}


def to_entity_rows(items):
    """Convert extracted entities into table rows."""

    rows = []

    for item in items or []:
        if isinstance(item, dict):
            rows.append([
                item.get("name", ""),
                item.get("entity_type", ""),
                item.get("description", ""),
            ])
        else:
            rows.append([str(item), "", ""])

    return rows


def to_relationship_rows(items):
    """Convert relationships into source-relation-target rows."""

    rows = []

    for item in items or []:
        if isinstance(item, dict):
            rows.append([
                item.get("source", ""),
                item.get("relation", item.get("relationship", "")),
                item.get("target", ""),
            ])

    return rows


# ---------------------------------------------------------
# Knowledge graph helpers
# ---------------------------------------------------------


def get_graph_html(paper_id):
    """Generate HTML for display and download."""

    if not paper_id:
        return (
            "Select a paper to view its knowledge graph.",
            None,
        )

    graph_path = GRAPH_DIR / f"{paper_id}_graph.json"

    if not graph_path.exists():
        return (
            f"Knowledge graph not found for paper: {paper_id}",
            None,
        )

    html_path = visualize_graph(
        graph_path,
        GRAPH_DIR / f"{paper_id}_graph.html",
    )

    html_content = html_path.read_text(encoding="utf-8")

    encoded_html = base64.b64encode(
        html_content.encode("utf-8")
    ).decode("utf-8")

    iframe_html = f"""
    <iframe
        src="data:text/html;base64,{encoded_html}"
        width="100%"
        height="750"
        style="border: 1px solid #ddd; border-radius: 8px;"
    ></iframe>
    """

    return iframe_html, str(html_path)



def evaluate_graph_quality(paper_id):
    """Evaluate structural quality of the selected paper's graph."""

    graph_path = GRAPH_DIR / f"{paper_id}_graph.json"

    if not graph_path.exists():
        return "Graph evaluation unavailable: graph file not found."

    try:
        graph_data = json.loads(
            graph_path.read_text(encoding="utf-8")
        )

        edge_key = (
            "edges" if "edges" in graph_data else "links"
        )

        graph = nx.node_link_graph(
            graph_data,
            directed=True,
            multigraph=True,
            edges=edge_key,
        )

        metrics = evaluate_graph_connectivity(graph)

        return "\n".join([
            f"### Knowledge Graph Quality — `{paper_id}`",
            f"- Nodes: {metrics['nodes']}",
            f"- Edges: {metrics['edges']}",
            f"- Connected components: "
            f"{metrics['connected_components']}",
            f"- Largest component ratio: "
            f"{metrics['largest_component_ratio']:.4f}",
            f"- Isolated nodes: {metrics['isolated_nodes']}",
            f"- Graph density: {metrics['density']:.4f}",
            "",
            "*These metrics measure graph structure, not factual accuracy.*",
        ])

    except (ValueError, TypeError, KeyError) as exc:
        return f"Graph evaluation failed: {exc}"



def update_selected_paper(paper_id):
    """Refresh selected paper and clear stale outputs."""

    if paper_id:
        graph_html, graph_file_path = get_graph_html(paper_id)
        graph_eval = evaluate_graph_quality(paper_id)
        entity_eval = evaluate_entity_extraction_quality(paper_id)
    else:
        graph_html = (
            "<p>Select a paper to view its knowledge graph.</p>"
        )
        graph_file_path = None
        graph_eval = "Select a paper to view graph metrics."
        entity_eval = "Select a paper to view entity metrics."

    return (
        graph_html,
        graph_file_path,
        graph_eval,
        entity_eval,
        "",                         # answer_output
        "",                         # answer_evaluation_output
        "",                         # sources_output
        "Select a paper and run a question to view retrieval metrics.",
        [], [], [], [], [], [],     # extraction tables
        gr.update(visible=False),   # extraction_results
        gr.update(visible=False),   # extraction_section
        gr.update(visible=False),   # qa_results_row
        "",                         # summary_output
        gr.update(visible=False),   # summary_results_section
        "",                         # progress_status
    )


def evaluate_entity_extraction_quality(paper_id):
    """Evaluate graph entities against the selected paper's ground truth."""

    graph_path = GRAPH_DIR / f"{paper_id}_graph.json"
    ground_truth_path = PROCESSED_DIR / "evaluation_ground_truth.json"

    if not graph_path.exists():
        return "Entity evaluation unavailable: graph file not found."

    if not ground_truth_path.exists():
        return "Entity evaluation unavailable: ground-truth file not found."

    try:
        ground_truth_data = json.loads(
            ground_truth_path.read_text(encoding="utf-8")
        )

        # Support the new multi-paper format and the legacy format.
        if "papers" in ground_truth_data:
            paper_ground_truth = next(
                (
                    item
                    for item in ground_truth_data["papers"]
                    if item.get("paper_id") == paper_id
                ),
                None,
            )
        else:
            paper_ground_truth = (
                ground_truth_data
                if ground_truth_data.get("paper_id") == paper_id
                else None
            )

        if paper_ground_truth is None:
            return f"No entity ground truth configured for `{paper_id}`."

        with open(graph_path, "r", encoding="utf-8") as file:
            graph_data = json.load(file)

        edge_key = "edges" if "edges" in graph_data else "links"

        graph = nx.node_link_graph(
            graph_data,
            directed=True,
            multigraph=True,
            edges=edge_key,
        )

        expected_entities = paper_ground_truth.get("entities", [])

        extracted_entities = [
            str(data.get("name", node_id))
            for node_id, data in graph.nodes(data=True)
            if str(data.get("node_type", data.get("type", ""))).lower()
            != "paper"
            and not str(node_id).startswith("paper::")
        ]

        metrics = evaluate_entity_extraction(
            expected_entities,
            extracted_entities,
        )

        return "\n".join([
            f"### Entity Extraction Quality — `{paper_id}`",
            f"- Expected entities: {metrics['expected_entities']}",
            f"- Extracted entities: {metrics['extracted_entities']}",
            f"- True positives: {metrics['true_positives']}",
            f"- False positives: {metrics['false_positives']}",
            f"- False negatives: {metrics['false_negatives']}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- F1 score: {metrics['f1_score']:.4f}",
            "",
            "*Metrics use exact normalized entity-name matching.*",
        ])

    except (ValueError, TypeError, KeyError, OSError) as exc:
        return f"Entity evaluation failed: {exc}"


# ---------------------------------------------------------
# RAG retrieval evaluation
# ---------------------------------------------------------


def evaluate_retrieved_sources(paper_id, question, sources, k=3):
    """Evaluate retrieval using the selected paper's labelled queries."""

    ground_truth_path = PROCESSED_DIR / "retrieval_ground_truth.json"

    if not ground_truth_path.exists():
        return (
            "### Retrieval Evaluation\n"
            "Ground-truth file not found. Precision@K and Recall@K "
            "cannot be calculated."
        )

    try:
        ground_truth_data = json.loads(
            ground_truth_path.read_text(encoding="utf-8")
        )

        # Support the new multi-paper format and the legacy format.
        if "papers" in ground_truth_data:
            paper_ground_truth = next(
                (
                    item
                    for item in ground_truth_data["papers"]
                    if item.get("paper_id") == paper_id
                ),
                None,
            )
        else:
            paper_ground_truth = (
                ground_truth_data
                if ground_truth_data.get("paper_id") == paper_id
                else None
            )

        if paper_ground_truth is None:
            return (
                "### Retrieval Evaluation\n"
                f"No ground truth configured for `{paper_id}`."
            )

        matched_item = next(
            (
                item
                for item in paper_ground_truth.get("queries", [])
                if item.get("query", "").strip().casefold()
                == (question or "").strip().casefold()
            ),
            None,
        )

        if matched_item is None:
            return (
                "### Retrieval Evaluation\n"
                "No ground-truth labels found for this exact question. "
                "Add verified labels to evaluate this query."
            )

        relevant_ids = matched_item.get("relevant_chunk_ids", [])

        if not relevant_ids or any(
            "REPLACE_WITH" in str(label)
            for label in relevant_ids
        ):
            return (
                "### Retrieval Evaluation\n"
                "Ground-truth chunk labels are missing or unverified. "
                "Precision@K and Recall@K are unavailable."
            )

        retrieved_ids = [
            f"{source.get('paper_id')}_chunk_{source.get('chunk_index')}"
            for source in sources
        ]

        metrics = evaluate_retrieval_relevance(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        distances = [
            source["distance"]
            for source in sources
            if isinstance(source.get("distance"), (int, float))
        ]

        lines = [
            "### Retrieval Evaluation",
            f"- Precision@{k}: {metrics['precision_at_k']:.4f}",
            f"- Recall@{k}: {metrics['recall_at_k']:.4f}",
            f"- Relevant chunks retrieved: "
            f"{metrics['relevant_retrieved']}/{metrics['relevant_count']}",
        ]

        if distances:
            lines.append(
                f"- Mean cosine distance: "
                f"{sum(distances) / len(distances):.4f}"
            )
        else:
            lines.append("- Mean cosine distance: unavailable")

        lines.append(
            "\n*Scores depend on the quality of the labelled ground truth.*"
        )

        return "\n".join(lines)

    except (ValueError, TypeError, KeyError, OSError) as exc:
        return f"Retrieval evaluation failed: {exc}"



# ---------------------------------------------------------
# Research task execution
# ---------------------------------------------------------

def _run_task_sync(paper_id, task, question, progress=gr.Progress()):
    """Run a task and update the corresponding UI components."""
    print(f"[RUN] paper_id={paper_id}, task={task}")
    progress(0, desc="Preparing request...")

    empty_tables = ([], [], [], [], [], [])
    answer_evaluation_text = ""
    extraction_visibility = gr.update(visible=False)
    qa_visibility = gr.update(visible=(task == "Ask a question"))
    summary_visibility = gr.update(visible=(task == "Summarize paper"))
    summary_text = ""

    if not paper_id:
        print("[ERROR] No paper selected")
        return (
            "Please select a paper.",                     # answer
            "",                                           # inline evaluation
            "",                                           # sources
            "Retrieval evaluation unavailable.",          # dashboard evaluation
            "Select a paper to view graph quality metrics.",
            "Select a paper to view its knowledge graph.",
            *empty_tables,
            gr.update(visible=False),                     # extraction_results
            gr.update(visible=False),                     # qa_results_row
            "",                                           # summary_output
            gr.update(visible=False),                     # summary_results_section
        )

    if task == "Ask a question" and not (question or "").strip():
        graph_html, _ = get_graph_html(paper_id)

        return (
            "Please enter a question.",
            "",                                           # inline evaluation
            "",                                           # sources
            "Retrieval evaluation unavailable.",
            evaluate_graph_quality(paper_id),
            graph_html,
            *empty_tables,
            gr.update(visible=False),                     # extraction_results
            gr.update(visible=True),                      # qa_results_row
            "",                                           # summary_output
            gr.update(visible=False),                     # summary_results_section
        )

    request_text = {
        "Ask a question": question,
        "Summarize paper": "Summarize this paper",
        "Extract knowledge": "Extract entities and research knowledge",
    }[task]

    print(f"[RUN] Invoking research_graph: {request_text}")
    try:
        result = research_graph.invoke({
            "question": request_text,
            "paper_id": paper_id,
            "request_type": "",
            "vector_context": [],
            "sources": [],
            "graph_relationships": [],
            "answer": "",
            "summary_path": "",
            "extraction_path": "",
        })
        progress(0.8, desc="Preparing results...")
        print("[RUN] Research graph completed")
        print("[RUN] Result keys:", list(result.keys()))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise

    answer = result.get("answer", "No result returned.")

    # Normalize Gemini response content for Gradio Markdown.
    if isinstance(answer, list):
        answer = "\n".join(
            item["text"]
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    elif not isinstance(answer, str):
        answer = str(answer)

    summary_text = answer if task == "Summarize paper" else ""
    sources = []
    rag_evaluation_text = (
        "Retrieval evaluation is available for Q&A tasks."
    )

    if task == "Ask a question":
        sources = result.get("sources", [])

        source_text = "\n".join(
            f"- Paper: {source.get('paper_id', paper_id)} | "
            f"Chunk: {source.get('chunk_index', 'N/A')} | "
            f"Cosine distance: {source['distance']:.4f}"
            if isinstance(source.get("distance"), (int, float))
            else (
                f"- Paper: {source.get('paper_id', paper_id)} | "
                f"Chunk: {source.get('chunk_index', 'N/A')} | "
                "Cosine distance: N/A"
            )
            for source in sources
        ) or "No sources returned."

        rag_evaluation_text = evaluate_retrieved_sources(
            paper_id,
            question,
            sources,
        )
        answer_evaluation_text = rag_evaluation_text
    else:
        source_text = ""
        answer_evaluation_text = ""

    graph_evaluation_text = evaluate_graph_quality(paper_id)
    graph_html, graph_file_path = get_graph_html(paper_id)

    method_rows = []
    dataset_rows = []
    metric_rows = []
    concept_rows = []
    finding_rows = []
    relationship_rows = []

    if task == "Extract knowledge":
        extraction = load_extraction(paper_id, result)

        method_rows = to_entity_rows(
            extraction.get("methods", [])
        )
        dataset_rows = to_entity_rows(
            extraction.get("datasets", [])
        )
        metric_rows = to_entity_rows(
            extraction.get("metrics", [])
        )
        concept_rows = to_entity_rows(
            extraction.get("concepts", [])
        )
        finding_rows = to_entity_rows(
            extraction.get("findings", [])
        )
        relationship_rows = to_relationship_rows(
            extraction.get("relationships", [])
        )

        answer = (
            f"## Knowledge Extraction Complete\n\n"
            f"**Paper:** `{paper_id}`\n\n"
            f"- Methods: {len(method_rows)}\n"
            f"- Datasets: {len(dataset_rows)}\n"
            f"- Metrics: {len(metric_rows)}\n"
            f"- Concepts: {len(concept_rows)}\n"
            f"- Findings: {len(finding_rows)}\n"
            f"- Relationships: {len(relationship_rows)}\n\n"
            "Review the tables below for details."
        )
        extraction_visibility = gr.update(visible=True)

    progress(0.95, desc="Finalizing results...")

    
    
    return (
        answer,
        answer_evaluation_text,
        source_text,
        rag_evaluation_text,
        graph_evaluation_text,
        graph_html,
        method_rows,
        dataset_rows,
        metric_rows,
        concept_rows,
        finding_rows,
        relationship_rows,
        extraction_visibility,
        qa_visibility,
        summary_text,
        summary_visibility,
    )



def run_task(paper_id, task, question, progress=gr.Progress()):
    """Run the task in a background thread and update elapsed time."""

    started_at = time.monotonic()
    outcome = {}

    def worker():
        try:
            outcome["result"] = _run_task_sync(
                paper_id, task, question, progress
            )
        except Exception as exc:
            outcome["error"] = exc

    # Clear previous outputs immediately when Run is clicked.
    cleared_outputs = (
        "",                         # answer_output
        "",                         # answer_evaluation_output
        "",                         # sources_output
        "Retrieval evaluation pending...",
        "Graph evaluation pending...",
        "",                         # graph_output
        [], [], [], [], [], [],     # six extraction tables
        gr.update(visible=False),   # extraction_results
        gr.update(visible=(task == "Ask a question")),
        "",                         # summary_output
        gr.update(visible=False),   # summary_results_section
    )

    yield (
        *cleared_outputs,
        f"⏳ Processing: **{task}** — starting...",
    )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive():
        elapsed = int(time.monotonic() - started_at)

        yield (
            *[gr.update() for _ in range(16)],
            f"⏳ Processing: **{task}** — {elapsed} seconds elapsed",
        )

        thread.join(timeout=1)

    if "error" in outcome:
        raise outcome["error"]

    elapsed = int(time.monotonic() - started_at)

    yield (
        *outcome["result"],
        f"✅ Completed in {elapsed}s",
    )




def update_task_visibility(task):
    """Clear stale results and show the layout for the selected task."""

    is_ask = task == "Ask a question"
    is_summary = task == "Summarize paper"
    is_extract = task == "Extract knowledge"

    return (
        gr.update(visible=is_ask),          # question_input
        gr.update(visible=is_extract),      # extraction_section
        "",                                 # answer_output
        gr.update(value="", visible=is_ask),# sources_output
        "",                                 # rag_evaluation_output
        [],                                 # methods_output
        [],                                 # datasets_output
        [],                                 # metrics_output
        [],                                 # concepts_output
        [],                                 # findings_output
        [],                                 # relationships_output
        gr.update(visible=False),           # extraction_results
        gr.update(visible=is_ask),          # qa_results_row
        "",                                 # summary_output
        gr.update(visible=is_summary),      # summary_results_section
        "",             
        "",  # progress_status                    # answer_evaluation_output
    )






# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------

with gr.Blocks(title="Research Paper Assistant",
    css="""
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
    }
    """
) as demo:
    gr.Markdown("# Research Paper Assistant")
    gr.Markdown(
        "Explore research papers, ask questions, inspect knowledge graphs, "
        "and evaluate retrieval and graph quality."
    )

    with gr.Row(equal_height=False):
        # Shared left sidebar
        with gr.Column(scale=1.,min_width=280):
            
            # Discover Papers
            gr.Markdown("## Discover Papers")

            search_query = gr.Textbox(
                label="Search arXiv",
                placeholder="e.g. RAG, knowledge graphs, LLM agents",
            )

            max_results_input = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Maximum results",
            )

            search_button = gr.Button("Search Papers", variant="primary")
            search_status = gr.Markdown()

            # Keep the search results in Gradio state.
            search_results_state = gr.State([])

            search_results_dropdown = gr.Dropdown(
                label="Search Results",
                choices=[],
                value=None,
                interactive=True,
            )

            paper_preview = gr.Markdown(
                "Search arXiv and select a result to preview its details."
            )

            ingest_button = gr.Button("Ingest Selected Paper")
            ingest_status = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("## Ingested Papers")

            paper_dropdown = gr.Dropdown(
                label="Select research paper",
                choices=get_paper_choices(),
                value="",
                interactive=True,
                filterable=True,
                allow_custom_value=False,
            )

            gr.Markdown(
                "The selected paper is shared across all three tabs."
            )
        gr.HTML(
            "<div style='height:100%; min-height:600px; "
            "border-left:1px solid #d1d5db; margin:0 12px;'></div>"
        )

        # Main content
        with gr.Column(scale=4):
            with gr.Tabs():
                # -----------------------------------------
                # Tab 1: Research Assistant
                # -----------------------------------------
                with gr.Tab("Research Assistant"):
                    gr.Markdown(
                        "### Ask, summarize, or extract"
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
                        placeholder=(
                            "e.g. How does mixed training affect "
                            "no-preference retention?"
                        ),
                        lines=3,
                        visible=True,
                    )

                    
                    
                    run_button = gr.Button("Run", variant="primary",)
                    progress_status = gr.Markdown(value="", visible=True,)
                    
                    # Q&A: answer and retrieval evaluation side by side
                    with gr.Row(visible=True) as qa_results_row:
                        with gr.Column(scale=3):
                            answer_output = gr.Markdown(label="Answer")

                        with gr.Column(scale=2):
                            answer_evaluation_output = gr.Markdown(
                                value="",
                                label="Retrieval Evaluation",
                            )

                    # Summary: full-width output
                    with gr.Column(visible=False) as summary_results_section:
                        summary_output = gr.Markdown(label="Paper Summary")


                    sources_output = gr.Textbox(
                        label="Sources",
                        lines=4,
                        interactive=False,
                    )

                    extraction_section = gr.Column(visible=False)

                    with extraction_section:
                        extraction_results = gr.Column(visible=False)

                        with extraction_results:
                            gr.Markdown("## Extracted Knowledge")

                            with gr.Accordion("Methods", open=True):
                                methods_output = gr.Dataframe(
                                    headers=["Name", "Type", "Description"],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Extracted methods",
                                )

                            with gr.Accordion("Datasets", open=False):
                                datasets_output = gr.Dataframe(
                                    headers=[
                                        "Name", "Type", "Description"
                                    ],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Extracted datasets",
                                )

                            with gr.Accordion("Metrics", open=False):
                                metrics_output = gr.Dataframe(
                                    headers=[
                                        "Name", "Type", "Description"
                                    ],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Extracted metrics",
                                )

                            with gr.Accordion("Concepts", open=False):
                                concepts_output = gr.Dataframe(
                                    headers=[
                                        "Name", "Type", "Description"
                                    ],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Extracted concepts",
                                )

                            with gr.Accordion(
                                "Key Findings", open=False
                            ):
                                findings_output = gr.Dataframe(
                                    headers=[
                                        "Name", "Type", "Description"
                                    ],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Extracted findings",
                                )

                            with gr.Accordion(
                                "Relationships", open=False
                            ):
                                relationships_output = gr.Dataframe(
                                    headers=[
                                        "Source",
                                        "Relationship",
                                        "Target",
                                    ],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Extracted relationships",
                                )
                        
                # -----------------------------------------
                # Tab 2: Knowledge Graph
                # -----------------------------------------
                with gr.Tab("Knowledge Graph"):
                    gr.Markdown(
                        "### Knowledge Graph Visualization"
                    )
                    gr.Markdown(
                        "The visualization updates when you select "
                        "a different paper in the sidebar."
                    )

                    graph_output = gr.HTML(
                        value=(
                            "Select a paper in the sidebar to view "
                            "its knowledge graph."
                        )
                    )

                    graph_download = gr.File(
                        label="Download Knowledge Graph HTML",
                        interactive=False,
                    )

                # -----------------------------------------
                # Tab 3: Evaluation Dashboard
                # -----------------------------------------
                with gr.Tab("Evaluation Dashboard"):
                    gr.Markdown("## Evaluation Dashboard")

                    with gr.Accordion(
                        "RAG Retrieval Evaluation", open=True
                    ):
                        rag_evaluation_output = gr.Markdown(
                            "Run a question in the Research Assistant "
                            "tab to view retrieval evaluation."
                        )

                    with gr.Accordion(
                        "Knowledge Graph Quality", open=True
                    ):
                        graph_evaluation_output = gr.Markdown(
                            "Select a paper in the sidebar to view "
                            "graph metrics."
                        )

                    with gr.Accordion("Entity Extraction Quality", open=True):
                        entity_evaluation_output = gr.Markdown(
                            "Select a paper to view entity extraction metrics."
                        )

                    gr.Markdown("### Planned Evaluation Dimensions")
                    gr.Markdown(
                        "- Summary quality: token overlap and semantic similarity\n"
                        "- Q&A quality: faithfulness, answer relevance, and correctness\n\n"
                        "These metrics are planned but not yet connected to the dashboard."
                    )


    # Update graph and graph metrics immediately when paper changes.
    paper_dropdown.change(
        fn=update_selected_paper,
        inputs=paper_dropdown,
        outputs=[
                graph_output,
                graph_download,
                graph_evaluation_output,
                entity_evaluation_output,
                answer_output,
                answer_evaluation_output,
                sources_output,
                rag_evaluation_output,
                methods_output,
                datasets_output,
                metrics_output,
                concepts_output,
                findings_output,
                relationships_output,
                extraction_results,
                extraction_section,
                qa_results_row,
                summary_output,
                summary_results_section,
                progress_status,
            ],
        )

    # Run task and update all related outputs.
    
    run_button.click(
        fn=run_task,
        inputs=[
            paper_dropdown,
            task_dropdown,
            question_input,
        ],
        outputs=[
            answer_output,
            answer_evaluation_output,
            sources_output,
            rag_evaluation_output,
            graph_evaluation_output,
            graph_output,
            methods_output,
            datasets_output,
            metrics_output,
            concepts_output,
            findings_output,
            relationships_output,
            extraction_results,
            qa_results_row,
            summary_output,
            summary_results_section,
            progress_status,
        ],
    )


    
    search_button.click(
        fn=search_papers_ui,
        inputs=[search_query, max_results_input],
        outputs=[
            search_results_state,
            search_results_dropdown,
            search_status,
        ],
    )

    search_results_dropdown.change(
        fn=preview_paper,
        inputs=[search_results_dropdown, search_results_state],
        outputs=paper_preview,
    )

    ingest_button.click(
        fn=ingest_paper_ui,
        inputs=[search_results_dropdown, search_results_state],
        outputs=[ingest_status, paper_dropdown],
    )

        
    task_dropdown.change(
        fn=update_task_visibility,
        inputs=task_dropdown,
        outputs=[
            question_input,
            extraction_section,
            answer_output,
            sources_output,
            rag_evaluation_output,
            methods_output,
            datasets_output,
            metrics_output,
            concepts_output,
            findings_output,
            relationships_output,
            extraction_results,
            qa_results_row,
            summary_output,
            summary_results_section,
            answer_evaluation_output,
            progress_status
        ],
    )




if __name__ == "__main__":
    demo.launch()
