# research-paper-knowledge-graph-
Autonomous Research Paper Summarization &amp; Knowledge Graph  Builder 

research-paper-knowledge-graph/
│
├── app.py                          # Gradio application entry point
├── main.py                         # Pipeline entry point
├── pyproject.toml                  # Dependencies and project configuration
├── .env.example                    # Environment variable template
├── .gitignore
├── README.md
│
├── src/
│   └── research_assistant/
│       ├── __init__.py
│       │
│       ├── config.py               # Configuration and environment variables
│       ├── models.py               # Pydantic data models
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── arxiv_client.py     # Fetch papers from arXiv
│       │   └── pdf_processor.py    # Extract text using PyMuPDF
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── chunker.py          # Split documents into chunks
│       │   └── embeddings.py       # Generate embeddings
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── vector_store.py     # ChromaDB operations
│       │   ├── document_store.py   # Paper text and metadata
│       │   └── graph_store.py      # NetworkX graph operations
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── orchestrator.py     # LangGraph workflow
│       │   ├── knowledge_extractor.py
│       │   ├── graph_builder.py
│       │   ├── summarizer.py
│       │   └── qa_agent.py         # Graph + RAG question answering
│       │
│       ├── workflows/
│       │   ├── __init__.py
│       │   └── research_graph.py   # LangGraph state and routing
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── extraction_eval.py
│       │   ├── retrieval_eval.py
│       │   ├── summary_eval.py
│       │   └── qa_eval.py
│       │
│       └── ui/
│           ├── __init__.py
│           └── graph_visualizer.py # PyVis visualization
│
├── data/
│   ├── papers/                     # Downloaded PDFs
│   ├── processed/                  # Extracted text and metadata
│   └── evaluation/                 # Evaluation datasets
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_chunker.py
│   ├── test_vector_store.py
│   ├── test_graph_builder.py
│   └── test_qa_agent.py
│
└── .github/
    └── workflows/
        └── tests.yml               # Optional GitHub Actions CI


Current Status

Successfully implemented and tested the initial ingestion, preprocessing, semantic retrieval, and knowledge extraction pipeline.

Next planned steps:

Entity deduplication and resolution.

Knowledge graph construction using NetworkX.

Research paper summarization.

Graph-based RAG question answering.

LangGraph orchestration.

Gradio interface and PyVis visualization.

Evaluation of extraction, retrieval, summaries, and Q&A.

## Implementation Progress

### Phase 1: Project Setup
- Set up the Python 3.11 development environment using `uv` in VS Code.
- Configured `pyproject.toml` for the `src`-based Python package structure.
- Created modular packages for ingestion, processing, storage, and agents.
- Organized test scripts under the root-level `scripts/` directory.

### Phase 2: Research Paper Ingestion
- Integrated the `arxiv` Python library to search for research papers.
- Implemented paper retrieval and PDF downloading.
- Retrieved paper metadata, including:
  - Paper ID and title
  - Abstract and authors
  - Publication and update dates
  - Categories and PDF URL
- Saved downloaded PDFs under `data/papers/`.
- Saved paper metadata to `data/processed/papers.json`.

### Phase 3: PDF Text Extraction
- Integrated PyMuPDF for PDF text extraction.
- Implemented page-by-page text extraction with page numbers.
- Captured the complete extracted text and paper metadata.
- Saved extracted paper content as JSON files under `data/processed/`.
- Successfully extracted text from a 24-page research paper.

### Phase 4: Document Chunking
- Integrated LangChain's `RecursiveCharacterTextSplitter`.
- Implemented overlapping text chunking using:
  - Chunk size: 1,000 characters
  - Chunk overlap: 200 characters
- Assigned unique chunk IDs and retained paper IDs and chunk indices.
- Saved chunked documents as JSON files under `data/chunks/`.
- Validated chunk generation and inspected sample chunks.

### Phase 5: Embeddings and Vector Storage
- Integrated Sentence Transformers using `all-MiniLM-L6-v2`.
- Generated 384-dimensional embeddings for paper chunks.
- Integrated ChromaDB with persistent local storage.
- Implemented chunk insertion and semantic similarity search.
- Stored paper IDs and chunk indices as metadata.
- Successfully tested semantic search using a natural-language query and retrieved relevant research-paper passages.

### Phase 6: LLM-Based Knowledge Extraction
- Integrated Google Gemini through `langchain-google-genai`.
- Configured Gemini API credentials using environment variables in `.env`.
- Created Pydantic models for structured research knowledge:
  - Methods
  - Datasets
  - Evaluation metrics
  - Concepts
  - Findings
  - Relationships
- Implemented knowledge extraction from research-paper text.
- Configured structured LLM output using Pydantic.
- Successfully extracted entities and relationships and saved the results as JSON under `data/processed/`.

### Current Status
- Successfully implemented and tested the initial ingestion, preprocessing, semantic retrieval, and knowledge extraction pipeline.
- Next planned steps:
  1. Entity deduplication and resolution.
  2. Knowledge graph construction using NetworkX.
  3. Research paper summarization.
  4. Graph-based RAG question answering.
  5. LangGraph orchestration.
  6. Gradio interface and PyVis visualization.
  7. Evaluation of extraction, retrieval, summaries, and Q&A.


  ## Development Progress – 18 October 2026

### 1. LangGraph Workflow Orchestration

Implemented a LangGraph-based workflow to coordinate research paper tasks.

**Request routing:**
- Added a request router to classify user requests into Q&A, summarization, and knowledge extraction.
- Implemented conditional edges to direct requests to the appropriate workflow branch.

**Q&A workflow:**
- Vector Retrieval Node – retrieves relevant paper chunks from ChromaDB.
- Graph Retrieval Node – retrieves related entities and relationships from the NetworkX knowledge graph.
- Answer Generation Node – uses Gemini to generate answers grounded in retrieved context, with paper ID and chunk citations.

**Summarization workflow:**
- Integrated the existing Gemini-based summarizer.
- Generates six sections: Objective, Methodology, Datasets, Key Findings, Contributions, and Limitations.
- Saves summaries as JSON under `data/processed/`.

**Knowledge Extraction workflow:**
- Integrated the existing structured knowledge extractor using Gemini and Pydantic.
- Extracts methods, datasets, metrics, concepts, findings, and relationships.
- Saves extracted knowledge as JSON under `data/processed/`.

### 2. End-to-End Paper Ingestion Pipeline

Created `src/research_assistant/workflows/ingestion_graph.py` to orchestrate paper ingestion.

**Pipeline:**

1. Search arXiv and download research papers.
2. Extract PDF text using PyMuPDF.
3. Split paper text into overlapping chunks.
4. Generate embeddings and store chunks in ChromaDB.
5. Extract research entities and relationships using Gemini.
6. Resolve duplicate entities using RapidFuzz.
7. Build a directed knowledge graph using NetworkX.
8. Persist the graph as JSON under `data/graph/`.

**Validation:** Successfully processed paper `2610.22372v1` (LIBERO-PeRM), extracted 22 pages, generated 116 chunks, stored them in ChromaDB, and built a knowledge graph containing 41 nodes and 64 edges.

### 3. Paper-Specific Retrieval

Enhanced the vector store to support filtering by paper ID.

- Updated `VectorStore.search()` to accept an optional `paper_id` parameter.
- Added ChromaDB metadata filtering to retrieve chunks only from the selected paper.
- Verified retrieval for `2610.22372v1`; all returned chunks belonged to the requested paper.
- Enabled dynamic knowledge-graph loading using the selected paper's graph file.

### 4. Gradio User Interface

Created `src/research_assistant/ui/app.py` using Gradio.

**Implemented features:**
- Select an ingested research paper.
- Choose a task: Ask a question, Summarize paper, or Extract knowledge.
- Display generated answers, summaries, and extracted knowledge.
- Display source citations for Q&A results.
- Generate and provide an interactive PyVis knowledge graph as an HTML file.

**Run the application:**

```bash
uv run python -m research_assistant.ui.app
```

### 5. Validation and Test Scripts

Added and executed test scripts to validate individual components and workflows:

- `scripts/test_research_graph.py` – LangGraph routing and task execution.
- `scripts/test_ingestion_graph.py` – End-to-end ingestion and graph construction.
- `scripts/test_dynamic_graph.py` – Loading and retrieving relationships from a selected paper's graph.
- `scripts/test_filtered_search.py` – Paper-specific ChromaDB retrieval.

### 6. Current Project Status

| Component | Status |
|---|---|
| arXiv ingestion and PDF processing | Implemented |
| Text chunking and embeddings | Implemented |
| ChromaDB vector storage and filtered retrieval | Implemented |
| Gemini knowledge extraction | Implemented |
| Entity resolution | Implemented |
| NetworkX knowledge graph construction | Implemented |
| PyVis graph visualization | Implemented |
| Gemini paper summarization | Implemented |
| LangGraph request routing | Implemented |
| Graph-based RAG Q&A | Implemented |
| End-to-end ingestion workflow | Implemented |
| Gradio UI | Implemented and tested |

### 7. Planned Improvements

- Display paper titles alongside paper IDs in the Gradio dropdown.
- Embed the interactive knowledge graph directly in the UI.
- Improve graph readability through layout tuning and relationship filtering.
- Allow users to search for and ingest papers directly from the UI.
- Improve entity resolution and reduce unresolved or ambiguous graph relationships.
- Add robust error handling and validation for missing papers, graphs, and retrieval results.
- Expand automated tests for routing, ingestion, retrieval, and output validation.

