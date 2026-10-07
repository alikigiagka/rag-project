# RAG System - Databases and Parallel & Distributed Computing Courses

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system tailored for the lecture slides, notes, and technical diagrams of two undergraduate courses at the Department of Applied Informatics, University of Macedonia: **"Databases"** (Greek: *Βάσεις Δεδομένων*) and **"Parallel & Distributed Computing"** (Greek: *Παράλληλος και Κατανεμημένος Υπολογισμός*).

This project constitutes one of the two codebase implementations developed for the Undergraduate Thesis titled **"Design and Evaluation of a Retrieval-Augmented Generation Question-Answering System for Educational Content"** (*Σχεδιασμός και Αξιολόγηση Συστήματος Ερωτοαπαντήσεων Retrieval-Augmented Generation για Εκπαιδευτικό Υλικό*) by Aliki Giagka, under the supervision of Assoc. Prof. Georgia Koloniari at the Department of Applied Informatics, University of Macedonia (July 2026).

---

## Architecture & System Features

The system implements an advanced hybrid RAG architecture specifically tuned for processing multi-course lecture slides, which often consist of fragmented bullet points, code snippets, and structural diagrams:

- **Isolated Course Indexing**: Each course is indexed separately, with its own ChromaDB collection (`db_course`, `parallel_course`) and its own sparse BM25 index (`bm25_databases.pkl`, `bm25_parallel.pkl`), so that retrieval never mixes content across courses.
- **Slide Buffer Chunking**: Text is extracted from PDF slides using PyMuPDF (`fitz`). Consecutive slides/pages are merged in a buffer until a minimum size is reached (`MIN_CHUNK_WORDS = 15` words), eliminating standalone slide titles, isolated page numbers, and structural noise. Slide/page and file name metadata is attached to every chunk.
- **Multimodal Visual Content Ingestion**: Diagrams (ER diagrams, relational schemas, architectural models, flowcharts) and code snippets are extracted from PDF slides and described in Greek using Gemini 2.5 Flash as a Vision LLM. Descriptions are stored as independent visual chunks in the vector database, allowing autonomous retrieval. Repeated API calls are avoided through MD5 hash caching (`vision_cache.json`).
- **Dense Vector Retrieval**: Uses `intfloat/multilingual-e5-large-instruct` sentence embeddings stored in ChromaDB, retrieving top 8 dense candidates (`TOP_K_DENSE = 8`).
- **Sparse Keyword Retrieval**: Uses BM25Okapi with custom Greek tokenization (`tokenize_greek`: Unicode NFD normalization, lowercasing, accent and punctuation removal), retrieving top 8 sparse candidates (`TOP_K_SPARSE = 8`).
- **Hybrid Fusion via RRF**: Combines dense and sparse search rankings using Reciprocal Rank Fusion (`RRF_K = 60`), applies a score threshold (> 0.01), and selects the top 5 chunks (`FINAL_TOP_K = 5`) for prompt construction.
- **Grounded Response Generation**: Uses Google's `gemini-2.5-flash` (`temperature = 0`) with system instructions enforcing answers strictly grounded in the retrieved text, explicit file name and slide/page citation, answers in Greek, and an explicit refusal (*"Δεν βρίσκω αυτή την πληροφορία στο διαθέσιμο υλικό."*) when information is missing.
- **Dual User Interfaces**: Interactive command-line interface (CLI) with course selection flags, and a FastAPI web server serving a single-page web UI with dynamic course switching, real-time message streaming, and source citation cards.
- **Evaluation Pipeline**: A custom dataset of 150 questions (`rag_evaluation_data.json`) spanning 5 difficulty categories (factual recall, conceptual understanding, multi-hop reasoning, image-based questions, unanswerable / false premise) and 3 content sources, evaluated via the LLM-as-a-Judge methodology using Gemma 3 4B across 4 key RAGAS metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall.

---

## Repository Structure

```
├── api.py                    # FastAPI web server and HTTP API endpoints
├── main.py                   # CLI entry point for interactive chat and ingestion
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment configuration
├── rag_evaluation_data.json  # 150-question RAG evaluation dataset
├── src/
│   ├── config.py             # Global configurations, course metadata, and hyperparameters
│   ├── ingest.py             # Document processing, chunking, vision API, and indexing
│   ├── llm.py                # Gemini LLM generation and system prompt logic
│   └── retriever.py          # Hybrid RRF retriever implementation (ChromaDB + BM25)
├── static/                   # Frontend Web UI static assets (HTML/CSS/JS)
├── evaluation/               # RAG evaluation scripts and Jupyter notebooks
├── google colab/             # Colab notebooks for GPU-accelerated ingestion & embeddings
└── data/                     # Source PDF material directory (data/databases/, data/parallel-distributed/)
```

---

## Prerequisites & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/USERNAME/rag-project-courses.git
   cd rag-project-courses
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy `.env.example` to `.env` and set your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

---

## Usage

### 1. Document Ingestion
Document ingestion (including GPU embedding calculation and Vision LLM diagram description) is optimized for Google Colab (`google colab/ingest_colab.ipynb`).

Alternatively, to run ingestion locally, place the course PDFs into `data/databases/` or `data/parallel-distributed/` and run:
```bash
# Databases
python main.py --course databases --ingest

# Parallel & Distributed Computing
python main.py --course parallel-distributed --ingest
```

### 2. Interactive CLI Chat
Run the terminal chat interface for a course:
```bash
# Databases
python main.py --course databases

# Parallel & Distributed Computing
python main.py --course parallel-distributed
```

### 3. FastAPI Web Application
Start the FastAPI server:
```bash
uvicorn api:app --reload
```
Open a browser and visit `http://127.0.0.1:8000`.

---

## Citation & Academic Context

If you use this codebase, please cite the underlying thesis:

```bibtex
@thesis{giagka2026rag,
  author       = {Aliki Giagka},
  title        = {Design and Evaluation of a Retrieval-Augmented Generation Question-Answering System for Educational Content},
  school       = {University of Macedonia, Department of Applied Informatics},
  year         = {2026},
  type         = {Undergraduate Thesis},
  supervisor   = {Georgia Koloniari},
  address      = {Thessaloniki, Greece}
}
```

---

## License

Distributed under the MIT License.
