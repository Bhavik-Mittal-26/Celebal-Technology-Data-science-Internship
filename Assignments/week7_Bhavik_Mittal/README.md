# Document Question Answering System (RAG) — 100% Local, No API Key

A Retrieval-Augmented Generation pipeline that answers questions from your own
PDFs, TXT, or DOCX files. Everything — embeddings, vector search, and answer
generation — runs **locally on your machine**. There is no API key, no paid
endpoint, and no external service call anywhere in this project.

## Architecture

```
Documents (PDF/TXT/DOCX)
        │
        ▼
 Ingestion (src/ingestion.py)        -> raw text
        │
        ▼
 Chunking (src/chunking.py)          -> overlapping text chunks
        │
        ▼
 Embeddings (src/embeddings.py)      -> sentence-transformers (all-MiniLM-L6-v2)
        │
        ▼
 Vector Store (src/vectorstore.py)   -> FAISS (local, in-memory index)
        │
        ▼
 Question ──> Query Embedding ──> Retrieval (top-k similar chunks)
        │
        ▼
 LLM (src/llm.py)                    -> google/flan-t5-large (local, via transformers)
        │
        ▼
 Grounded Answer
```

## Why no API key is needed

| Component | Tool used | Notes |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Public model, downloaded once, cached, runs on CPU |
| Vector DB | `FAISS` (`faiss-cpu`) | Runs fully in-memory/on-disk, no server |
| Answer generation | `transformers` (`google/flan-t5-large`) | Public model, downloaded once, runs locally |

The first run will download the two models from Hugging Face (a few hundred
MB total). After that, everything works fully offline.

## Setup

```bash
pip install -r requirements.txt
```

(Optional but recommended: use a virtual environment first.)

## Usage

### Option A — Web UI (recommended)

```bash
streamlit run app.py
```

Then in the browser:
1. Upload one or more PDF/TXT/DOCX files in the sidebar.
2. Click **Process documents**.
3. Ask questions in the main panel — answers are grounded in your documents,
   with retrieved source chunks shown for transparency.

### Option B — Command line

```bash
python run_cli.py --docs notes.pdf --question "What is the main idea of the document?"
```

Or run interactively:

```bash
python run_cli.py --docs notes.pdf resume.pdf
```

### Option C — Evaluation / validation report

Runs a batch of sample questions and writes a metrics report + validation log:

```bash
python evaluate.py --docs notes.pdf --questions "What is this about?" "What are the key skills listed?"
```

Outputs:
- `logs/validation_log.jsonl` — every question, answer, retrieved sources, and similarity scores
- `logs/metrics_report.md` — chunking profile, embedding model, vector store, and aggregate scores

## Project structure

```
rag_project/
├── app.py                 # Streamlit UI (no API key fields anywhere)
├── run_cli.py              # Terminal entry point
├── evaluate.py              # Validation + metrics report generator
├── requirements.txt
├── src/
│   ├── ingestion.py         # PDF/TXT/DOCX loading
│   ├── chunking.py          # Sentence-aware overlapping chunking
│   ├── embeddings.py        # Local embedding model wrapper
│   ├── vectorstore.py       # FAISS vector store wrapper
│   ├── llm.py               # Local LLM wrapper (flan-t5-large)
│   └── rag_pipeline.py      # Orchestrates the full RAG flow
└── logs/                    # Validation logs + metrics reports land here
```

## Tuning for better answers

- **Chunk size / overlap** (sidebar sliders or `--chunk_size`, `--chunk_overlap`):
  smaller chunks improve retrieval precision, larger chunks preserve more context.
- **top_k**: more retrieved chunks give the LLM more context, but too many can
  dilute the prompt. 3–5 is a good starting range.
- **Swap the LLM**: in `src/llm.py`, change `model_name` to `google/flan-t5-base`
  (faster, lower quality) or `google/flan-t5-xl` (slower, higher quality, needs more RAM).
- **Hybrid search / re-ranking**: the codebase is structured so you can add a
  keyword search (e.g. BM25) alongside FAISS in `vectorstore.py` and merge/re-rank
  results before passing them to the LLM — a good "improvements" experiment to
  write up in your assignment report.

## Key learnings this project demonstrates

- How retrieval and generation combine to ground LLM answers in real documents
- Practical use of embeddings and vector similarity search
- Handling unstructured text (PDF/DOCX/TXT) end-to-end
- Designing a modular, swappable RAG pipeline (each stage is a separate file)
