# 📚 Smart Research Assistant (RAG-based Knowledge System)

A Retrieval-Augmented Generation assistant: upload a document, ask questions in
plain English, get answers grounded in the document with sources and quality scores.

## Stack

| Purpose            | Tool                                              |
|---------------------|----------------------------------------------------|
| Orchestration        | LangChain                                          |
| Embeddings           | Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) — free, local |
| Vector store          | FAISS                                              |
| Generation (LLM)   | Groq — Llama 3.1 8B (free tier, hosted)             |
| Evaluation           | RAGAs (faithfulness, answer relevancy, context precision) |
| UI                    | Streamlit                                          |

No OpenAI key is required anywhere in this project.

## Folder structure

```
rag-assistant/
├── notebook.ipynb        # step-by-step pipeline walkthrough
├── app.py                 # Streamlit chat UI
├── rag_pipeline.py        # core pipeline logic (shared by notebook + app)
├── evaluation.py          # RAGAs evaluation logic
├── requirements.txt
├── .env.example           # copy to .env and add your Groq key
├── README.md
└── data/
    ├── sample_hr_policy.pdf   # sample document to test with
    └── make_sample_pdf.py     # regenerates the sample PDF
```

## Setup

1. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Groq API key**
   - Sign up at https://console.groq.com (free, no credit card)
   - Create a key under **API Keys**

4. **Set up your `.env` file**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your key:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```
   `.env` is only read locally and should never be committed to git.

## Running the notebook

```bash
jupyter notebook notebook.ipynb
```
Run cells top to bottom. The first run downloads the embedding model
(~90MB) from Hugging Face, which takes a minute; it's cached after that.

## Running the Streamlit app

```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`),
upload a PDF/TXT in the sidebar, and start asking questions.

## Using your own documents

Replace `data/sample_hr_policy.pdf` with your own PDF, or upload any PDF/TXT
directly in the Streamlit sidebar. No code changes needed.

## How it works (pipeline)

1. **Ingestion** — PDF/TXT loaded and split into ~500-character overlapping chunks
2. **Embedding** — each chunk converted to a vector locally via Hugging Face
3. **Storage** — vectors indexed in FAISS for fast similarity search
4. **Retrieval** — top-k most similar chunks fetched for a given question
5. **Generation** — retrieved chunks + question sent to Groq's Llama 3.1, which
   answers using only that context (reduces hallucination)
6. **Evaluation** — RAGAs scores the answer's faithfulness and relevance

## Troubleshooting

- **`GROQ_API_KEY not found`** — make sure `.env` exists (not just `.env.example`)
  and is in the same folder you're running from.
- **Slow first run** — the Hugging Face embedding model downloads once (~90MB);
  subsequent runs use the local cache.
- **FAISS import errors on Apple Silicon** — use `faiss-cpu` (already in
  `requirements.txt`); avoid mixing with `faiss-gpu`.
