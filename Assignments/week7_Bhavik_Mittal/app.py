"""
Document Question Answering System (RAG) — Streamlit UI
----------------------------------------------------------
Fully local. No API keys, no external services, no billing.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st

from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Document Q&A (RAG)",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------
# Sidebar — document upload + system settings (NO API key fields)
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Retrieval-Augmented Generation — 100% local, no API key required")

    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    st.subheader("2. Pipeline settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 10)
    top_k = st.slider("Chunks retrieved per question", 1, 8, 4, 1)

    process_btn = st.button("🚀 Process documents", use_container_width=True, type="primary")

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Loading models and building the vector index... (first run downloads models, may take a minute)"):
                tmp_dir = tempfile.mkdtemp()
                paths = []
                for f in uploaded_files:
                    path = os.path.join(tmp_dir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(path)

                pipeline = RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                metrics = pipeline.ingest(paths)

                st.session_state.pipeline = pipeline
                st.session_state.metrics = metrics
                st.session_state.history = []
            st.success("Documents processed and indexed!")

    if st.session_state.metrics:
        st.subheader("📊 System metrics")
        m = st.session_state.metrics
        st.metric("Documents ingested", m["num_documents"])
        st.metric("Chunks created", m["num_chunks"])
        st.caption(f"Embedding model: `{m['embedding_model']}` ({m['embedding_dim']}-dim)")
        st.caption(f"LLM: `{m['llm_model']}`")
        st.caption(f"Vector store: {m['vector_store']}")
        st.caption(f"Chunk size / overlap: {m['chunk_size']} / {m['chunk_overlap']}")
        st.caption(f"Ingestion time: {m['ingestion_time_sec']}s")

# ---------------------------------------------------------------------
# Main panel — Q&A interface
# ---------------------------------------------------------------------
st.header("Ask a question about your documents")

if st.session_state.pipeline is None:
    st.info("👈 Upload one or more documents in the sidebar and click **Process documents** to get started.")
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the main idea of the document?",
    )
    ask_btn = st.button("🔍 Get answer", type="primary")

    if ask_btn and question.strip():
        with st.spinner("Retrieving relevant context and generating an answer..."):
            result = st.session_state.pipeline.query(question, top_k=top_k)
            st.session_state.pipeline.log_interaction(result)
            st.session_state.history.insert(0, result)

    for result in st.session_state.history:
        st.markdown("---")
        st.markdown(f"**Q: {result['question']}**")
        st.success(result["answer"])
        st.caption(f"Answered in {result['latency_sec']}s using {len(result['retrieved_chunks'])} retrieved chunk(s)")

        with st.expander("🔎 View retrieved context (grounding evidence)"):
            for c in result["retrieved_chunks"]:
                st.markdown(
                    f"**Source:** `{c['source']}` &nbsp;·&nbsp; "
                    f"**Chunk #{c['chunk_id']}** &nbsp;·&nbsp; "
                    f"**Similarity:** {c['score']:.3f}"
                )
                st.text(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))
                st.markdown("")
