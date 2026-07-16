"""
app.py — Streamlit UI for the Smart Research Assistant (RAG system)

Run with:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_document,
    chunk_documents,
    get_embeddings,
    
    build_vectorstore,
    build_qa_chain,
    answer_question,
)

load_dotenv()

st.set_page_config(page_title="Smart Research Assistant", page_icon="📚", layout="wide")

# Global UI CSS: background and center-panel styling
st.markdown(
    """
    <style>
    /* Apply to multiple possible root selectors so it works across Streamlit versions */
    body, .stApp, .reportview-container, .main {
        background-color: #0b0f12 !important;
        background-image:
            repeating-linear-gradient(180deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 36px),
            radial-gradient(circle at 50% 10%, rgba(255,255,255,0.03), rgba(255,255,255,0) 40%);
        background-size: 100% 36px, 420px 300px;
        background-position: 0 120px, center 80px;
        background-repeat: repeat, no-repeat;
    }

    .block-container { padding-top: 2rem; }

    .center-panel {
        border: none !important;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        background: rgba(255,255,255,0.02) !important;
        box-shadow: 0 8px 30px rgba(2,6,23,0.6) inset;
        backdrop-filter: blur(6px);
    }

    .center-panel > h1, .center-panel > h2 { margin-top: 0.25rem; }

    /* Hide stray top-level input-like elements that can render as a white pill */
    input[type="text"], .stTextInput, [data-testid="stTextInput"] { background: transparent !important; border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Smart Research Assistant")
st.caption("Upload a document and ask questions — answers are grounded in your content, with sources shown.")

# ---- Session state ----
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# ---- Main UI: centered upload + query ----

center_col_1, center_col_2, center_col_3 = st.columns([1, 2, 1])
with center_col_2:
    st.markdown('<div class="center-panel">', unsafe_allow_html=True)

    st.header("1. Upload & process document")
    st.caption("Upload a PDF or TXT file, chunk it into searchable passages, and then ask questions below.")
    uploaded_file = st.file_uploader("PDF or TXT file", type=["pdf", "txt"])

    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=3)

    process_clicked = st.button("Process document", type="primary", use_container_width=True)

    if uploaded_file is not None and process_clicked:
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing document (chunking + embedding)... this can take a minute on first run."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    docs = load_document(tmp_path)
                    chunks = chunk_documents(docs)
                    embeddings = get_embeddings()
                    vectorstore = build_vectorstore(chunks, embeddings)
                    st.session_state.chain = build_qa_chain(vectorstore, k=k)
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'")
                except EnvironmentError as e:
                    st.error(str(e))
                finally:
                    os.unlink(tmp_path)
        else:
            st.info("This document has already been processed and is ready to query.")
    elif uploaded_file is not None and st.session_state.processed_file == uploaded_file.name:
        st.success(f"'{uploaded_file.name}' is already indexed and ready to query.")

    st.divider()
    st.header("2. Ask question")

    if st.session_state.chain is None:
        st.info("Upload and process a document to start asking questions.")
    else:
        # First show the query input so the latest result appears above the historical list
        question = st.chat_input("Ask a question about the uploaded document...")

        if question:
            # Display the current interaction immediately
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    answer, sources = answer_question(st.session_state.chain, question)
                st.write(answer)
                with st.expander("View source chunks"):
                    for i, src in enumerate(sources, 1):
                        page = src.metadata.get("page", "N/A")
                        st.markdown(f"**Source {i}** (page {page})")
                        st.text(src.page_content[:500])

            # Persist the new interaction
            st.session_state.chat_history.append(
                {"question": question, "answer": answer, "sources": sources}
            )

            # After showing the latest result, show previous history below (exclude the just-added latest)
            if len(st.session_state.chat_history) > 1:
                st.markdown("---")
                st.subheader("Previous interactions")
                for entry in st.session_state.chat_history[:-1]:
                    with st.chat_message("user"):
                        st.write(entry["question"])
                    with st.chat_message("assistant"):
                        st.write(entry["answer"])
                        with st.expander("View source chunks"):
                            for i, src in enumerate(entry["sources"], 1):
                                page = src.metadata.get("page", "N/A")
                                st.markdown(f"**Source {i}** (page {page})")
                                st.text(src.page_content[:500])
        else:
            # No new question: show full history (latest first)
            if st.session_state.chat_history:
                for entry in reversed(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.write(entry["question"])
                    with st.chat_message("assistant"):
                        st.write(entry["answer"])
                        with st.expander("View source chunks"):
                            for i, src in enumerate(entry["sources"], 1):
                                page = src.metadata.get("page", "N/A")
                                st.markdown(f"**Source {i}** (page {page})")
                                st.text(src.page_content[:500])

    st.divider()
    st.caption(
        "Stack: LangChain · Hugging Face embeddings (local, free) · "
        "FAISS · Groq (Llama 3.1) for generation"
    )
    if not os.getenv("GROQ_API_KEY"):
        st.warning("No GROQ_API_KEY found. Copy .env.example to .env and add your free key from console.groq.com")

    st.markdown('</div>', unsafe_allow_html=True)
