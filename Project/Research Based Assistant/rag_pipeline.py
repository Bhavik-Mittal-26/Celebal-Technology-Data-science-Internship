"""
rag_pipeline.py
----------------
Core RAG (Retrieval-Augmented Generation) pipeline.

Pipeline:
  PDF/TXT -> chunk -> HuggingFace embeddings -> FAISS vector store
  -> retrieve top-k chunks -> Groq LLM (Llama 3.1) -> grounded answer

Used by both notebook.ipynb (for step-by-step exploration) and app.py
(the Streamlit UI), so the logic only lives in one place.
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the
context provided below. If the answer is not contained in the context, say
"I don't have enough information in the provided documents to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer (be concise and cite specifics from the context where possible):"""


def load_document(file_path: str):
    """Load a PDF or TXT file into LangChain Document objects."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return loader.load()


def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings():
    """Free, local Hugging Face sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks, embeddings):
    """Build an in-memory FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore, path: str = "faiss_index"):
    vectorstore.save_local(path)


def load_vectorstore(path: str, embeddings):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_llm(temperature: float = 0.0):
    """Groq-hosted Llama model. Needs GROQ_API_KEY set in the environment."""
    api_key = os.getenv("GROQ_API_KEY")

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)


def build_qa_chain(vectorstore, k: int = 3):
    """Wire retriever + LLM + prompt into a RetrievalQA chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = get_llm()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def answer_question(chain, question: str):
    """Run a query through the chain. Returns (answer, source_documents)."""
    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]


def build_pipeline_from_file(file_path: str, k: int = 3):
    """Convenience one-shot: file -> ready-to-query QA chain."""
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    chain = build_qa_chain(vectorstore, k=k)
    return chain, vectorstore
