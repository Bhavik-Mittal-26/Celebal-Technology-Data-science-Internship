"""
RAG Pipeline
-------------
Orchestrates the full Retrieval-Augmented Generation flow:

  Documents -> Chunking -> Embeddings -> Vector Store
  Question  -> Query Embedding -> Retrieval -> Prompt -> LLM -> Answer

Everything runs locally. No API keys are used anywhere in this pipeline.
"""

import time
import json
import os

from src.ingestion import load_multiple_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore
from src.llm import LocalLLM


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-large",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedder = EmbeddingModel(embedding_model_name)
        self.vectorstore = VectorStore(self.embedder.embedding_dim)
        self.llm = LocalLLM(llm_model_name)

        self.metrics = {
            "num_documents": 0,
            "num_chunks": 0,
            "embedding_model": embedding_model_name,
            "embedding_dim": self.embedder.embedding_dim,
            "llm_model": llm_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "vector_store": "FAISS (IndexFlatIP, cosine similarity)",
        }

    def ingest(self, file_paths: list):
        """Load, chunk, embed, and index a list of document file paths."""
        t0 = time.time()

        docs = load_multiple_documents(file_paths)
        self.metrics["num_documents"] = len(docs)

        chunks = chunk_documents(docs, self.chunk_size, self.chunk_overlap)
        self.metrics["num_chunks"] = len(chunks)

        if not chunks:
            raise ValueError("No text could be extracted from the uploaded document(s).")

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.vectorstore.add(embeddings, chunks)

        self.metrics["ingestion_time_sec"] = round(time.time() - t0, 2)
        return self.metrics

    def query(self, question: str, top_k: int = 4) -> dict:
        """
        Answer a question grounded in the ingested documents.

        Returns:
            {
              "answer": str,
              "retrieved_chunks": [ {text, source, chunk_id, score}, ... ],
              "latency_sec": float
            }
        """
        t0 = time.time()

        query_embedding = self.embedder.encode_query(question)
        retrieved = self.vectorstore.search(query_embedding, top_k=top_k)
        answer = self.llm.generate_answer(question, retrieved)

        latency = round(time.time() - t0, 2)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "latency_sec": latency,
        }

    def log_interaction(self, result: dict, log_path: str = "logs/validation_log.jsonl"):
        """Append a query/answer/retrieval record to a JSONL validation log."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        record = {
            "question": result["question"],
            "answer": result["answer"],
            "retrieved_sources": [
                {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
                for c in result["retrieved_chunks"]
            ],
            "latency_sec": result["latency_sec"],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
