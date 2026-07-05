"""
Vector Store Module
---------------------
Stores chunk embeddings in a local FAISS index for fast similarity search.
Fully in-memory / on-disk — no external vector DB service or API key required.
"""

import faiss
import numpy as np
import pickle
import os


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []  # parallel list: metadata[i] describes vector i

    def add(self, embeddings: np.ndarray, metadata: list):
        """
        Add vectors + their metadata (text, source, chunk_id) to the index.
        """
        assert embeddings.shape[0] == len(metadata), "Embeddings/metadata length mismatch"
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list:
        """
        Search for the top_k most similar chunks to the query embedding.

        Returns:
            List of dicts: [{"text":.., "source":.., "chunk_id":.., "score":..}, ...]
            sorted by descending similarity score.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = np.expand_dims(query_embedding, axis=0)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, folder: str):
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
