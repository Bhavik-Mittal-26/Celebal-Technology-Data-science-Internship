"""
Embedding Module
-----------------
Converts text chunks into dense vector representations using a
pre-trained sentence-transformers model. Runs 100% locally after the
first download (weights are cached by Hugging Face — no API key needed
since this is a public open-source model).
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        all-MiniLM-L6-v2:
          - 384-dimensional embeddings
          - Fast on CPU
          - Strong general-purpose semantic search quality
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        return self.model

    def encode(self, texts: list) -> np.ndarray:
        """Encode a list of strings into a (N, embedding_dim) float32 array."""
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # so we can use cosine similarity via inner product
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]
