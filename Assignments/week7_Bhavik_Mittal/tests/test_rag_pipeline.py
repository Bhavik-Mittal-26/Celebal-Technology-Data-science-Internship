import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.rag_pipeline import RAGPipeline


class RAGPipelinePerformanceTests(unittest.TestCase):
    def test_llm_is_loaded_lazily_for_queries(self):
        with patch("src.rag_pipeline.EmbeddingModel") as embedding_model_cls, patch(
            "src.rag_pipeline.VectorStore"
        ) as vectorstore_cls, patch("src.rag_pipeline.LocalLLM") as llm_cls:
            embedder = Mock()
            embedder.embedding_dim = 3
            embedder.encode_query.return_value = np.array([0.1, 0.2, 0.3], dtype="float32")
            embedding_model_cls.return_value = embedder

            vectorstore = Mock()
            vectorstore.search.return_value = [{"text": "context", "source": "doc.pdf", "chunk_id": 1}]
            vectorstore_cls.return_value = vectorstore

            llm = Mock()
            llm.generate_answer.return_value = "Answer"
            llm_cls.return_value = llm

            pipeline = RAGPipeline()

            self.assertIsNone(pipeline.llm)
            self.assertEqual(llm_cls.call_count, 0)

            result = pipeline.query("What is this?", top_k=1)

            self.assertEqual(result["answer"], "Answer")
            self.assertEqual(llm_cls.call_count, 1)
            llm.generate_answer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
