"""
evaluation.py
--------------
Evaluates RAG answer quality using RAGAs metrics:
  - Faithfulness: is the answer grounded in the retrieved context (no hallucination)?
  - Answer Relevance: does the answer actually address the question?
  - Context Precision: how relevant/well-ranked are the retrieved chunks?

Requires GROQ_API_KEY (RAGAs uses an LLM internally to judge these metrics).
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def evaluate_answer(question: str, answer: str, contexts: list[str], ground_truth: str = None):
    """
    Score a single Q&A pair.

    contexts: list of the retrieved chunk texts used to generate the answer.
    ground_truth: optional reference answer, improves relevancy scoring if provided.
    """
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    metrics = [faithfulness, answer_relevancy]
    if ground_truth:
        metrics.append(context_precision)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return result.to_pandas()
