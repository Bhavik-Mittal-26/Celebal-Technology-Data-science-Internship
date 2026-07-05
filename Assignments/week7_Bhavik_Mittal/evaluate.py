"""
Evaluation / Validation Script
--------------------------------
Runs a set of sample questions against ingested documents and produces:
  1. logs/validation_log.jsonl   -> per-question retrieval + answer log
  2. logs/metrics_report.md      -> human-readable system metrics report

Usage:
    python evaluate.py --docs notes.pdf --questions "What is this about?" "Who is the author?"
"""

import argparse
import os
import statistics
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--docs", nargs="+", required=True)
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[
            "What is the main idea of the document?",
            "Summarize the key points in a few sentences.",
            "What conclusions or recommendations are given?",
        ],
    )
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    pipeline = RAGPipeline()
    metrics = pipeline.ingest(args.docs)

    all_scores = []
    all_latencies = []

    print("Running validation questions...\n")
    for q in args.questions:
        result = pipeline.query(q, top_k=args.top_k)
        pipeline.log_interaction(result, log_path="logs/validation_log.jsonl")

        scores = [c["score"] for c in result["retrieved_chunks"]]
        all_scores.extend(scores)
        all_latencies.append(result["latency_sec"])

        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Top similarity score: {max(scores):.3f}" if scores else "No chunks retrieved")
        print("-" * 60)

    avg_score = round(statistics.mean(all_scores), 3) if all_scores else 0
    avg_latency = round(statistics.mean(all_latencies), 2) if all_latencies else 0

    os.makedirs("logs", exist_ok=True)
    report_path = "logs/metrics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG System Metrics Report\n\n")
        f.write("## Configuration\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Validation Run Summary\n")
        f.write(f"- Questions tested: {len(args.questions)}\n")
        f.write(f"- Average top retrieval similarity score: {avg_score}\n")
        f.write(f"- Average end-to-end latency (retrieval + generation): {avg_latency}s\n")
        f.write("\n## Notes\n")
        f.write(
            "- Similarity scores use cosine similarity (via normalized embeddings + FAISS inner product).\n"
            "- Scores closer to 1.0 indicate stronger semantic match between the question and retrieved chunk.\n"
            "- Full per-question logs are available in `logs/validation_log.jsonl`.\n"
        )

    print(f"\nMetrics report written to {report_path}")
    print(f"Full validation logs written to logs/validation_log.jsonl")


if __name__ == "__main__":
    main()
