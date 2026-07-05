"""
CLI entry point — run the RAG pipeline end-to-end from the terminal.

Usage:
    python run_cli.py --docs notes.pdf resume.pdf --question "What is this document about?"

    # Or run interactively (keeps asking questions until you type 'exit'):
    python run_cli.py --docs notes.pdf
"""

import argparse
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Local RAG Document Question Answering")
    parser.add_argument("--docs", nargs="+", required=True, help="Paths to PDF/TXT/DOCX files")
    parser.add_argument("--question", type=str, default=None, help="A single question to ask (optional)")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("Loading models and building index (no API key used)...")
    pipeline = RAGPipeline()
    metrics = pipeline.ingest(args.docs)

    print("\n=== System Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=======================\n")

    def ask(question):
        result = pipeline.query(question, top_k=args.top_k)
        pipeline.log_interaction(result)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"(retrieved {len(result['retrieved_chunks'])} chunks in {result['latency_sec']}s)")
        for c in result["retrieved_chunks"]:
            print(f"   - [{c['source']} chunk#{c['chunk_id']}] score={c['score']:.3f}")

    if args.question:
        ask(args.question)
    else:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            q = input("\nAsk a question: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q:
                ask(q)


if __name__ == "__main__":
    main()
