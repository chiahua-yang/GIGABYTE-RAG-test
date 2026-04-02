"""
Interactive RAG server (CLI).

Usage:
  uv run serve
  # or
  python -m src.rag.server
"""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.rag.indexer import load_index, retrieve, EMBED_MODEL, INDEX_PATH
from src.rag.generator import load_model, build_prompt, stream_generate


def run_interactive(
    index_path: Path = INDEX_PATH,
    model_path: Path | None = None,
    top_k: int = 3,
):
    # Load index
    if not index_path.exists():
        print(f"Index not found at {index_path}. Run `uv run index` first.")
        sys.exit(1)

    print("Loading index …")
    index = load_index(index_path)

    print("Loading embedding model …")
    embed_model = SentenceTransformer(index["model"], device="cpu")

    print("Loading LLM …")
    llm = load_model(model_path)

    print("\nAORUS MASTER 16 AM6H RAG Assistant")
    print("支援繁體中文與英文提問 | Type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            query = input("\n你的問題: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        # Retrieve relevant chunks
        chunks = retrieve(query, index, embed_model, top_k=top_k)
        print(f"\n[Retrieval] Top-{top_k} chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. [{c['model']}] {c['category']} (score={c['score']:.4f})")

        # Build prompt and stream response
        messages = build_prompt(query, chunks)
        print("\n[Assistant]")

        metrics = None
        for token, m in stream_generate(llm, messages):
            print(token, end="", flush=True)
            if m:
                metrics = m

        print()  # newline after streaming
        if metrics:
            print(
                f"\n[Metrics] TTFT={metrics['ttft']}s | "
                f"TPS={metrics['tps']} tok/s | "
                f"Total={metrics['total_tokens']} tokens"
            )


def main():
    run_interactive()


if __name__ == "__main__":
    main()
