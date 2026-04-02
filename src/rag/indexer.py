"""
Build and persist a vector index from chunks (data/chunks.json).

Embedding model : BAAI/bge-small-zh-v1.5 (CPU, ~93 MB)
  - 512-token context, 512-dim embeddings
  - Optimised for zh/en mixed queries (Traditional Chinese included)

Index format : plain numpy + pickle (no external vector DB)
  - At ~20 chunks, brute-force cosine similarity is instant.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("data/index.pkl")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# bge models perform better with this instruction prefix for queries
QUERY_PREFIX = "为我找到与以下问题相关的段落："


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict], model_name: str = EMBED_MODEL) -> dict:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks …")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine = dot product after L2 norm
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,  # shape: (N, 512)
        "model": model_name,
    }


def save_index(index: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved → {path}  ({len(index['chunks'])} chunks)")


def load_index(path: Path = INDEX_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve(
    query: str,
    index: dict,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """
    Return top_k most relevant chunks for a query.
    Uses L2-normalised dot product (= cosine similarity).
    """
    q_emb = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    )
    scores = index["embeddings"] @ q_emb  # shape: (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = index["chunks"][idx]
        results.append({**chunk, "score": float(scores[idx])})
    return results


def main():
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(chunks)
    save_index(index, INDEX_PATH)


if __name__ == "__main__":
    main()
