"""
Build and persist a vector index from chunks (data/chunks.json).

Embedding model : BAAI/bge-small-zh-v1.5 (CPU, ~93 MB)
  - 512-token context, 512-dim embeddings
  - Optimised for zh/en mixed queries (Traditional Chinese included)

Retrieval : Hybrid BM25 + embedding cosine similarity
  - BM25 handles exact keyword matches (e.g. "CPU", "RTX", "DDR5")
  - Embedding handles semantic similarity
  - Final score = alpha * embedding + (1-alpha) * bm25 (both normalised)
"""

import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("data/index.pkl")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# bge instruction prefix (Traditional Chinese)
QUERY_PREFIX = "為我找到與以下問題相關的段落："

# Weight for embedding vs BM25 (0 = pure BM25, 1 = pure embedding)
HYBRID_ALPHA = 0.7


# ---------------------------------------------------------------------------
# BM25 (no external dependency)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for zh/en mixed text.
    - Splits on whitespace for English words
    - Extracts individual CJK characters
    - Lowercases everything
    """
    tokens: list[str] = []
    for word in text.lower().split():
        tokens.append(word)
    for char in text:
        if "\u4e00" <= char <= "\u9fff":   # CJK Unified Ideographs
            tokens.append(char)
    return tokens


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.tokenized = [_tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.tokenized]
        self.avgdl = sum(self.doc_lens) / self.n if self.n else 1.0

        df: Counter = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                df[term] += 1

        self.idf: dict[str, float] = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = _tokenize(query)
        result = np.zeros(self.n)
        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.tokenized):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                result[i] += idf * num / den
        return result


# Ensure pickle records the canonical module path regardless of how this
# file is executed (e.g. `python -m src.rag.indexer` sets __name__=="__main__")
BM25.__module__ = "src.rag.indexer"


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------

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
        normalize_embeddings=True,
    )

    print("Building BM25 index …")
    bm25 = BM25(texts)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "bm25": bm25,
        "model": model_name,
    }


def save_index(index: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved → {path}  ({len(index['chunks'])} chunks)")


def load_index(path: Path = INDEX_PATH) -> dict:
    import sys
    # pickle may have stored BM25 as __main__.BM25 (when built inside a notebook);
    # inject it into __main__ so deserialization succeeds without rebuilding.
    main = sys.modules.setdefault("__main__", sys.modules[__name__])
    if not hasattr(main, "BM25"):
        main.BM25 = BM25
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index: dict,
    model: SentenceTransformer,
    top_k: int = 3,
    alpha: float = HYBRID_ALPHA,
) -> list[dict]:
    """
    Hybrid BM25 + embedding retrieval.

    Both scores are min-max normalised to [0, 1] before combining:
      final = alpha * embedding_score + (1 - alpha) * bm25_score
    """
    # --- Embedding score ---
    q_emb = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    emb_scores = index["embeddings"] @ q_emb        # cosine in [-1, 1]

    # --- BM25 score ---
    bm25: BM25 = index.get("bm25")
    if bm25 is not None:
        bm25_raw = bm25.scores(query)
    else:
        bm25_raw = np.zeros(len(index["chunks"]))

    # --- Normalise both to [0, 1] ---
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    combined = alpha * _minmax(emb_scores) + (1 - alpha) * _minmax(bm25_raw)

    top_indices = np.argsort(combined)[::-1][:top_k]
    return [
        {**index["chunks"][i], "score": float(combined[i])}
        for i in top_indices
    ]


def main():
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(chunks)
    save_index(index, INDEX_PATH)


if __name__ == "__main__":
    main()
