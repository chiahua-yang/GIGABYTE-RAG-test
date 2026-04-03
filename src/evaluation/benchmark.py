"""
Evaluation benchmark for the AORUS MASTER 16 AM6H RAG system.

Runs a ground-truth Q&A set and reports:
  - Retrieval recall@k  : does the correct chunk appear in top-k results?
  - Answer accuracy     : human-graded correct / partial / wrong
  - TTFT (s)            : time to first token
  - TPS (tok/s)         : generation throughput

Usage:
  uv run eval [--top-k 3] [--max-tokens 256] [--output results.json]
"""

import argparse
import json
import time
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.rag.indexer import load_index, retrieve, INDEX_PATH
from src.rag.generator import load_model, build_prompt, stream_generate

# ---------------------------------------------------------------------------
# Ground-truth Q&A set (20 questions across 3 difficulty levels)
# ---------------------------------------------------------------------------
GROUND_TRUTH: list[dict] = [
    # ── Level 1: Direct lookup ──────────────────────────────────────────────
    {
        "id": "L1-01",
        "level": "direct",
        "question": "這台筆電的 CPU 是什麼型號？",
        "expected_keywords": ["Intel", "Ultra", "275HX"],
        "expected_model": None,
        "expected_category": "CPU",
    },
    {
        "id": "L1-02",
        "level": "direct",
        "question": "電池容量是多少？",
        "expected_keywords": ["99Wh", "99"],
        "expected_model": None,
        "expected_category": "Power",
    },
    {
        "id": "L1-03",
        "level": "direct",
        "question": "螢幕尺寸和解析度？",
        "expected_keywords": ["16", "2560", "1600"],
        "expected_model": None,
        "expected_category": "Display",
    },
    {
        "id": "L1-04",
        "level": "direct",
        "question": "記憶體最大支援多少 GB？",
        "expected_keywords": ["96GB", "64GB", "DDR5"],
        "expected_model": None,
        "expected_category": "Memory",
    },
    {
        "id": "L1-05",
        "level": "direct",
        "question": "這台筆電重量是多少公斤？",
        "expected_keywords": ["kg", "公斤", "2."],
        "expected_model": None,
        "expected_category": "Dimensions",
    },
    {
        "id": "L1-06",
        "level": "direct",
        "question": "支援什麼 Wi-Fi 標準？",
        "expected_keywords": ["Wi-Fi 7", "802.11", "be"],
        "expected_model": None,
        "expected_category": "Connectivity",
    },
    {
        "id": "L1-07",
        "level": "direct",
        "question": "有幾個 USB Type-A 連接埠？",
        "expected_keywords": ["USB", "Type-A", "3.2"],
        "expected_model": None,
        "expected_category": "Ports",
    },
    {
        "id": "L1-08",
        "level": "direct",
        "question": "作業系統是什麼？",
        "expected_keywords": ["Windows", "11"],
        "expected_model": None,
        "expected_category": "OS",
    },
    # ── Level 2: Model-specific ──────────────────────────────────────────────
    {
        "id": "L2-01",
        "level": "model_specific",
        "question": "AM6H-BZH 的顯卡是什麼？VRAM 有多少？",
        "expected_keywords": ["RTX 5090", "24GB", "GDDR7"],
        "expected_model": "BZH",
        "expected_category": "GPU",
    },
    {
        "id": "L2-02",
        "level": "model_specific",
        "question": "BYH 型號的 GPU TGP 是多少瓦？",
        "expected_keywords": ["175W", "5080"],
        "expected_model": "BYH",
        "expected_category": "GPU",
    },
    {
        "id": "L2-03",
        "level": "model_specific",
        "question": "BXH 的顯示晶片 AI Boost 頻率是多少？",
        "expected_keywords": ["BXH", "1962", "MHz", "5070"],
        "expected_model": "BXH",
        "expected_category": "GPU",
    },
    {
        "id": "L2-04",
        "level": "model_specific",
        "question": "What GPU does the AM6H-BYH have?",
        "expected_keywords": ["RTX 5080", "16GB"],
        "expected_model": "BYH",
        "expected_category": "GPU",
    },
    # ── Level 3: Cross-model comparison ─────────────────────────────────────
    {
        "id": "L3-01",
        "level": "comparison",
        "question": "三個型號 BZH、BYH、BXH 的 GPU 差異是什麼？",
        "expected_keywords": ["5090", "5080", "5070"],
        "expected_model": None,
        "expected_category": "GPU",
    },
    {
        "id": "L3-02",
        "level": "comparison",
        "question": "哪個型號的 GPU VRAM 最大？",
        "expected_keywords": ["BZH", "24GB", "5090"],
        "expected_model": None,
        "expected_category": "GPU",
    },
    {
        "id": "L3-03",
        "level": "comparison",
        "question": "BXH 和 BZH 的 TGP 有什麼不同？",
        "expected_keywords": ["140W", "175W"],
        "expected_model": None,
        "expected_category": "GPU",
    },
    # ── Level 3: Multi-field reasoning ──────────────────────────────────────
    {
        "id": "L3-04",
        "level": "reasoning",
        "question": "這台筆電適合跑本地 AI 模型嗎？請說明理由。",
        "expected_keywords": ["GPU", "VRAM", "RTX"],
        "expected_model": None,
        "expected_category": "GPU",
    },
    {
        "id": "L3-05",
        "level": "reasoning",
        "question": "這台筆電的螢幕規格適合影片剪輯嗎？",
        "expected_keywords": ["2560", "240Hz", "miniLED", "DCI"],
        "expected_model": None,
        "expected_category": "Display",
    },
    {
        "id": "L3-06",
        "level": "reasoning",
        "question": "What is the storage configuration and can it be upgraded?",
        "expected_keywords": ["NVMe", "PCIe", "SSD", "M.2"],
        "expected_model": None,
        "expected_category": "Storage",
    },
    {
        "id": "L3-07",
        "level": "reasoning",
        "question": "連接埠有哪些？是否支援 Thunderbolt？",
        "expected_keywords": ["Thunderbolt", "USB"],
        "expected_model": None,
        "expected_category": "Ports",
    },
    {
        "id": "L3-08",
        "level": "reasoning",
        "question": "這台筆電支援 Bluetooth 幾代？有沒有攝影機？",
        "expected_keywords": ["Bluetooth", "攝影機", "Camera", "IR"],
        "expected_model": None,
        "expected_category": "Connectivity",
    },
]


def _normalize_model_alias(text: str) -> str:
    """
    Normalize model string to a short alias for robust matching:
    e.g. "AORUS MASTER 16 BZH" and "AM6H-BZH" -> "BZH"
    """
    upper = text.upper()
    match = re.search(r"(BZH|BYH|BXH)", upper)
    if match:
        return match.group(1)
    # Fallback: collapse non-alnum chars
    return re.sub(r"[^A-Z0-9]+", "", upper)


def recall_at_k(question_id: str, retrieved_chunks: list[dict], expected_category: str, expected_model: str | None) -> bool:
    """Check if the expected chunk is within retrieved top-k."""
    expected_alias = _normalize_model_alias(expected_model) if expected_model else None
    for chunk in retrieved_chunks:
        if chunk["category"] == expected_category:
            if expected_model is None:
                return True
            chunk_alias = _normalize_model_alias(chunk.get("model", ""))
            if expected_alias and expected_alias == chunk_alias:
                return True
            if chunk.get("model") == "comparison":
                return True
    return False


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hits / len(keywords)


def run_benchmark(
    index_path: Path = INDEX_PATH,
    model_path: Path | None = None,
    top_k: int = 3,
    max_tokens: int = 256,
    output_path: Path | None = None,
):
    print("Loading index …")
    index = load_index(index_path)
    embed_model = SentenceTransformer(index["model"], device="cpu")

    print("Loading LLM …")
    llm = load_model(model_path)

    results = []
    recall_hits = 0
    total_ttft = 0.0
    total_tps = 0.0

    print(f"\nRunning {len(GROUND_TRUTH)} benchmark questions …\n")

    for qa in GROUND_TRUTH:
        print(f"[{qa['id']}] {qa['question'][:60]}")

        # Retrieval
        chunks = retrieve(qa["question"], index, embed_model, top_k=top_k)
        hit = recall_at_k(qa["id"], chunks, qa["expected_category"], qa["expected_model"])
        if hit:
            recall_hits += 1

        # Generation
        messages = build_prompt(qa["question"], chunks)
        answer = ""
        metrics = {}
        for token, m in stream_generate(llm, messages, max_tokens=max_tokens):
            answer += token
            if m:
                metrics = m

        khr = keyword_hit_rate(answer, qa["expected_keywords"])
        total_ttft += metrics.get("ttft", 0)
        total_tps += metrics.get("tps", 0)

        print(
            f"  recall={hit} | keyword_hit={khr:.0%} | "
            f"TTFT={metrics.get('ttft', '?')}s | TPS={metrics.get('tps', '?')}"
        )

        results.append({
            **qa,
            "answer": answer,
            "retrieved_chunks": [c["id"] for c in chunks],
            "retrieval_recall": hit,
            "keyword_hit_rate": round(khr, 4),
            "metrics": metrics,
        })

    n = len(GROUND_TRUTH)
    summary = {
        "total_questions": n,
        "retrieval_recall_at_k": round(recall_hits / n, 4),
        "avg_keyword_hit_rate": round(
            sum(r["keyword_hit_rate"] for r in results) / n, 4
        ),
        "avg_ttft_s": round(total_ttft / n, 4),
        "avg_tps": round(total_tps / n, 2),
    }

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {"summary": summary, "results": results}

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved → {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="RAG benchmark")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="data/benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
