"""
Convert parsed spec records (data/specs.json) into RAG chunks (data/chunks.json).

Chunking strategy:
  - Each chunk = one semantic group (e.g. GPU, Display) for one SKU
  - Plus one cross-model comparison chunk per semantic group
  - This keeps chunks ≤ 200 tokens while preserving relational context
"""

import json
from pathlib import Path
from collections import defaultdict

SPECS_PATH = Path("data/specs.json")
CHUNKS_PATH = Path("data/chunks.json")

# Ordered display names for semantic groups
GROUP_ORDER = [
    "CPU",
    "OS",
    "Memory",
    "Storage",
    "Display",
    "GPU",
    "Camera",
    "Connectivity",
    "Audio",
    "Input",
    "Ports",
    "Power",
    "Dimensions",
    "Security",
    "Accessories",
    "General",
]


def load_specs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_chunks(records: list[dict]) -> list[dict]:
    """
    Group records by (model, category) → one chunk per group per model.
    Also build cross-model comparison chunks for categories that differ across SKUs.
    """
    # { model: { category: [(key, value), ...] } }
    grouped: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        grouped[r["model"]][r["category"]].append((r["key"], r["value"]))

    chunks: list[dict] = []

    all_categories = set()
    for model_data in grouped.values():
        all_categories.update(model_data.keys())

    # Per-model chunks
    for model, cat_map in grouped.items():
        for category in GROUP_ORDER:
            if category not in cat_map:
                continue
            pairs = cat_map[category]
            text = f"[{model}] {category}\n"
            text += "\n".join(f"  {k}: {v}" for k, v in pairs)
            chunks.append(
                {
                    "id": f"{model}::{category}",
                    "model": model,
                    "category": category,
                    "text": text,
                }
            )

    # Cross-model comparison chunks (GPU is always interesting to compare)
    models = list(grouped.keys())
    for category in GROUP_ORDER:
        models_with_cat = [m for m in models if category in grouped[m]]
        if len(models_with_cat) < 2:
            continue

        # Check if values differ across models
        first_pairs = grouped[models_with_cat[0]][category]
        all_same = all(
            grouped[m][category] == first_pairs for m in models_with_cat[1:]
        )
        if all_same:
            continue  # No need to compare identical specs

        text = f"[型號比較] {category}\n"
        for m in models_with_cat:
            pairs = grouped[m][category]
            text += f"\n  {m}:\n"
            text += "\n".join(f"    {k}: {v}" for k, v in pairs)

        chunks.append(
            {
                "id": f"comparison::{category}",
                "model": "comparison",
                "category": category,
                "text": text,
            }
        )

    return chunks


def save_chunks(chunks: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks → {path}")


def main():
    records = load_specs(SPECS_PATH)
    chunks = build_chunks(records)
    save_chunks(chunks, CHUNKS_PATH)
    for c in chunks[:2]:
        print("\n---")
        print(c["text"])


if __name__ == "__main__":
    main()
