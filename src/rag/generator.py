"""
LLM generation with llama.cpp (llama-cpp-python).

Model recommendation (fits 4 GB VRAM):
  Qwen2.5-3B-Instruct-Q4_K_M.gguf  (~2.0 GB)
  → Bilingual (zh/en), instruction-tuned, small enough to leave
    headroom for the embedding model running on CPU.

Download via:
  huggingface-hub CLI or the download_model() helper below.

TTFT and TPS are measured and returned with every generation.
"""

import time
from pathlib import Path
from typing import Generator

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_PATH = Path("models") / DEFAULT_MODEL_FILE

SYSTEM_PROMPT = """你是 GIGABYTE AORUS MASTER 16 AM6H 的產品規格專家助手。
請根據提供的規格資料，用繁體中文或英文（依使用者提問語言）簡潔、精確地回答。
若規格資料中沒有相關資訊，請直接說明無法確認，不要猜測。"""


def download_model(
    repo: str = DEFAULT_MODEL_REPO,
    filename: str = DEFAULT_MODEL_FILE,
    local_dir: Path = Path("models"),
) -> Path:
    local_dir.mkdir(exist_ok=True)
    print(f"Downloading {filename} from {repo} …")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(path)


def load_model(
    model_path: Path | None = None,
    n_gpu_layers: int = -1,  # -1 = offload all layers to GPU
    n_ctx: int = 4096,
    verbose: bool = False,
) -> Llama:
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        model_path = download_model()

    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
    )
    return llm


def build_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """Build a chat-format prompt with retrieved context."""
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    user_message = f"""以下是相關的產品規格資料：

{context_text}

---
問題：{query}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def stream_generate(
    llm: Llama,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Generator[tuple[str, dict | None], None, None]:
    """
    Yield (token_text, metrics_or_None) tuples.
    metrics dict is yielded once on the final token:
      { "ttft": float (seconds), "tps": float (tokens/sec), "total_tokens": int }
    """
    t_start = time.perf_counter()
    ttft: float | None = None
    token_count = 0
    t_first_token: float | None = None

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        text = delta.get("content", "")
        if not text:
            continue

        now = time.perf_counter()
        if ttft is None:
            ttft = now - t_start
            t_first_token = now

        token_count += 1
        finish_reason = chunk["choices"][0].get("finish_reason")

        if finish_reason is not None:
            elapsed = now - t_first_token if t_first_token else 1e-9
            tps = token_count / elapsed if elapsed > 0 else 0
            yield text, {
                "ttft": round(ttft, 4),
                "tps": round(tps, 2),
                "total_tokens": token_count,
            }
        else:
            yield text, None
