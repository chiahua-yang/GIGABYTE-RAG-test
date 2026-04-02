# GIGABYTE AORUS MASTER 16 AM6H — RAG Q&A System

純 Python 實作的 RAG 系統，無 LangChain / LlamaIndex 依賴。  
支援繁體中文與英文混合提問，運行於 ≤ 4 GB VRAM 環境。

---

## 系統需求

| 項目 | 需求 |
|------|------|
| Python | ≥ 3.10 |
| 套件管理 | [uv](https://docs.astral.sh/uv/) |
| GPU VRAM | ≥ 4 GB（或 CPU-only 模式） |
| 推論引擎 | llama-cpp-python |

---

## 快速啟動

```bash
# 1. 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安裝依賴
uv sync

# 3. 爬取並解析規格資料 → data/specs.json
uv run scrape

# 4. 分塊並建立向量索引 → data/index.pkl
uv run index

# 5. 啟動互動問答
uv run serve
```

### 執行評測

```bash
uv run eval --top-k 3 --max-tokens 256 --output data/benchmark_results.json
```

---

## 模型選擇理由（4 GB VRAM 限制）

### LLM：Qwen2.5-3B-Instruct-Q4_K_M（~2.0 GB）

| 考量 | 說明 |
|------|------|
| **記憶體佔用** | Q4_K_M 量化後 ≈ 2.0 GB，在 4 GB VRAM 下有充裕空間 |
| **雙語能力** | Qwen2.5 系列原生支援繁體中文與英文，對規格術語識別優異 |
| **指令遵循** | Instruct 版本在 Q&A 任務準確遵守「不猜測、引用來源」等指令 |
| **推論引擎** | llama-cpp-python GGUF 格式，支援 GPU offload 及 CPU fallback |

下載指令（首次 `uv run serve` 時自動觸發，或手動執行）：

```bash
# 手動下載
uv run python -c "from src.rag.generator import download_model; download_model()"
```

### Embedding Model：BAAI/bge-small-zh-v1.5（~93 MB，CPU）

| 考量 | 說明 |
|------|------|
| **體積** | 93 MB，完全不佔 VRAM，所有 VRAM 預算留給 LLM |
| **語言覆蓋** | 針對中文優化，對繁體中文 query 召回率優於純英文模型 |
| **速度** | 20 個 chunk 的 CPU 推論 < 100 ms，不是瓶頸 |

### VRAM 使用分配

```
LLM  (Qwen2.5-3B Q4_K_M, GPU offload) ≈ 2.0 GB
KV Cache (n_ctx=4096)                  ≈ 0.8 GB
其他 (CUDA context, etc.)              ≈ 0.2 GB
────────────────────────────────────────────────
Total Peak                             ≈ 3.0 GB  (< 4 GB ✓)

Embedding Model (CPU)                  ≈ 93 MB RAM
```

---

## 架構說明

```
使用者問題
    │
    ▼
[Embedding]  BAAI/bge-small-zh-v1.5 (CPU)
    │         QUERY_PREFIX + query → 512-dim vector
    ▼
[Retrieval]  Cosine similarity (numpy brute-force)
    │         Top-3 chunks from ~20 semantic chunks
    ▼
[Generation] Qwen2.5-3B-Instruct-Q4_K_M (llama-cpp-python)
    │         Streaming output with TTFT / TPS metrics
    ▼
   回答（串流輸出）
```

### Chunking 策略

規格資料為 key-value 結構，**不適合** fixed-size text splitting。採用以下方式：

1. **語意分組**：將相關規格合併（CPU、GPU、Display、Ports…等共 15 個語意群組）
2. **型號獨立**：BZH / BYH / BXH 三個子型號各自獨立的 chunks，chunk prefix 帶型號名稱
3. **跨型號比較 chunk**：針對規格有差異的類別（主要是 GPU）額外建立比較 chunk
4. **Token 控制**：每個 chunk ≤ 200 tokens（遠低於 bge-small-zh 的 512 上限）

---

## 評測結果

> 測試環境：[待填入硬體規格]  
> 模型：Qwen2.5-3B-Instruct-Q4_K_M + bge-small-zh-v1.5

### 定量指標

| 指標 | 數值 |
|------|------|
| TTFT（首字延遲） | — s |
| TPS（生成速度） | — tok/s |
| Retrieval Recall@3 | — |
| Keyword Hit Rate | — |

> 執行 `uv run eval` 後將結果填入此表。

### 測試題目設計

共 20 題，分三個難度層次：

| 層次 | 題數 | 說明 |
|------|------|------|
| Level 1 直接查找 | 8 | 單一 key-value 查詢（CPU、電池、重量…） |
| Level 2 型號指定 | 4 | 需識別特定子型號（BZH/BYH/BXH）的規格 |
| Level 3 比較/推理 | 8 | 跨型號比較、多欄位綜合推理 |

評測方式：
- **Retrieval Recall@k**：正確 chunk 是否出現在 top-k 結果中
- **Keyword Hit Rate**：回答中包含預期關鍵字的比例（自動計算）
- **TTFT / TPS**：每題皆記錄，取平均值

---

## 目錄結構

```
.
├── pyproject.toml          # uv 環境設定
├── README.md
├── src/
│   ├── data/
│   │   ├── scraper.py      # 網頁爬取與 HTML 解析
│   │   └── chunker.py      # 語意分組 chunking
│   ├── rag/
│   │   ├── indexer.py      # Embedding + 向量檢索
│   │   ├── generator.py    # LLM 串流生成 + TTFT/TPS 量測
│   │   └── server.py       # 互動式 CLI
│   └── evaluation/
│       └── benchmark.py    # 20 題 benchmark + 自動評分
├── data/                   # 執行後產生（specs.json, index.pkl…）
└── tests/
```
