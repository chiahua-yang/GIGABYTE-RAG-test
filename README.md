# GIGABYTE AORUS MASTER 16 AM6H — RAG Q&A System

純 Python 實作的 RAG 系統，無 LangChain / LlamaIndex 依賴。  
支援繁體中文與英文混合提問，運行於 ≤ 4 GB VRAM 環境。

---

## 快速理解

- **專案目的**：把 GIGABYTE 官方規格頁轉成可檢索知識庫，提供可追溯的產品問答。
- **技術重點**：採用 Hybrid Retrieval（Embedding + BM25）降低專有名詞漏召回風險。
- **可靠性**：支援離線 HTML 載入，不依賴即時爬網，降低 Demo 當場失敗機率。
- **可量化指標**：`Recall@k`、`Keyword Hit Rate`、`TTFT`、`TPS` 皆可重現評測。
- **交付形式**：CLI 問答、benchmark 報告（`data/benchmark_results.json`）、可讀文件。

---

## Google Colab 執行（開發／測試推薦）
我以 **Colab + GPU** 為主；下列步驟可完整重現「上傳離線 HTML → 解析 → chunk → embedding → 問答」，無需在本機安裝 `uv`。

### 事前準備

1. 用一般瀏覽器開啟產品規格頁，等表格完整載入（可捲動確認有 RTX、DDR5 等內容）：  
   [AORUS MASTER 16 AM6H 規格頁（台灣）](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
2. **另存為 HTML**：`Ctrl+S`（Mac：`Cmd+S`），類型選 **「網頁，僅 HTML」**（或等同選項）。  
   GIGABYTE 常會封鎖 Colab 出口 IP（403），因此 **必須** 在本機先存好再帶進 Colab。

### 在 Colab 裡怎麼跑

1. 開啟 [Google Colab](https://colab.research.google.com/)，點 **Open notebook**（開啟筆記本）。
2. 在跳出視窗左側選 **GitHub**，搜尋欄貼上 `https://github.com/chiahua-yang/GIGABYTE-RAG-test.git`，按搜尋（放大鏡圖示），在結果中點選 **`colab_demo.ipynb`** 開啟。
3. 開啟後，到 **Runtime → Change runtime type** 選 **T4 GPU**（與 `colab_demo.ipynb` 開頭說明一致）。  
   > 從 GitHub 開啟的是筆記本檔本身；完整程式與資料仍由筆記本 **Section A2** 的 `git clone` 拉到 `/content/GIGABYTE-RAG-test`（分支與 `REPO_URL`／`BRANCH` 請與該格設定一致）。若列表裡看不到預期檔案或版本不對，在 Colab 的 GitHub 檔案介面確認分支是否為 `claude/build-rag-product-qa-2nMZM`。
4. 依 `colab_demo.ipynb` 由上而下執行（或 **Runtime → Run all**；仍建議在 **Section A 結束後手動 Restart** 再續跑，見下項）：
   - **第一次使用**：Section A（安裝）→ **務必 Runtime → Restart session** → Section B（重啟後初始化）→ Section C 起。
   - **Section C1**：依提示用 `files.upload()` 上傳剛才存好的 `.html`，會寫入 `data/spec_page.html`（若該檔已存在會跳過上傳）。
   - 之後 **Section C3** 解析 → **Section D** 語意 chunk 與建立向量索引（embedding）→ **Section E～G** 下載 LLM、載入與互動問答；**一路 Run 完所有 cell** 即完成一輪流程。
5. 若之後 Colab 斷線重連：**runtime 還在** 時通常從 Section B 再接續；**runtime 已釋放** 則需重跑 Section A → Restart → B → 後續段落。

**備選**：若無法用 GitHub 分頁開啟，可到 **Upload** 上傳本機的 `colab_demo.ipynb`，接著照常執行 **Section A2** 完成 clone 即可。

本機 `uv run …` 的流程與筆記本內對應步驟一致；差別在 Colab 以 `pip` 安裝依賴並以 GPU runtime 跑 `llama-cpp-python`。

---

## 系統需求

| 項目 | 需求 |
|------|------|
| Python | ≥ 3.10 |
| 套件管理 | [uv](https://docs.astral.sh/uv/) |
| GPU VRAM | ≥ 4 GB（或 CPU-only 模式） |
| 推論引擎 | llama-cpp-python |

---

## 本機快速啟動（uv）

```bash
# 1. 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安裝依賴
uv sync

# 3. 爬取並解析規格資料 → data/specs.json
uv run scrape

# 4. 語意分塊 → data/chunks.json
uv run chunk

# 5. 建立向量索引 → data/index.pkl
uv run index

# 6. 啟動互動問答
uv run serve
```

### 面試官機器（推薦 Demo 流程：本地 HTML，不依賴即時爬網）

你的資料來源若是先在瀏覽器另存官方頁面，此流程最穩定：

1. 在瀏覽器打開產品頁，等待規格完整載入後另存為 HTML  
2. 將檔案放到 `data/spec_page.html`
3. 依序執行：

```bash
uv sync
uv run scrape
uv run chunk
uv run index
uv run serve
```

> `uv run scrape` 會優先讀取 `data/spec_page.html`；若不存在才會改用線上抓取。  
> 這樣可避免公司網路、地區封鎖或反爬機制造成 demo 失敗。

### Windows PowerShell 指令（面試最實用）

```powershell
# 1) 安裝依賴
uv sync

# 2) 將你準備好的 HTML 複製到固定位置
Copy-Item ".\your_saved_page.html" ".\data\spec_page.html" -Force

# 3) 建立資料與索引
uv run scrape
uv run chunk
uv run index

# 4) 啟動互動問答（含 streaming 與 TTFT/TPS）
uv run serve
```

### 執行評測

```bash
uv run eval --top-k 3 --max-tokens 256 --output data/benchmark_results.json
```

---

## 模型選擇理由（4 GB VRAM 限制）

### LLM：Qwen2.5-3B-Instruct-Q4_K_M（~2.0 GB）

**為何選 Qwen2.5-3B Instruct**

| 考量 | 說明 |
|------|------|
| **繁體中文能力** | 阿里系模型，訓練資料含大量中文（含繁體），整體中文理解通常優於同級 Llama / Mistral。 |
| **3B 的定位** | 在相近中文能力前提下，3B 的 VRAM 需求約為 7B 的一半；相較 1.5B 級別，規格問答與指令遵循通常更穩。 |
| **Instruct 版本** | 經指令微調，較能依提示輸出固定格式與條列，減少閒聊式亂答。 |

**為何用 Q4_K_M 量化**

- **原始 3B FP16** 權重約 **6 GB 級別**，超出本專案 4 GB VRAM 目標。
- **Q4_K_M（4-bit）** 將權重壓到約 **2.0 GB**，才能與 KV cache 同時落在 GPU。
- **K_M（K-quant Medium）** 比一般 **Q4_0** 保留較多精度，在體積與品質之間折衷。

推論使用 **llama-cpp-python**（GGUF），支援 GPU offload 與 CPU fallback。

下載指令（首次 `uv run serve` 時自動觸發，或手動執行）：

```bash
# 手動下載
uv run python -c "from src.rag.generator import download_model; download_model()"
```

### Embedding：BAAI/bge-small-zh-v1.5（~93 MB，CPU）

| 考量 | 說明 |
|------|------|
| **中文語意** | 智源（BAAI）訓練的中文語意向量模型；中文語意表現通常優於 `multilingual-e5-small` 這類泛多語小模型。 |
| **Small 體積** | 約 **93 MB**，**全跑在 CPU**，幾乎不佔 VRAM，把顯存留給 LLM。 |
| **v1.5 與 Query 前綴** | 建議對 **query** 加前綴做**非對稱檢索**（與 passage 編碼分開），有助提高召回。本專案在 `src/rag/indexer.py` 使用 `QUERY_PREFIX`：**「為我找到與以下問題相關的段落：」** + 使用者問題。 |

### 怎麼壓在 4 GB VRAM

```
LLM (Q4_K_M)           ≈ 2.0 GB  ← GPU
KV Cache               ≈ 0.8 GB  ← GPU（推理時動態分配；本專案 n_ctx=4096）
Embedding              ≈ 0.0 GB  ← CPU（不佔 VRAM；權重約 93 MB 在系統 RAM）
────────────────────────────────────────────────────────
合計                   ≈ 2.8 GB < 4 GB 
```

關鍵是 **embedding 全跑在 CPU，不搶 VRAM**。RAG 架構適合這樣切：**embedding 只在建 index 與每次 query 各算一次**，延遲可接受；**瓶頸在 LLM 生成**，必須優先放 GPU。實際執行時若驅動／CUDA context 另有少量開銷，仍通常低於 4 GB 上限；Colab T4 與 4 GB 級顯卡可穩定跑完整「檢索 + 串流生成 + benchmark」。

---

## 架構說明

```
使用者問題
    │
    ▼
[Embedding]  BAAI/bge-small-zh-v1.5 (CPU)
    │         QUERY_PREFIX + query → 512-dim vector
    ▼
[Retrieval]  Hybrid Search = Embedding + BM25
    │         Top-k chunks from row/category/comparison multi-granularity chunks
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

## 評測結果（Colab，20 題）

> 執行環境：`colab_demo.ipynb`（T4 GPU）  
> 指令：`uv run eval --top-k 3 --max-tokens 256 --output data/benchmark_results.json`

### 指標定義（先讀這段再對照數字）

| 符號／指標 | 代表什麼 | 刻意不代表什麼 |
|------------|----------|----------------|
| **✓／✗**（逐題） | 僅 **Retrieval Recall@k**：top-k 檢索結果裡是否出現預期 **category／型號** 所對應的 chunk（與 `colab_demo.ipynb` 輸出一致） | **不是**「整題答案正確」、也不是人工審閱結論 |
| **Keyword Hit（kw）** | 回答文字中含多少個預設 `expected_keywords`（子字串比對、不分大小寫） | **不是**語意正確性的完整判斷；同義改寫、中英不同說法可能分數偏低，敘述仍可能對 |
| **TTFT／TPS** | 首字延遲與生成吞吐 | 與答對與否無直接對應 |

### 定量指標

| 指標 | 結果 |
|------|------|
| total_questions | 20 |
| Retrieval Recall@k | 0.95 |
| Avg Keyword Hit Rate | 0.7333 |
| Avg TTFT | 0.1798 s |
| Avg TPS | 65.12 tok/s |

### 結果解讀

- **檢索為主軸**：Recall@k **0.95** 表示在固定 **k=3** 下，多數題能把「對的規格脈絡」送進上下文；RAG 上游 **定位資料** 的表現穩，適合向讀者強調「可追溯 chunk」而非猜答。
- **關鍵字為輔**：平均 Keyword **~0.73** 與大量 ✓ **並存是正常現象**——代表「找得到 chunk」和「回答用語完全貼齊預設詞」是兩件事；此指標成本低、可重現，適合當 **粗對齊**，不宜單獨當 gold。
- **失敗點集中**：僅 **L3-08** 檢索未命中（單題混問 Bluetooth 與攝影機、且 ground truth 只標單一 category 時，top-k 較吃緊）；其餘 19 題在「是否把對的 chunk 排進上下文」上通過。
- **體驗**：平均 TTFT **~0.18 s**、TPS **~65**，在 Colab T4 + 3B 量化設定下，**互動展示與多題連跑**仍屬流暢區間。

### 每題結果（摘要）

**再次提醒：下列 ✓ 僅表示該題 Retrieval Recall@k 命中，不代表該題敘述已通過完整答案評測。**

```
[L1-01 ] ✓ | kw=100% | ttft=0.3916s | tps=63.42
[L1-02 ] ✓ | kw=100% | ttft=0.0831s | tps=67.37
[L1-03 ] ✓ | kw=100% | ttft=0.2573s | tps=69.87
[L1-04 ] ✓ | kw=33%  | ttft=0.1014s | tps=66.60
[L1-05 ] ✓ | kw=67%  | ttft=0.1129s | tps=66.61
[L1-06 ] ✓ | kw=100% | ttft=0.1034s | tps=67.14
[L1-07 ] ✓ | kw=67%  | ttft=0.2602s | tps=62.52
[L1-08 ] ✓ | kw=100% | ttft=0.0971s | tps=66.28
[L2-01 ] ✓ | kw=67%  | ttft=0.1595s | tps=67.17
[L2-02 ] ✓ | kw=50%  | ttft=0.2537s | tps=65.62
[L2-03 ] ✓ | kw=75%  | ttft=0.1596s | tps=58.20
[L2-04 ] ✓ | kw=0%   | ttft=0.1600s | tps=64.95
[L3-01 ] ✓ | kw=100% | ttft=0.2569s | tps=66.62
[L3-02 ] ✓ | kw=67%  | ttft=0.1182s | tps=68.83
[L3-03 ] ✓ | kw=50%  | ttft=0.0942s | tps=67.22
[L3-04 ] ✓ | kw=67%  | ttft=0.1259s | tps=64.04
[L3-05 ] ✓ | kw=50%  | ttft=0.2572s | tps=62.09
[L3-06 ] ✓ | kw=100% | ttft=0.2328s | tps=64.17
[L3-07 ] ✓ | kw=100% | ttft=0.2678s | tps=63.12
[L3-08 ] ✗ | kw=75%  | ttft=0.1032s | tps=60.62
```

### 可重現評測流程（建議在面試前先跑完一次）

```bash
uv run eval --top-k 3 --max-tokens 256 --output data/benchmark_results.json
```

面試時可直接展示：
- `data/benchmark_results.json`（完整每題結果）
- README 的定量表格（平均 TTFT / TPS / Recall）

### 測試題目設計

共 20 題，分三個難度層次：

| 層次 | 題數 | 說明 |
|------|------|------|
| Level 1 直接查找 | 8 | 單一 key-value 查詢（CPU、電池、重量…） |
| Level 2 型號指定 | 4 | 需識別特定子型號（BZH/BYH/BXH）的規格 |
| Level 3 比較/推理 | 8 | 跨型號比較、多欄位綜合推理 |

評測方式（與「評測結果」一節的 **指標定義** 表一致；✓ 僅對應 Recall，不表示整題答對）：
- **Retrieval Recall@k**：正確 chunk 是否出現在 top-k 結果中
- **Keyword Hit Rate**：回答中包含預期關鍵字的比例（自動計算）
- **TTFT / TPS**：每題皆記錄，取平均值

### 你可自行設計題目與預期答案（建議格式）

建議每題都寫成以下欄位，方便主管快速理解可驗證性：

- `question`：問題（可中英混合）
- `expected_keywords`：預期答案關鍵字（2-4 個）
- `expected_category`：預期命中的規格類別（CPU/GPU/Display...）
- `expected_model`：若是型號題則填 BZH/BYH/BXH，否則留空

範例：

```json
{
  "question": "AM6H-BZH 的顯卡與 VRAM 是什麼？",
  "expected_keywords": ["RTX 5090", "24GB", "GDDR7"],
  "expected_category": "GPU",
  "expected_model": "BZH"
}
```

---

## 目前限制與已知風險

- 規格頁可能有地區封鎖（403）；建議使用本地 `data/spec_page.html` 確保流程穩定。
- 推理型問題（例如「是否適合某用途」）屬半開放題，適合人工複核，不建議作為唯一自動化指標。
- benchmark 的自動評分以關鍵字為主，仍需搭配抽樣人工檢查敘述正確性。

---

## 目錄結構

```
.
├── pyproject.toml          # uv 環境設定
├── README.md
├── colab_demo.ipynb        # Google Colab 逐步執行（GPU、上傳離線 HTML）
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
