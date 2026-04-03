"""
Scrape and parse GIGABYTE AORUS MASTER 16 AM6H spec page.
Produces data/specs.json — a list of {model, category, key, value} records.

Parsing strategy
----------------
The page is a Nuxt.js SSR app.  The spec data for all three sub-models
(BZH / BYH / BXH) is embedded as a deduped JSON array inside a <script>
tag.  We parse that JSON directly instead of scraping HTML elements,
because:
  - The rendered HTML only shows ONE model's specs (the active tab).
  - The embedded JSON contains ALL three models' complete spec data.
  - JSON parsing is more robust than HTML selector-based parsing.

Fallback: if the JSON cannot be found/parsed, fall back to the
<ul class="spec-item-list"> HTML structure (which covers the active model).
"""

import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SPEC_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
LOCAL_HTML_PATH = Path("data/spec_page.html")

# Actual model title strings as they appear in the page JSON
MODEL_TITLES = [
    "AORUS MASTER 16 BZH",
    "AORUS MASTER 16 BYH",
    "AORUS MASTER 16 BXH",
]

# Map Chinese spec key substrings → semantic group names
CATEGORY_GROUPS = {
    "中央處理器": "CPU",
    "處理器": "CPU",
    "晶片組": "CPU",
    "作業系統": "OS",
    "記憶體": "Memory",
    "儲存裝置": "Storage",
    "顯示器": "Display",
    "顯示晶片": "GPU",
    "視訊鏡頭": "Camera",
    "攝影機": "Camera",
    "通訊": "Connectivity",
    "網路": "Connectivity",
    "音效": "Audio",
    "鍵盤": "Input",
    "觸控板": "Input",
    "電池": "Power",
    "變壓器": "Power",
    "電源": "Power",
    "尺寸": "Dimensions",
    "重量": "Dimensions",
    "規格": "Dimensions",
    "顏色": "General",
    "連接埠": "Ports",
    "介面": "Ports",
    "安全": "Security",
    "隨附": "Accessories",
}


# ---------------------------------------------------------------------------
# HTML loading
# ---------------------------------------------------------------------------

def load_or_fetch_html() -> str:
    """
    Load HTML from data/spec_page.html (preferred) or fetch from URL.

    Colab / cloud IPs are blocked (403) by GIGABYTE.
    Workaround: save the page from your browser and upload it.
      1. Open https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp
      2. Wait for specs to fully load
      3. Ctrl+S → "Webpage, HTML Only"
      4. Upload to data/spec_page.html
    """
    if LOCAL_HTML_PATH.exists():
        print(f"Loading local HTML: {LOCAL_HTML_PATH}")
        return LOCAL_HTML_PATH.read_text(encoding="utf-8", errors="replace")

    print(f"Fetching {SPEC_URL} ...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(SPEC_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch page: {e}\n"
            "If running on Colab, the IP is likely blocked (403).\n"
            "Save the page from your browser and place it at data/spec_page.html"
        ) from e


# ---------------------------------------------------------------------------
# Primary parser: Nuxt SSR deduped JSON
# ---------------------------------------------------------------------------

def _resolve(data: list, idx):
    """
    Recursively resolve a value in Nuxt's deduped JSON array.

    In Nuxt SSR format the entire state is serialised as a flat array.
    Objects and arrays store integer indices instead of inline values;
    an integer N means "look up data[N]".  Strings, booleans and None
    are stored as-is.
    """
    if isinstance(idx, int) and 0 <= idx < len(data):
        val = data[idx]
        if isinstance(val, dict):
            return {k: _resolve(data, v) for k, v in val.items()}
        if isinstance(val, list):
            return [_resolve(data, v) for v in val]
        return val
    return idx


def _html_to_text(html_str: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", html_str)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _map_category(key: str) -> str:
    for kw, group in CATEGORY_GROUPS.items():
        if kw in key:
            return group
    return "General"


def parse_specs_from_json(html: str) -> list[dict]:
    """
    Extract specs from the Nuxt SSR deduped JSON blob embedded in <script>.

    Returns a list of {model, category, key, value} records for all 3 models,
    or [] if the JSON blob cannot be found or parsed.
    """
    soup = BeautifulSoup(html, "lxml")

    for script in soup.find_all("script"):
        content = script.string or ""
        # The right script contains both tabSpec and itemTitle
        if "tabSpec" not in content or "itemTitle" not in content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, list):
            continue

        # Find the dict entry that has a "tabSpec" key
        tab_spec_resolved = None
        for item in data:
            if isinstance(item, dict) and "tabSpec" in item:
                tab_spec_resolved = _resolve(data, item["tabSpec"])
                break

        if not tab_spec_resolved or not isinstance(tab_spec_resolved, list):
            continue

        records: list[dict] = []
        for model_entry in tab_spec_resolved:
            if not isinstance(model_entry, dict):
                continue

            model_name = model_entry.get("title", "Unknown")
            spec_items = model_entry.get("specItem", [])
            if not isinstance(spec_items, list):
                continue

            for item in spec_items:
                if not isinstance(item, dict):
                    continue
                key = item.get("itemTitle", "")
                value_raw = item.get("itemContent", "")
                value = _html_to_text(str(value_raw))
                if key and value:
                    records.append({
                        "model": model_name,
                        "category": _map_category(key),
                        "key": key,
                        "value": value,
                    })

        if records:
            print(f"Parsed {len(records)} records from embedded JSON.")
            return records

    return []


# ---------------------------------------------------------------------------
# Fallback parser: <ul class="spec-item-list"> HTML structure
# (covers the single active model tab only)
# ---------------------------------------------------------------------------

def parse_specs_from_html(html: str) -> list[dict]:
    """
    Fallback: parse <li class='spec-title'> / <li class='spec-desc'> pairs.
    Only the currently selected model tab is present in the rendered HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []

    # Try to detect which model is currently shown
    active_model = "AORUS MASTER 16 AM6H"
    for candidate in MODEL_TITLES:
        if candidate in html:
            active_model = candidate
            break

    for ul in soup.find_all("ul", class_="spec-item-list"):
        items = ul.find_all("li")
        i = 0
        while i + 1 < len(items):
            title_li = items[i]
            desc_li = items[i + 1]
            if (
                "spec-title" in (title_li.get("class") or [])
                and "spec-desc" in (desc_li.get("class") or [])
            ):
                key = title_li.get_text(strip=True)
                value = desc_li.get_text(separator=" ", strip=True)
                value = re.sub(r"\s+", " ", value).strip()
                if key and value:
                    records.append({
                        "model": active_model,
                        "category": _map_category(key),
                        "key": key,
                        "value": value,
                    })
                i += 2
            else:
                i += 1

    if records:
        print(f"Parsed {len(records)} records from HTML spec-item-list "
              f"(active model only).")
    return records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_specs(html: str) -> list[dict]:
    """Try JSON parsing first; fall back to HTML parsing."""
    records = parse_specs_from_json(html)
    if records:
        return records

    print("JSON parsing yielded no records, trying HTML fallback ...",
          file=sys.stderr)
    return parse_specs_from_html(html)


def save_specs(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records → {out_path}")


def main():
    out_path = Path("data/specs.json")
    html = load_or_fetch_html()
    records = parse_specs(html)

    if not records:
        print("ERROR: No records parsed.", file=sys.stderr)
        print("Possible causes:", file=sys.stderr)
        print("  1. The saved HTML is incomplete (page not fully loaded before saving)", file=sys.stderr)
        print("  2. GIGABYTE updated the page structure", file=sys.stderr)
        sys.exit(1)

    save_specs(records, out_path)
    models_found = sorted({r["model"] for r in records})
    print(f"\nModels: {models_found}")
    print(f"\nSample (first 3 records):")
    for r in records[:3]:
        print(f"  [{r['model']}] {r['key']}: {r['value'][:80]}")


if __name__ == "__main__":
    main()
