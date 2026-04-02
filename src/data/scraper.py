"""
Scrape and parse GIGABYTE AORUS MASTER 16 AM6H spec page.
Produces data/specs.json — a list of {model, category, key, value} records.
"""

import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SPEC_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"

# Map sub-model suffix → full name
SKU_MAP = {
    "BZH": "AORUS MASTER 16 AM6H-BZH",
    "BYH": "AORUS MASTER 16 AM6H-BYH",
    "BXH": "AORUS MASTER 16 AM6H-BXH",
}

# Semantic group mapping (Traditional Chinese category names on the page)
CATEGORY_GROUPS = {
    "處理器": "CPU",
    "晶片組": "CPU",
    "作業系統": "OS",
    "記憶體": "Memory",
    "儲存裝置": "Storage",
    "顯示器": "Display",
    "顯示晶片": "GPU",
    "攝影機": "Camera",
    "網路": "Connectivity",
    "音效": "Audio",
    "鍵盤": "Input",
    "觸控板": "Input",
    "電源": "Power",
    "電池": "Power",
    "規格": "Dimensions",
    "尺寸": "Dimensions",
    "重量": "Dimensions",
    "介面": "Ports",
    "連接埠": "Ports",
    "安全性": "Security",
    "隨附配件": "Accessories",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_specs(html: str) -> list[dict]:
    """
    Parse the GIGABYTE spec page HTML.

    The page renders specs inside elements like:
      <div class="spec-detail-container" data-sku="BZH">
        ...
        <ul class="spec-list">
          <li class="spec-item">
            <span class="spec-key">...</span>
            <span class="spec-value">...</span>
          </li>
        </ul>
    Falls back to a generic table/list traversal when selectors differ.
    """
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []

    # --- Strategy 1: structured spec containers per SKU ---
    for sku_code, sku_name in SKU_MAP.items():
        container = soup.find(
            attrs={"data-sku": re.compile(sku_code, re.I)}
        ) or soup.find(
            attrs={"data-model": re.compile(sku_code, re.I)}
        )

        if container:
            _parse_container(container, sku_name, records)
            continue

        # --- Strategy 2: fallback — look for headings with SKU in text ---
        heading = soup.find(
            lambda tag: tag.name in ("h2", "h3", "h4", "div", "span")
            and sku_code in tag.get_text()
        )
        if heading:
            section = heading.find_parent(
                lambda tag: tag.name in ("section", "div", "article")
            )
            if section:
                _parse_container(section, sku_name, records)

    # --- Strategy 3: global fallback — all spec tables / dl / ul on page ---
    if not records:
        records = _parse_global_fallback(soup)

    return records


def _parse_container(container, sku_name: str, records: list[dict]):
    """Extract key-value pairs from a single SKU container."""
    current_category = "General"

    # Try <dl> first (definition list pattern)
    dls = container.find_all("dl")
    if dls:
        for dl in dls:
            dt_tags = dl.find_all("dt")
            dd_tags = dl.find_all("dd")
            for dt, dd in zip(dt_tags, dd_tags):
                key = _normalize_whitespace(dt.get_text())
                value = _normalize_whitespace(dd.get_text())
                if key and value:
                    records.append(
                        {
                            "model": sku_name,
                            "category": _map_category(key),
                            "key": key,
                            "value": value,
                        }
                    )
        return

    # Try <tr> (table rows)
    rows = container.find_all("tr")
    if rows:
        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if th and td:
                key = _normalize_whitespace(th.get_text())
                value = _normalize_whitespace(td.get_text())
                if key and value:
                    records.append(
                        {
                            "model": sku_name,
                            "category": _map_category(key),
                            "key": key,
                            "value": value,
                        }
                    )
        return

    # Try alternating <li> pattern (key/value in sibling li elements)
    items = container.find_all("li")
    i = 0
    while i < len(items) - 1:
        key = _normalize_whitespace(items[i].get_text())
        value = _normalize_whitespace(items[i + 1].get_text())
        # Heuristic: key is short (< 30 chars), value can be longer
        if key and value and len(key) < 40 and len(value) > 0:
            records.append(
                {
                    "model": sku_name,
                    "category": _map_category(key),
                    "key": key,
                    "value": value,
                }
            )
            i += 2
        else:
            i += 1


def _parse_global_fallback(soup: BeautifulSoup) -> list[dict]:
    """Last-resort: collect every table row across the page."""
    records = []
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = _normalize_whitespace(cells[0].get_text())
            value = _normalize_whitespace(" ".join(c.get_text() for c in cells[1:]))
            if key and value:
                records.append(
                    {
                        "model": "AORUS MASTER 16 AM6H",
                        "category": _map_category(key),
                        "key": key,
                        "value": value,
                    }
                )
    return records


def _map_category(key: str) -> str:
    for keyword, group in CATEGORY_GROUPS.items():
        if keyword in key:
            return group
    return "General"


def save_specs(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records → {out_path}")


def main():
    out_path = Path("data/specs.json")
    print(f"Fetching {SPEC_URL} ...")
    html = fetch_html(SPEC_URL)
    records = parse_specs(html)

    if not records:
        print("WARNING: No records parsed. Check the page structure.", file=sys.stderr)
        sys.exit(1)

    save_specs(records, out_path)
    # Print sample
    for r in records[:3]:
        print(r)


if __name__ == "__main__":
    main()
