"""
診斷 GIGABYTE 規格頁面的渲染方式與 HTML 結構。
在 Colab 直接執行：python scripts/diagnose_page.py
"""

import json
import re
import sys
import requests
from bs4 import BeautifulSoup

URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
SPEC_KEYWORDS = ["RTX", "DDR5", "Ryzen", "顯示", "處理器", "GPU", "CPU", "記憶體"]

headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

print("=" * 60)
print("Step 1: Fetch static HTML")
print("=" * 60)
resp = requests.get(URL, headers=headers, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Content-Length: {len(resp.text):,} bytes")
print(f"Content-Type: {resp.headers.get('Content-Type')}")

html = resp.text

print("\n--- Keyword check ---")
is_static = False
for kw in SPEC_KEYWORDS:
    found = kw in html
    if found:
        is_static = True
    print(f"  '{kw}': {'✓ FOUND' if found else '✗ missing'}")

print(f"\n>>> Page is {'STATICALLY rendered ✓' if is_static else 'JS-rendered (static HTML has no spec data) ✗'}")

# ── Parse structure if data is present ──────────────────────────────────────
if is_static:
    print("\n" + "=" * 60)
    print("Step 2: Analyze HTML structure")
    print("=" * 60)
    soup = BeautifulSoup(html, "lxml")

    # Find all elements near keyword "RTX"
    rtx_tags = soup.find_all(string=re.compile("RTX"))
    print(f"\nElements containing 'RTX': {len(rtx_tags)}")
    for tag in rtx_tags[:3]:
        parent = tag.parent
        grandparent = parent.parent if parent else None
        print(f"  tag: <{parent.name} class='{parent.get('class', '')}'>")
        if grandparent:
            print(f"  parent: <{grandparent.name} class='{grandparent.get('class', '')}'>")
        print(f"  text snippet: {str(tag)[:120]}")
        print()

    # Check for <dl> / <table> / <ul> spec structures
    print("--- Structural elements ---")
    print(f"  <dl> count: {len(soup.find_all('dl'))}")
    print(f"  <table> count: {len(soup.find_all('table'))}")
    print(f"  <tr> count: {len(soup.find_all('tr'))}")

    # Look for spec-related class names
    print("\n--- Classes containing 'spec' ---")
    spec_classes = set()
    for tag in soup.find_all(class_=True):
        for cls in tag.get("class", []):
            if "spec" in cls.lower():
                spec_classes.add((tag.name, cls))
    for name, cls in sorted(spec_classes)[:20]:
        print(f"  <{name} class='{cls}'>")

    # Check for JSON in <script> tags (SSR data)
    print("\n--- JSON in <script> tags ---")
    for script in soup.find_all("script"):
        content = script.string or ""
        if any(kw in content for kw in ["RTX", "DDR5", "Ryzen"]):
            print(f"  Found spec data in <script>! Length: {len(content):,} chars")
            # Try to extract JSON
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    print(f"  Valid JSON found. Top-level keys: {list(data.keys())[:10]}")
                except Exception:
                    print("  (not valid JSON, might be inline JS)")
            break
    else:
        print("  No spec data found in <script> tags")

    # Sample raw HTML around first spec keyword
    print("\n--- Raw HTML sample (200 chars around first 'RTX') ---")
    idx = html.find("RTX")
    if idx >= 0:
        snippet = html[max(0, idx-100):idx+200]
        print(snippet)

else:
    print("\n" + "=" * 60)
    print("Step 2: Page requires JS rendering")
    print("=" * 60)
    print("Checking for clues about API endpoints or SSR data...")

    soup = BeautifulSoup(html, "lxml")

    # Look for API endpoint hints
    api_hints = re.findall(r'(https?://[^\s"\']+(?:api|spec|product)[^\s"\']*)', html)
    if api_hints:
        print("\nPossible API endpoints found:")
        for h in api_hints[:10]:
            print(f"  {h}")

    # Look for __NEXT_DATA__, __NUXT__, window.__data__ etc.
    for marker in ["__NEXT_DATA__", "__NUXT__", "__INITIAL_STATE__", "window.__data__"]:
        if marker in html:
            print(f"\n✓ Found {marker} (SSR data blob) — might contain specs!")
            idx = html.find(marker)
            print(f"  Snippet: {html[idx:idx+300]}")

    print("\nFull page <title>:", soup.title.string if soup.title else "N/A")
    print("\nFirst 500 chars of <body>:")
    body = soup.body
    if body:
        print(body.get_text()[:500])
