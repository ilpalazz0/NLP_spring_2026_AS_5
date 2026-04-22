import argparse
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup

API_URL = "https://az.wikipedia.org/w/api.php"
WIKI_BASE = "https://az.wikipedia.org/wiki/"
USER_AGENT = "AzerbaijaniLiteratureRAGBot/1.0 (academic project; contact: local)"

DEFAULT_SEED_PAGES = [
    "Azərbaycan ədəbiyyatı",
    "Azərbaycan yazıçılarının siyahısı",
]

DEFAULT_SEED_CATEGORIES = [
    "Kateqoriya:Azərbaycan yazıçıları",
    "Kateqoriya:Azərbaycan şairləri",
    "Kateqoriya:Azərbaycanlı şairlər",
    "Kateqoriya:Azərbaycan Respublikasının xalq yazıçıları",
    "Kateqoriya:Azərbaycan Respublikasının xalq şairləri",
]

SKIP_TITLE_PREFIXES = (
    "Kateqoriya:",
    "Şablon:",
    "Fayl:",
    "Vikipediya:",
    "Kömək:",
    "Portal:",
    "Müzakirə:",
    "İstifadəçi:",
    "Xüsusi:",
)

SKIP_SECTION_TITLES = {
    "istinadlar",
    "xarici keçidlər",
    "həmçinin bax",
    "qeydlər",
    "mənbələr",
    "ədəbiyyat",
    "galereya",
    "qalereya",
}

REMOVE_SELECTORS = [
    "table",
    "style",
    "script",
    "sup.reference",
    "ol.references",
    "div.reflist",
    "span.mw-editsection",
    "div.navbox",
    "div.metadata",
    "div.hatnote",
    "div.thumb",
    "div.toc",
    "ul.gallery",
]

def normalize_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    return text.strip("_") or "output"

def make_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()

def page_url(title: str) -> str:
    return WIKI_BASE + quote(title.replace(" ", "_"), safe="()")

class WikiScraper:
    def __init__(self, sleep_s: float = 0.2):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.sleep_s = sleep_s

    def api_get(self, params: Dict) -> Dict:
        params = dict(params)
        params["format"] = "json"
        params["formatversion"] = "2"
        r = self.session.get(API_URL, params=params, timeout=60)
        r.raise_for_status()
        time.sleep(self.sleep_s)
        return r.json()

    def get_parse(self, title: str) -> Dict:
        data = self.api_get({
            "action": "parse",
            "page": title,
            "prop": "text|categories",
            "redirects": 1,
        })
        return data.get("parse", {})

    def get_category_members(self, category_title: str) -> List[Dict]:
        members = []
        cont = {}
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_title,
                "cmlimit": "max",
                "cmtype": "page|subcat",
            }
            params.update(cont)
            data = self.api_get(params)
            members.extend(data.get("query", {}).get("categorymembers", []))
            if "continue" not in data:
                break
            cont = data["continue"]
        return members

def extract_categories(parse_obj: Dict) -> List[str]:
    cats = []
    for c in parse_obj.get("categories", []):
        if isinstance(c, str):
            cats.append(c)
        elif isinstance(c, dict):
            value = c.get("*") or c.get("title") or c.get("category") or c.get("name")
            if value:
                cats.append(value)
    return cats

def should_skip_title(title: str) -> bool:
    if any(title.startswith(prefix) for prefix in SKIP_TITLE_PREFIXES):
        return True
    lower = title.lower()
    if "(dəqiqləşdirmə)" in lower:
        return True
    return False

def extract_internal_links_from_html(html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup
    links = set()

    for a in content.select("a[href]"):
        href = a.get("href", "")
        if not href.startswith("/wiki/"):
            continue
        title = unquote(href.split("/wiki/", 1)[1])
        title = title.split("#", 1)[0].replace("_", " ")
        if not title or ":" in title:
            continue
        if should_skip_title(title):
            continue
        links.add(title)
    return links

def section_docs_from_html(
    title: str,
    html: str,
    categories: List[str],
    min_section_chars: int = 220,
) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup

    for sel in REMOVE_SELECTORS:
        for tag in content.select(sel):
            tag.decompose()

    docs = []
    current_heading = "Lead"
    buffer = []

    def flush():
        nonlocal buffer, current_heading, docs
        text = normalize_ws(" ".join(buffer))
        if not text:
            buffer = []
            return

        heading_norm = normalize_ws(current_heading).lower()
        if heading_norm in SKIP_SECTION_TITLES:
            buffer = []
            return

        if len(text) < min_section_chars:
            buffer = []
            return

        # Skip pure list-like pages/sections
        semicolon_parts = [p.strip() for p in re.split(r"[;•]\s+", text) if p.strip()]
        short_parts = sum(1 for p in semicolon_parts if len(p.split()) <= 4)
        if len(semicolon_parts) >= 12 and short_parts / max(len(semicolon_parts), 1) > 0.65:
            buffer = []
            return

        docs.append({
            "title": title,
            "section_title": current_heading,
            "text": text,
            "categories": categories,
        })
        buffer = []

    for child in content.children:
        if not getattr(child, "name", None):
            continue

        name = child.name.lower()

        if name in {"h2", "h3", "h4"}:
            flush()
            current_heading = normalize_ws(child.get_text(" ", strip=True)) or "Untitled"
            continue

        if name in {"p", "ul", "ol", "blockquote", "dl"}:
            text = normalize_ws(child.get_text(" ", strip=True))
            if text:
                buffer.append(text)

    flush()
    return docs

def crawl_categories(scraper: WikiScraper, seed_categories: List[str], depth: int = 1) -> Set[str]:
    pages = set()
    seen_cats = set()
    q = deque((cat, 0) for cat in seed_categories)

    while q:
        cat, d = q.popleft()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)

        try:
            members = scraper.get_category_members(cat)
        except Exception as e:
            print(f"[WARN] Failed category {cat}: {e}")
            continue

        for m in members:
            title = m.get("title", "").strip()
            ns = m.get("ns")
            if not title:
                continue

            if ns == 14 and d < depth:  # subcategory
                q.append((title, d + 1))
            elif ns == 0 and not should_skip_title(title):
                pages.add(title)

    return pages

def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f"[WARN] Could not load tokenizer {model_name}: {e}")
        print("[WARN] Falling back to word-based chunking.")
        return None

def chunk_text(text: str, tokenizer, chunk_size: int = 320, overlap: int = 64) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if tokenizer is None:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap
        return chunks

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        chunk = normalize_ws(chunk)
        if chunk:
            chunks.append(chunk)
        if end == len(token_ids):
            break
        start = end - overlap
    return chunks

def build_dataset(
    out_dir: Path,
    seed_pages: List[str],
    seed_categories: List[str],
    category_depth: int,
    max_seed_links: int,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
    sleep_s: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    scraper = WikiScraper(sleep_s=sleep_s)

    print("[INFO] Gathering seed-page links...")
    candidate_pages = set(seed_pages)
    raw_seed_html = {}

    for title in seed_pages:
        try:
            parsed = scraper.get_parse(title)
            html = parsed.get("text", "")
            raw_seed_html[title] = html
            links = list(extract_internal_links_from_html(html))[:max_seed_links]
            candidate_pages.update(links)
            print(f"[OK] Seed page: {title} -> +{len(links)} linked pages")
        except Exception as e:
            print(f"[WARN] Failed seed page {title}: {e}")

    print("[INFO] Gathering category members...")
    category_pages = crawl_categories(scraper, seed_categories, depth=category_depth)
    candidate_pages.update(category_pages)
    print(f"[OK] Category expansion added {len(category_pages)} pages")

    candidate_pages = {t for t in candidate_pages if not should_skip_title(t)}
    print(f"[INFO] Total candidate pages: {len(candidate_pages)}")

    tokenizer = load_tokenizer(tokenizer_name)

    documents = []
    chunks = []
    pages_seen = 0
    pages_kept = 0

    for i, title in enumerate(sorted(candidate_pages), start=1):
        try:
            parsed = scraper.get_parse(title)
            html = parsed.get("text", "")
            categories = extract_categories(parsed)

            if not html:
                continue

            # skip list/disambiguation-ish pages as final docs,
            # but still let them contribute links through seed expansion
            lower_title = title.lower()
            if "siyahısı" in lower_title or any("dəqiqləşdirmə" in c.lower() for c in categories):
                pages_seen += 1
                continue

            section_docs = section_docs_from_html(title, html, categories)

            if not section_docs:
                pages_seen += 1
                continue

            page_type = "history" if title == "Azərbaycan ədəbiyyatı" else "author_or_topic"
            source = page_url(title)

            kept_this_page = 0
            for d in section_docs:
                doc_id = make_hash(d["title"], d["section_title"], d["text"])
                doc = {
                    "doc_id": doc_id,
                    "title": d["title"],
                    "section_title": d["section_title"],
                    "text": d["text"],
                    "language": "az",
                    "source": source,
                    "source_title": title,
                    "page_type": page_type,
                    "categories": d["categories"],
                    "license": "CC BY-SA 4.0 / GFDL",
                    "attribution_required": True,
                }
                documents.append(doc)
                kept_this_page += 1

                doc_chunks = chunk_text(d["text"], tokenizer, chunk_size=chunk_size, overlap=overlap)
                for j, ch in enumerate(doc_chunks):
                    chunk_id = make_hash(doc_id, str(j), ch)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": j,
                        "title": d["title"],
                        "section_title": d["section_title"],
                        "text": ch,
                        "language": "az",
                        "source": source,
                        "page_type": page_type,
                        "categories": d["categories"],
                    })

            pages_seen += 1
            if kept_this_page > 0:
                pages_kept += 1

            if i % 25 == 0:
                print(f"[INFO] Processed {i}/{len(candidate_pages)} pages")

        except Exception as e:
            print(f"[WARN] Failed page {title}: {e}")

    # deduplicate docs/chunks by text hash
    unique_docs = {}
    for d in documents:
        key = make_hash(d["title"], d["section_title"], d["text"])
        unique_docs[key] = d
    documents = list(unique_docs.values())

    unique_chunks = {}
    for c in chunks:
        key = make_hash(c["doc_id"], c["text"])
        unique_chunks[key] = c
    chunks = list(unique_chunks.values())

    docs_path = out_dir / "documents.jsonl"
    chunks_path = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "manifest.json"

    with docs_path.open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    manifest = {
        "wiki": "az.wikipedia.org",
        "seed_pages": seed_pages,
        "seed_categories": seed_categories,
        "candidate_pages": len(candidate_pages),
        "pages_kept": pages_kept,
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "tokenizer": tokenizer_name,
        "license": "CC BY-SA 4.0 / GFDL",
        "note": "Keep attribution and source URLs when reusing or distributing this dataset.",
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Saved to:  {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/wiki_az_literature")
    parser.add_argument("--category_depth", type=int, default=1)
    parser.add_argument("--max_seed_links", type=int, default=150)
    parser.add_argument("--chunk_size", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sleep_s", type=float, default=2)
    args = parser.parse_args()

    build_dataset(
        out_dir=Path(args.out_dir),
        seed_pages=DEFAULT_SEED_PAGES,
        seed_categories=DEFAULT_SEED_CATEGORIES,
        category_depth=args.category_depth,
        max_seed_links=args.max_seed_links,
        tokenizer_name=args.tokenizer,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        sleep_s=args.sleep_s,
    )

if __name__ == "__main__":
    main()