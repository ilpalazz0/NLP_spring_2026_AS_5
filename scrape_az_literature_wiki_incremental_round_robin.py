import argparse
import hashlib
import json
import random
import re
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://az.wikipedia.org/w/api.php"
WIKI_BASE = "https://az.wikipedia.org/wiki/"

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

YEAR_TITLE_RE = re.compile(r"^(?:\d{1,2}\s+)?\d{3,4}$")
LISTY_TITLE_TOKENS = (
    " siyahısı",
    "əlifba",
    "dəqiqləşdirmə",
)

LITERATURE_TITLE_KEYWORDS = (
    "ədəbiyyat",
    "yazıçı",
    "şair",
    "roman",
    "povest",
    "hekayə",
    "şeir",
    "poema",
    "dram",
    "pyes",
    "qəzəl",
    "divan",
    "folklor",
    "epos",
    "dastan",
    "təmsil",
    "publisist",
    "dramaturq",
    "ədəbiyyatşünas",
    "tənqidçi",
    "filoloq",
)

LITERATURE_CATEGORY_KEYWORDS = (
    "ədəbiyyat",
    "yazıçı",
    "şair",
    "roman",
    "povest",
    "hekayə",
    "şeir",
    "poema",
    "dram",
    "pyes",
    "qəzəl",
    "divan",
    "folklor",
    "epos",
    "dastan",
    "publisist",
    "dramaturq",
    "ədəbiyyatşünas",
    "tənqidçi",
    "filoloq",
    "tərcüməçi",
)

LITERATURE_TEXT_KEYWORDS = (
    "azərbaycan ədəbiyyatı",
    "ədəbiyyatının",
    "ədəbiyyatının nümayəndəsi",
    "yazıçı",
    "şair",
    "dramaturq",
    "ədəbiyyatşünas",
    "publisist",
    "nasir",
    "tənqidçi",
    "roman",
    "povest",
    "hekayə",
    "şeir",
    "poema",
    "pyes",
    "qəzəl",
    "divan",
    "dastan",
    "əsər",
    "yaradıcılığı",
)

NON_LITERATURE_KEYWORDS = (
    "təhsil",
    "idman",
    "döyüş sənətləri",
    "oyunları",
    "xalq oyunları",
    "bayram",
    "xüsusi günlər",
    "təqvim inancları",
    "mərasimlər",
    "novruz",
    "toy adətləri",
    "fişəng",
    "səma fənərləri",
    "kulinariya",
    "çay mədəniyyəti",
    "şərabçılıq",
    "köçəri həyatı",
    "ziyarətgah",
    "xalq təbabəti",
    "memarlıq",
    "coğrafiya",
    "iqtisadiyyat",
    "siyasət",
    "din",
)


def normalize_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def page_url(title: str) -> str:
    return WIKI_BASE + quote(title.replace(" ", "_"), safe="()")


def should_skip_title(title: str) -> bool:
    if not title:
        return True
    if any(title.startswith(prefix) for prefix in SKIP_TITLE_PREFIXES):
        return True

    lower = title.lower().strip()
    if any(token in lower for token in LISTY_TITLE_TOKENS):
        return True
    if YEAR_TITLE_RE.match(lower):
        return True
    if re.fullmatch(r"[0-9\W_]+", lower):
        return True
    return False


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def is_probable_person_name(title: str) -> bool:
    tokens = [token for token in title.split() if token]
    if not 2 <= len(tokens) <= 5:
        return False

    for token in tokens:
        cleaned = token.strip("()[]{}.,")
        if not cleaned or any(char.isdigit() for char in cleaned):
            return False

        alpha_chars = [char for char in cleaned if char.isalpha()]
        if len(alpha_chars) < 2:
            return False
        if not alpha_chars[0].isupper():
            return False

    return True


def is_literature_title_candidate(title: str) -> bool:
    lower = title.lower().strip()
    if contains_keyword(lower, NON_LITERATURE_KEYWORDS):
        return False
    return contains_keyword(lower, LITERATURE_TITLE_KEYWORDS) or is_probable_person_name(title)


def is_literature_page(title: str, categories: List[str], html: str = "") -> bool:
    lower_title = title.lower().strip()
    lower_categories = [category.lower() for category in categories]
    category_blob = " | ".join(lower_categories)
    is_person_page = is_probable_person_name(title)
    category_hits = count_keyword_hits(category_blob, LITERATURE_CATEGORY_KEYWORDS)

    if title == "Azərbaycan ədəbiyyatı":
        return True

    if contains_keyword(lower_title, NON_LITERATURE_KEYWORDS):
        return False
    if contains_keyword(category_blob, NON_LITERATURE_KEYWORDS) and category_hits == 0 and not is_person_page:
        return False

    score = 0
    if contains_keyword(lower_title, LITERATURE_TITLE_KEYWORDS):
        score += 3
    if category_hits:
        score += 3
    if is_person_page:
        score += 1

    text_hits = 0
    strong_role_match = False
    if html:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.select_one("div.mw-parser-output") or soup
        text = normalize_ws(content.get_text(" ", strip=True)).lower()

        negative_hits = count_keyword_hits(text, NON_LITERATURE_KEYWORDS)
        text_hits = count_keyword_hits(text, LITERATURE_TEXT_KEYWORDS)
        strong_role_match = bool(
            re.search(
                r"\b(yazıçı|şair|dramaturq|ədəbiyyatşünas|publisist|nasir|tənqidçi|tərcüməçi|roman|povest|hekayə|şeir|poema|pyes|qəzəl|divan|dastan|folklor)\b",
                text,
            )
        )

        if negative_hits >= 3 and text_hits == 0 and not strong_role_match and not is_person_page:
            return False

        if text_hits >= 2:
            score += 2
        elif text_hits >= 1:
            score += 1

        if strong_role_match:
            score += 2 if is_person_page else 1

    if is_person_page:
        return category_hits > 0 or text_hits > 0 or strong_role_match

    return score >= 3


class JsonCache:
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, payload: Dict[str, Any]) -> Path:
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return self.cache_dir / f"{make_hash(key)}.json"

    def get(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path_for(payload)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, payload: Dict[str, Any], data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path_for(payload)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class WikiScraper:
    def __init__(
        self,
        user_agent: str,
        min_delay_s: float = 1.5,
        jitter_s: float = 0.5,
        max_retries: int = 5,
        maxlag: int = 5,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=0,
            allowed_methods=frozenset(["GET"]),
            backoff_factor=0.5,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.min_delay_s = max(min_delay_s, 0.0)
        self.jitter_s = max(jitter_s, 0.0)
        self.max_retries = max(max_retries, 0)
        self.maxlag = max(maxlag, 1)
        self.last_request_ts = 0.0
        self.cache = JsonCache(cache_dir or Path(".cache/wiki_api"), enabled=use_cache)

    def _throttle(self) -> None:
        now = time.monotonic()
        wait_for = self.min_delay_s + random.uniform(0, self.jitter_s)
        elapsed = now - self.last_request_ts
        if elapsed < wait_for:
            time.sleep(wait_for - elapsed)

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> Optional[float]:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def api_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(params)
        payload["format"] = "json"
        payload["formatversion"] = "2"
        payload["maxlag"] = self.maxlag

        cached = self.cache.get(payload)
        if cached is not None:
            return cached

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(API_URL, params=payload, timeout=(10, 45))
                self.last_request_ts = time.monotonic()

                if response.status_code == 429:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    retry_after = self._retry_after_seconds(response)
                    sleep_for = retry_after if retry_after is not None else min(60, 2 ** (attempt + 1))
                    time.sleep(sleep_for + random.uniform(0, self.jitter_s))
                    continue

                response.raise_for_status()
                data = response.json()
                error = data.get("error") or {}
                if error.get("code") == "maxlag":
                    if attempt >= self.max_retries:
                        raise RuntimeError(error.get("info", "MediaWiki maxlag error"))
                    retry_after = self._retry_after_seconds(response)
                    lag = error.get("lag")
                    sleep_for = retry_after if retry_after is not None else float(lag or (attempt + 2))
                    time.sleep(sleep_for + random.uniform(0, self.jitter_s))
                    continue

                self.cache.set(payload, data)
                return data
            except (requests.RequestException, ValueError, RuntimeError):
                if attempt >= self.max_retries:
                    raise
                sleep_for = min(60, 2 ** (attempt + 1))
                time.sleep(sleep_for + random.uniform(0, self.jitter_s))

        raise RuntimeError("Request failed after retries")

    def get_parse(self, title: str) -> Dict[str, Any]:
        data = self.api_get(
            {
                "action": "parse",
                "page": title,
                "prop": "text|categories",
                "redirects": 1,
                "disabletoc": 1,
                "disableeditsection": 1,
                "disablelimitreport": 1,
            }
        )
        return data.get("parse", {})

    def iter_category_members(
        self,
        category_title: str,
        batch_limit: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        cont: Dict[str, Any] = {}
        batches = 0

        while True:
            params: Dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_title,
                "cmlimit": "max",
                "cmtype": "page|subcat",
            }
            params.update(cont)
            data = self.api_get(params)

            for member in data.get("query", {}).get("categorymembers", []):
                yield member

            batches += 1
            if batch_limit is not None and batches >= batch_limit:
                return
            if "continue" not in data:
                return
            cont = data["continue"]


class IncrementalState:
    """Keeps small append-only index files so reruns can extend the dataset safely."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.state_dir = out_dir / ".state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.doc_ids_path = self.state_dir / "doc_ids.txt"
        self.chunk_ids_path = self.state_dir / "chunk_ids.txt"
        self.processed_pages_path = self.state_dir / "processed_pages.txt"
        self.run_history_path = self.state_dir / "run_history.jsonl"

        self.docs_path = out_dir / "documents.jsonl"
        self.chunks_path = out_dir / "chunks.jsonl"

        self.doc_ids = self._load_or_bootstrap_ids(self.doc_ids_path, self.docs_path, "doc_id")
        self.chunk_ids = self._load_or_bootstrap_ids(self.chunk_ids_path, self.chunks_path, "chunk_id")
        self.processed_pages = self._load_or_bootstrap_processed_pages()

    @staticmethod
    def _load_line_set(path: Path) -> Set[str]:
        if not path.exists():
            return set()
        with path.open("r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def _load_or_bootstrap_ids(self, index_path: Path, jsonl_path: Path, key: str) -> Set[str]:
        ids = self._load_line_set(index_path)
        if ids:
            return ids

        if not jsonl_path.exists():
            return set()

        recovered: Set[str] = set()
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = obj.get(key)
                if value:
                    recovered.add(value)

        if recovered:
            with index_path.open("a", encoding="utf-8") as handle:
                for value in sorted(recovered):
                    handle.write(value + "\n")
        return recovered

    def _load_or_bootstrap_processed_pages(self) -> Set[str]:
        pages = self._load_line_set(self.processed_pages_path)
        if pages:
            return pages

        if not self.docs_path.exists():
            return set()

        recovered: Set[str] = set()
        with self.docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = obj.get("source_title") or obj.get("title")
                if value:
                    recovered.add(value)

        if recovered:
            with self.processed_pages_path.open("a", encoding="utf-8") as handle:
                for value in sorted(recovered):
                    handle.write(value + "\n")
        return recovered

    @contextmanager
    def open_append_handles(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)

        docs_file = self.docs_path.open("a", encoding="utf-8")
        chunks_file = self.chunks_path.open("a", encoding="utf-8")
        doc_ids_file = self.doc_ids_path.open("a", encoding="utf-8")
        chunk_ids_file = self.chunk_ids_path.open("a", encoding="utf-8")
        processed_pages_file = self.processed_pages_path.open("a", encoding="utf-8")

        try:
            yield (
                docs_file,
                chunks_file,
                doc_ids_file,
                chunk_ids_file,
                processed_pages_file,
            )
        finally:
            for handle in (
                docs_file,
                chunks_file,
                doc_ids_file,
                chunk_ids_file,
                processed_pages_file,
            ):
                handle.close()

    def add_doc_id(self, doc_id: str, handle) -> None:
        if doc_id in self.doc_ids:
            return
        self.doc_ids.add(doc_id)
        handle.write(doc_id + "\n")
        handle.flush()

    def add_chunk_id(self, chunk_id: str, handle) -> None:
        if chunk_id in self.chunk_ids:
            return
        self.chunk_ids.add(chunk_id)
        handle.write(chunk_id + "\n")
        handle.flush()

    def mark_page_processed(self, title: str, handle) -> None:
        if title in self.processed_pages:
            return
        self.processed_pages.add(title)
        handle.write(title + "\n")
        handle.flush()

    def append_run_history(self, payload: Dict[str, Any]) -> None:
        with self.run_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_categories(parse_obj: Dict[str, Any]) -> List[str]:
    categories = []
    for category in parse_obj.get("categories", []):
        if isinstance(category, str):
            categories.append(category)
        elif isinstance(category, dict):
            value = category.get("*") or category.get("title") or category.get("category") or category.get("name")
            if value:
                categories.append(value)
    return categories


def extract_internal_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup
    links: List[str] = []
    seen: Set[str] = set()

    for anchor in content.select("a[href]"):
        href = anchor.get("href", "")
        if not href.startswith("/wiki/"):
            continue

        title = unquote(href.split("/wiki/", 1)[1])
        title = title.split("#", 1)[0].replace("_", " ").strip()
        if ":" in title or should_skip_title(title) or title in seen:
            continue

        seen.add(title)
        links.append(title)

    return links


def extract_internal_links_round_robin_by_section(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup

    groups: Dict[str, List[str]] = {}
    group_order: List[str] = []
    current_group = "__lead__"
    seen_global: Set[str] = set()

    def ensure_group(name: str) -> None:
        if name not in groups:
            groups[name] = []
            group_order.append(name)

    ensure_group(current_group)

    for child in content.children:
        tag_name = getattr(child, "name", None)
        if not tag_name:
            continue

        tag_name = tag_name.lower()
        if tag_name == "h2":
            heading = normalize_ws(child.get_text(" ", strip=True))
            if re.fullmatch(r"[A-ZƏÖÜÇŞĞİXJQLa-zəöüçşğiıxjq]{1,3}", heading):
                current_group = heading
                ensure_group(current_group)
            continue

        if tag_name not in {"p", "ul", "ol", "dl", "div"}:
            continue

        for anchor in child.select("a[href]"):
            href = anchor.get("href", "")
            if not href.startswith("/wiki/"):
                continue

            title = unquote(href.split("/wiki/", 1)[1])
            title = title.split("#", 1)[0].replace("_", " ").strip()
            if ":" in title or should_skip_title(title) or title in seen_global:
                continue

            seen_global.add(title)
            groups[current_group].append(title)

    ordered: List[str] = []
    max_len = max((len(items) for items in groups.values()), default=0)
    for index in range(max_len):
        for group in group_order:
            items = groups[group]
            if index < len(items):
                ordered.append(items[index])
    return ordered


def section_docs_from_html(
    title: str,
    html: str,
    categories: List[str],
    min_section_chars: int = 220,
) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup

    for selector in REMOVE_SELECTORS:
        for tag in content.select(selector):
            tag.decompose()

    docs: List[Dict[str, Any]] = []
    current_heading = "Lead"
    buffer: List[str] = []

    def flush() -> None:
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

        parts = [part.strip() for part in re.split(r"[;•]\s+", text) if part.strip()]
        short_parts = sum(1 for part in parts if len(part.split()) <= 4)
        if len(parts) >= 12 and short_parts / max(len(parts), 1) > 0.65:
            buffer = []
            return

        docs.append(
            {
                "title": title,
                "section_title": current_heading,
                "text": text,
                "categories": categories,
            }
        )
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


def load_tokenizer(model_name: str):
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"[WARN] Could not load tokenizer {model_name}: {exc}")
        print("[WARN] Falling back to word-based chunking.")
        return None


def chunk_text(text: str, tokenizer, chunk_size: int = 320, overlap: int = 64) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if tokenizer is None:
        words = text.split()
        chunks: List[str] = []
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
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk = normalize_ws(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if chunk:
            chunks.append(chunk)
        if end == len(token_ids):
            break
        start = end - overlap
    return chunks


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_dataset_stats(documents_path: Path, tokenizer) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "num_documents": 0,
        "total_characters": 0,
        "total_tokens": 0,
        "average_characters": 0.0,
        "average_tokens": 0.0,
        "longest_document": None,
        "shortest_document": None,
    }

    if not documents_path.exists():
        return stats

    longest_doc: Optional[Dict[str, Any]] = None
    shortest_doc: Optional[Dict[str, Any]] = None

    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "")
            char_count = len(text)
            token_count = count_tokens(text, tokenizer)

            doc_summary = {
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "section_title": doc.get("section_title"),
                "source_title": doc.get("source_title"),
                "characters": char_count,
                "tokens": token_count,
            }

            stats["num_documents"] += 1
            stats["total_characters"] += char_count
            stats["total_tokens"] += token_count

            if longest_doc is None or char_count > longest_doc["characters"]:
                longest_doc = doc_summary
            if shortest_doc is None or char_count < shortest_doc["characters"]:
                shortest_doc = doc_summary

    if stats["num_documents"] > 0:
        stats["average_characters"] = round(stats["total_characters"] / stats["num_documents"], 2)
        stats["average_tokens"] = round(stats["total_tokens"] / stats["num_documents"], 2)

    stats["longest_document"] = longest_doc
    stats["shortest_document"] = shortest_doc
    return stats


def append_candidate(
    title: str,
    scheduled: List[str],
    scheduled_set: Set[str],
    processed_pages: Set[str],
    limit: int,
    require_literature_title_hint: bool = False,
) -> bool:
    if len(scheduled) >= limit:
        return False
    if should_skip_title(title) or title in processed_pages or title in scheduled_set:
        return False
    if require_literature_title_hint and not is_literature_title_candidate(title):
        return False
    scheduled.append(title)
    scheduled_set.add(title)
    return True


def discover_from_seed_pages(
    scraper: WikiScraper,
    seed_pages: List[str],
    processed_pages: Set[str],
    scheduled: List[str],
    scheduled_set: Set[str],
    max_new_pages: int,
    max_seed_links_per_page: int,
) -> None:
    print("[INFO] Scanning seed pages...")

    for seed_title in seed_pages:
        append_candidate(seed_title, scheduled, scheduled_set, processed_pages, max_new_pages)
        if len(scheduled) >= max_new_pages:
            return

        try:
            parsed = scraper.get_parse(seed_title)
            html = parsed.get("text", "")
            if not html:
                continue

            new_from_seed = 0
            if normalize_ws(seed_title) == "Azərbaycan yazıçılarının siyahısı":
                candidate_links = extract_internal_links_round_robin_by_section(html)
            else:
                candidate_links = extract_internal_links_from_html(html)

            for linked_title in candidate_links:
                if append_candidate(
                    linked_title,
                    scheduled,
                    scheduled_set,
                    processed_pages,
                    max_new_pages,
                    require_literature_title_hint=True,
                ):
                    new_from_seed += 1
                    if new_from_seed >= max_seed_links_per_page or len(scheduled) >= max_new_pages:
                        break
            print(f"[OK] Seed page: {seed_title} -> +{new_from_seed} new pages scheduled")
            if len(scheduled) >= max_new_pages:
                return
        except Exception as exc:
            print(f"[WARN] Failed seed page {seed_title}: {exc}")


def discover_from_categories(
    scraper: WikiScraper,
    seed_categories: List[str],
    processed_pages: Set[str],
    scheduled: List[str],
    scheduled_set: Set[str],
    max_new_pages: int,
    category_depth: int,
    max_members_per_category: int,
    max_category_pages: int,
    max_category_batches_per_run: int,
) -> None:
    if len(scheduled) >= max_new_pages:
        return

    print("[INFO] Scanning categories for new pages...")
    queue: Deque[Tuple[str, int]] = deque((category, 0) for category in seed_categories)
    seen_categories: Set[str] = set()
    total_category_pages = 0

    while queue and len(scheduled) < max_new_pages and total_category_pages < max_category_pages:
        category_title, depth = queue.popleft()
        if category_title in seen_categories:
            continue
        seen_categories.add(category_title)

        new_from_category = 0
        try:
            for member in scraper.iter_category_members(category_title, batch_limit=max_category_batches_per_run):
                title = (member.get("title") or "").strip()
                namespace = member.get("ns")
                if not title:
                    continue

                if namespace == 14 and depth < category_depth:
                    queue.append((title, depth + 1))
                    continue

                if namespace != 0:
                    continue

                if append_candidate(title, scheduled, scheduled_set, processed_pages, max_new_pages):
                    new_from_category += 1
                    total_category_pages += 1
                    if (
                        new_from_category >= max_members_per_category
                        or total_category_pages >= max_category_pages
                        or len(scheduled) >= max_new_pages
                    ):
                        break

            print(f"[OK] Category: {category_title} -> +{new_from_category} new pages scheduled")
        except Exception as exc:
            print(f"[WARN] Failed category {category_title}: {exc}")


def build_dataset(
    out_dir: Path,
    seed_pages: List[str],
    seed_categories: List[str],
    category_depth: int,
    max_seed_links: int,
    max_members_per_category: int,
    max_category_pages: int,
    max_new_pages: int,
    max_category_batches_per_run: int,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
    min_delay_s: float,
    jitter_s: float,
    max_retries: int,
    maxlag: int,
    min_section_chars: int,
    use_cache: bool,
    contact: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = IncrementalState(out_dir)

    cache_dir = out_dir / ".wiki_api_cache"
    user_agent = f"AzerbaijaniLiteratureRAGBot/2.0 (academic project; contact: {contact})"
    scraper = WikiScraper(
        user_agent=user_agent,
        min_delay_s=min_delay_s,
        jitter_s=jitter_s,
        max_retries=max_retries,
        maxlag=maxlag,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    tokenizer = load_tokenizer(tokenizer_name)

    scheduled: List[str] = []
    scheduled_set: Set[str] = set()

    discover_from_seed_pages(
        scraper=scraper,
        seed_pages=seed_pages,
        processed_pages=state.processed_pages,
        scheduled=scheduled,
        scheduled_set=scheduled_set,
        max_new_pages=max_new_pages,
        max_seed_links_per_page=max_seed_links,
    )

    discover_from_categories(
        scraper=scraper,
        seed_categories=seed_categories,
        processed_pages=state.processed_pages,
        scheduled=scheduled,
        scheduled_set=scheduled_set,
        max_new_pages=max_new_pages,
        category_depth=category_depth,
        max_members_per_category=max_members_per_category,
        max_category_pages=max_category_pages,
        max_category_batches_per_run=max_category_batches_per_run,
    )

    if not scheduled:
        print("[DONE] No new pages scheduled. The dataset is already up to date for the current seeds and limits.")
        dataset_stats = compute_dataset_stats(out_dir / "documents.jsonl", tokenizer)
        manifest = {
            "wiki": "az.wikipedia.org",
            "seed_pages": seed_pages,
            "seed_categories": seed_categories,
            "new_pages_scheduled": 0,
            "new_pages_processed": 0,
            "total_processed_pages": len(state.processed_pages),
            "documents_total": len(state.doc_ids),
            "chunks_total": len(state.chunk_ids),
            "dataset_stats": dataset_stats,
            "cache_enabled": use_cache,
            "note": "Increase limits or add new seeds/categories to keep growing the dataset.",
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        state.append_run_history(manifest)
        print(f"Documents total: {dataset_stats['num_documents']}")
        if dataset_stats["longest_document"]:
            print(
                f"Longest doc: {dataset_stats['longest_document']['title']} | "
                f"section={dataset_stats['longest_document']['section_title']} | "
                f"chars={dataset_stats['longest_document']['characters']} | "
                f"tokens={dataset_stats['longest_document']['tokens']}"
            )
        if dataset_stats["shortest_document"]:
            print(
                f"Shortest doc: {dataset_stats['shortest_document']['title']} | "
                f"section={dataset_stats['shortest_document']['section_title']} | "
                f"chars={dataset_stats['shortest_document']['characters']} | "
                f"tokens={dataset_stats['shortest_document']['tokens']}"
            )
        return

    print(f"[INFO] New pages scheduled for this run: {len(scheduled)}")

    new_pages_processed = 0
    new_documents = 0
    new_chunks = 0

    docs_path = out_dir / "documents.jsonl"
    chunks_path = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "manifest.json"

    with state.open_append_handles() as (
        docs_file,
        chunks_file,
        doc_ids_file,
        chunk_ids_file,
        processed_pages_file,
    ):
        for index, title in enumerate(scheduled, start=1):
            try:
                parsed = scraper.get_parse(title)
                html = parsed.get("text", "")
                categories = extract_categories(parsed)
                if not html:
                    print(f"[WARN] Empty page content for {title}")
                    state.mark_page_processed(title, processed_pages_file)
                    new_pages_processed += 1
                    continue

                if not is_literature_page(title, categories, html):
                    print(f"[SKIP] Irrelevant page rejected: {title}")
                    state.mark_page_processed(title, processed_pages_file)
                    new_pages_processed += 1
                    continue

                lower_title = title.lower()
                if "siyahısı" in lower_title or any("dəqiqləşdirmə" in category.lower() for category in categories):
                    state.mark_page_processed(title, processed_pages_file)
                    new_pages_processed += 1
                    continue

                section_docs = section_docs_from_html(
                    title=title,
                    html=html,
                    categories=categories,
                    min_section_chars=min_section_chars,
                )

                page_type = "history" if title == "Azərbaycan ədəbiyyatı" else "author_or_topic"
                source = page_url(title)

                for section in section_docs:
                    doc_id = make_hash(section["title"], section["section_title"], section["text"])
                    if doc_id in state.doc_ids:
                        continue

                    doc = {
                        "doc_id": doc_id,
                        "title": section["title"],
                        "section_title": section["section_title"],
                        "text": section["text"],
                        "language": "az",
                        "source": source,
                        "source_title": title,
                        "page_type": page_type,
                        "categories": section["categories"],
                        "license": "CC BY-SA 4.0 / GFDL",
                        "attribution_required": True,
                    }
                    docs_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    docs_file.flush()
                    state.add_doc_id(doc_id, doc_ids_file)
                    new_documents += 1

                    for chunk_index, chunk_text_value in enumerate(
                        chunk_text(section["text"], tokenizer, chunk_size=chunk_size, overlap=overlap)
                    ):
                        chunk_id = make_hash(doc_id, chunk_text_value)
                        if chunk_id in state.chunk_ids:
                            continue

                        chunk = {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "title": section["title"],
                            "section_title": section["section_title"],
                            "text": chunk_text_value,
                            "language": "az",
                            "source": source,
                            "page_type": page_type,
                            "categories": section["categories"],
                        }
                        chunks_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        chunks_file.flush()
                        state.add_chunk_id(chunk_id, chunk_ids_file)
                        new_chunks += 1

                state.mark_page_processed(title, processed_pages_file)
                new_pages_processed += 1

                if index % 25 == 0 or index == len(scheduled):
                    print(
                        f"[INFO] Processed {index}/{len(scheduled)} new pages | "
                        f"new_docs={new_documents} new_chunks={new_chunks} "
                        f"total_docs={len(state.doc_ids)} total_chunks={len(state.chunk_ids)}"
                    )
            except Exception as exc:
                print(f"[WARN] Failed page {title}: {exc}")

    dataset_stats = compute_dataset_stats(docs_path, tokenizer)

    manifest = {
        "wiki": "az.wikipedia.org",
        "seed_pages": seed_pages,
        "seed_categories": seed_categories,
        "new_pages_scheduled": len(scheduled),
        "new_pages_processed": new_pages_processed,
        "new_documents": new_documents,
        "new_chunks": new_chunks,
        "total_processed_pages": len(state.processed_pages),
        "documents_total": len(state.doc_ids),
        "chunks_total": len(state.chunk_ids),
        "dataset_stats": dataset_stats,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "tokenizer": tokenizer_name or None,
        "min_delay_s": min_delay_s,
        "jitter_s": jitter_s,
        "max_retries": max_retries,
        "maxlag": maxlag,
        "max_seed_links_per_page": max_seed_links,
        "max_members_per_category": max_members_per_category,
        "max_category_pages_per_run": max_category_pages,
        "max_category_batches_per_run": max_category_batches_per_run,
        "max_new_pages_per_run": max_new_pages,
        "cache_enabled": use_cache,
        "license": "CC BY-SA 4.0 / GFDL",
        "note": "Append-only incremental build. Existing documents/chunks/pages are skipped on later runs.",
        "outputs": {
            "documents": str(docs_path),
            "chunks": str(chunks_path),
            "dataset_stats": str(out_dir / "dataset_stats.json"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    state.append_run_history(manifest)

    print("\n[DONE]")
    print(f"New documents: {new_documents}")
    print(f"New chunks:    {new_chunks}")
    print(f"Total docs:    {len(state.doc_ids)}")
    print(f"Total chunks:  {len(state.chunk_ids)}")
    print(f"Saved to:      {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/wiki_az_literature")
    parser.add_argument(
        "--seed_page",
        action="append",
        default=[],
        help="Add a seed page. Repeat the flag to add multiple pages.",
    )
    parser.add_argument(
        "--seed_category",
        action="append",
        default=[],
        help="Add a seed category. Repeat the flag to add multiple categories.",
    )
    parser.add_argument("--category_depth", type=int, default=1)
    parser.add_argument(
        "--max_seed_links",
        type=int,
        default=40,
        help="Maximum new pages to schedule from each seed page in a single run.",
    )
    parser.add_argument(
        "--max_members_per_category",
        type=int,
        default=80,
        help="Maximum new pages to schedule from each category in a single run.",
    )
    parser.add_argument(
        "--max_category_pages",
        type=int,
        default=200,
        help="Maximum total new pages to schedule from category scanning in a single run.",
    )
    parser.add_argument(
        "--max_new_pages",
        type=int,
        default=250,
        help="Maximum number of brand-new pages to process in this run.",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=None,
        help="Backward-compatible alias for --max_new_pages.",
    )
    parser.add_argument(
        "--max_category_batches_per_run",
        type=int,
        default=3,
        help="Maximum categorymembers API batches to scan per category during one run.",
    )
    parser.add_argument("--min_section_chars", type=int, default=220)
    parser.add_argument("--chunk_size", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--min_delay_s", type=float, default=1.5)
    parser.add_argument("--jitter_s", type=float, default=0.5)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--maxlag", type=int, default=5)
    parser.add_argument("--contact", type=str, default="replace-with-your-email@example.com")
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_new_pages = args.max_pages if args.max_pages is not None else args.max_new_pages
    seed_pages = args.seed_page or DEFAULT_SEED_PAGES
    seed_categories = args.seed_category or DEFAULT_SEED_CATEGORIES

    build_dataset(
        out_dir=Path(args.out_dir),
        seed_pages=seed_pages,
        seed_categories=seed_categories,
        category_depth=args.category_depth,
        max_seed_links=args.max_seed_links,
        max_members_per_category=args.max_members_per_category,
        max_category_pages=args.max_category_pages,
        max_new_pages=max_new_pages,
        max_category_batches_per_run=args.max_category_batches_per_run,
        tokenizer_name=args.tokenizer,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_delay_s=args.min_delay_s,
        jitter_s=args.jitter_s,
        max_retries=args.max_retries,
        maxlag=args.maxlag,
        min_section_chars=args.min_section_chars,
        use_cache=not args.no_cache,
        contact=args.contact,
    )


if __name__ == "__main__":
    main()
