from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from rag_system.schemas import ChunkRecord, DocumentRecord


def _extract_author(doc: DocumentRecord) -> str:
    for key in ("author", "Author", "creator"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    return "Unknown"


def _extract_book_title(doc: DocumentRecord) -> str:
    for key in ("book_title", "book", "source_title", "collection"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    return doc.title or "Unknown"


def _document_payload(doc: DocumentRecord) -> dict:
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "author": _extract_author(doc),
        "book_title": _extract_book_title(doc),
        "char_count": len(doc.text),
        "text": doc.text,
    }


def build_dataset_summary(
    dataset_name: str,
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
) -> dict:
    token_counts = [chunk.token_count for chunk in chunks]
    category_counter = Counter()
    source_counter = Counter()
    author_counter = Counter()
    book_titles_by_author: dict[str, set[str]] = defaultdict(set)

    for doc in documents:
        author = _extract_author(doc)
        book_title = _extract_book_title(doc)
        author_counter[author] += 1
        book_titles_by_author[author].add(book_title)
        if "category" in doc.metadata and doc.metadata["category"]:
            category_counter[str(doc.metadata["category"])] += 1
        if "source" in doc.metadata and doc.metadata["source"]:
            source_counter[str(doc.metadata["source"])] += 1

    doc_preview = [_document_payload(doc) for doc in documents[:10]]
    char_counts = [len(doc.text) for doc in documents]
    shortest_doc = min(documents, key=lambda d: len(d.text), default=None)
    longest_doc = max(documents, key=lambda d: len(d.text), default=None)

    return {
        "dataset_name": dataset_name,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "author_count": len(author_counter),
        "book_count": sum(len(books) for books in book_titles_by_author.values()),
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "total_characters": sum(char_counts),
        "avg_document_characters": round(sum(char_counts) / len(char_counts), 2) if char_counts else 0,
        "min_document_characters": min(char_counts) if char_counts else 0,
        "max_document_characters": max(char_counts) if char_counts else 0,
        "top_categories": category_counter.most_common(10),
        "top_sources": source_counter.most_common(10),
        "top_authors": author_counter.most_common(10),
        "sample_documents": doc_preview,
        "shortest_document": _document_payload(shortest_doc) if shortest_doc else None,
        "longest_document": _document_payload(longest_doc) if longest_doc else None,
    }
