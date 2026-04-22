from __future__ import annotations

import random
from collections import defaultdict

from rag_system.schemas import DocumentRecord


def _extract_author(doc: DocumentRecord) -> str:
    for key in ("author", "Author", "creator"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    source_title = doc.metadata.get("source_title")
    page_type = doc.metadata.get("page_type")
    if source_title and str(page_type) == "author_or_topic":
        return str(source_title)
    return "Unknown"


def _extract_book_title(doc: DocumentRecord) -> str:
    for key in ("book_title", "book", "source_title", "collection"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    return doc.title or "Unknown"


def build_library_summary(documents: list[DocumentRecord]) -> dict:
    by_author: dict[str, list[DocumentRecord]] = defaultdict(list)
    for doc in documents:
        by_author[_extract_author(doc)].append(doc)

    authors_payload: list[dict] = []
    for author in sorted(by_author):
        docs = by_author[author]
        books = sorted({_extract_book_title(doc) for doc in docs})
        sampler = random.Random(author)
        sample_size = min(10, len(docs))
        sampled_docs = sampler.sample(docs, sample_size) if sample_size else []
        sampled_docs = sorted(sampled_docs, key=lambda d: d.title.casefold())

        authors_payload.append(
            {
                "author": author,
                "book_count": len(books),
                "books": books,
                "document_count": len(docs),
                "random_documents": [
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "book_title": _extract_book_title(doc),
                    }
                    for doc in sampled_docs
                ],
            }
        )

    return {
        "author_count": len(authors_payload),
        "authors": authors_payload,
    }
