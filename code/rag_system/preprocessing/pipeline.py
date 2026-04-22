from __future__ import annotations

from collections.abc import Iterable

from rag_system.preprocessing.cleaning import TextCleaner
from rag_system.preprocessing.chunking import TextChunker
from rag_system.schemas import ChunkRecord, DatasetConfig, DocumentRecord


def clean_documents(config: DatasetConfig, documents: Iterable[DocumentRecord]) -> list[DocumentRecord]:
    cleaner = TextCleaner(
        lowercase=config.lowercase,
        strip_html_tags=config.strip_html,
        remove_extra_whitespace=config.remove_extra_whitespace,
    )
    cleaned: list[DocumentRecord] = []
    for document in documents:
        cleaned_doc = cleaner.clean_document(document)
        if cleaned_doc.text:
            cleaned.append(cleaned_doc)
    return cleaned


def build_chunks(
    documents: list[DocumentRecord],
    tokenizer_name: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[ChunkRecord]:
    chunker = TextChunker(
        tokenizer_name=tokenizer_name,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )
    all_chunks: list[ChunkRecord] = []
    for document in documents:
        all_chunks.extend(chunker.chunk_document(document))
    return all_chunks
