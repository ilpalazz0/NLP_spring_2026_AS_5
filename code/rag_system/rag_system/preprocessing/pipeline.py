from __future__ import annotations

from collections.abc import Iterable

from rag_system.preprocessing.chunking import TextChunker
from rag_system.preprocessing.cleaning import TextCleaner
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
        if cleaned_doc.text.strip():
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


def normalize_existing_chunks(
    config: DatasetConfig,
    chunks: list[ChunkRecord],
    documents: list[DocumentRecord],
    tokenizer_name: str,
) -> list[ChunkRecord]:
    cleaner = TextCleaner(
        lowercase=config.lowercase,
        strip_html_tags=config.strip_html,
        remove_extra_whitespace=config.remove_extra_whitespace,
    )
    token_counter = TextChunker(tokenizer_name=tokenizer_name)
    documents_by_id = {doc.doc_id: doc for doc in documents}

    normalized: list[ChunkRecord] = []
    for chunk in chunks:
        text = cleaner.clean_text(chunk.text)
        if not text:
            continue

        title = cleaner.clean_text(chunk.title) or chunk.doc_id
        parent_doc = documents_by_id.get(chunk.doc_id)
        merged_metadata = {}
        if parent_doc:
            merged_metadata.update(parent_doc.metadata)
        merged_metadata.update(chunk.metadata)

        source = chunk.source or str(merged_metadata.get("source", "") or "") or None
        source_title = chunk.source_title or str(merged_metadata.get("source_title", "") or "") or None
        language = chunk.language or str(merged_metadata.get("language", "") or "") or None
        section_title = chunk.section_title

        token_count = chunk.token_count if chunk.token_count > 0 else token_counter.count_tokens(text)

        normalized.append(
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=title,
                text=text,
                token_count=token_count,
                chunk_index=chunk.chunk_index,
                section_title=section_title,
                source=source,
                source_title=source_title,
                language=language,
                metadata=merged_metadata,
            )
        )
    return normalized
