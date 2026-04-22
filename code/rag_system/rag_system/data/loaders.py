from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rag_system.schemas import ChunkRecord, DatasetConfig, DocumentRecord
from rag_system.utils.text import safe_text


def _coerce_path(value: str | None) -> Path:
    if not value:
        raise ValueError("Expected a non-empty file path.")
    return Path(value)


def _metadata_from_row(row: dict, fields: list[str]) -> dict:
    metadata = {}
    for field in fields:
        if field in row:
            metadata[field] = row.get(field)
    return metadata


def _build_document(
    row: dict,
    config: DatasetConfig,
    fallback_id: str,
) -> DocumentRecord:
    doc_id = safe_text(row.get(config.document_id_field)) if config.document_id_field else fallback_id
    title = safe_text(row.get(config.title_field)) if config.title_field else doc_id
    text = safe_text(row.get(config.text_field)) if config.text_field else ""
    metadata = _metadata_from_row(row, config.metadata_fields)
    return DocumentRecord(doc_id=doc_id, title=title or doc_id, text=text, metadata=metadata)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _build_chunk(
    row: dict,
    config: DatasetConfig,
    fallback_id: str,
    documents_by_id: dict[str, DocumentRecord] | None = None,
) -> ChunkRecord:
    chunk_id = safe_text(row.get(config.chunk_id_field)) or fallback_id
    doc_id = safe_text(row.get(config.chunk_doc_id_field))
    parent_doc = (documents_by_id or {}).get(doc_id)

    title = safe_text(row.get(config.chunk_title_field))
    if not title and parent_doc:
        title = parent_doc.title
    if not title:
        title = doc_id or chunk_id

    text = safe_text(row.get(config.chunk_text_field))

    section_title = safe_text(row.get(config.chunk_section_title_field)) if config.chunk_section_title_field else ""
    language = safe_text(row.get(config.chunk_language_field)) if config.chunk_language_field else ""
    source = safe_text(row.get(config.chunk_source_field)) if config.chunk_source_field else ""
    source_title = safe_text(row.get(config.chunk_source_title_field)) if config.chunk_source_title_field else ""

    token_count = 0
    if config.chunk_token_count_field:
        token_count = _to_int(row.get(config.chunk_token_count_field), default=0)

    metadata = {}
    if parent_doc:
        metadata.update(parent_doc.metadata)
    metadata.update(_metadata_from_row(row, config.chunk_metadata_fields))

    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        text=text,
        token_count=token_count,
        chunk_index=_to_int(row.get(config.chunk_index_field), default=0),
        section_title=section_title or None,
        source=source or None,
        source_title=source_title or None,
        language=language or None,
        metadata=metadata,
    )


def load_csv(config: DatasetConfig) -> list[DocumentRecord]:
    df = pd.read_csv(_coerce_path(config.path))
    return [
        _build_document(row, config, fallback_id=f"doc_{idx + 1}")
        for idx, row in enumerate(df.to_dict(orient="records"))
    ]


def load_jsonl_documents_from_path(path: str | Path, config: DatasetConfig) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents.append(_build_document(row, config, fallback_id=f"doc_{idx + 1}"))
    return documents


def load_jsonl(config: DatasetConfig) -> list[DocumentRecord]:
    return load_jsonl_documents_from_path(_coerce_path(config.path), config)


def load_json(config: DatasetConfig) -> list[DocumentRecord]:
    with _coerce_path(config.path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "documents" in payload and isinstance(payload["documents"], list):
            payload = payload["documents"]
        else:
            payload = [payload]

    return [
        _build_document(row, config, fallback_id=f"doc_{idx + 1}")
        for idx, row in enumerate(payload)
    ]


def load_text_dir(config: DatasetConfig) -> list[DocumentRecord]:
    base = _coerce_path(config.path)
    extensions = {".txt", ".md"}
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in extensions)

    documents: list[DocumentRecord] = []
    for idx, path in enumerate(files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(base).as_posix()
        documents.append(
            DocumentRecord(
                doc_id=f"doc_{idx + 1}",
                title=rel,
                text=text,
                metadata={"source_path": rel},
            )
        )
    return documents


def load_documents(config: DatasetConfig) -> list[DocumentRecord]:
    source_type = config.source_type.lower()
    if source_type == "csv":
        return load_csv(config)
    if source_type == "jsonl":
        return load_jsonl(config)
    if source_type == "json":
        return load_json(config)
    if source_type == "text_dir":
        return load_text_dir(config)
    if source_type == "paired_jsonl":
        return load_jsonl_documents_from_path(_coerce_path(config.documents_path), config)
    raise ValueError(f"Unsupported source_type: {config.source_type}")


def load_chunks(config: DatasetConfig, documents: list[DocumentRecord] | None = None) -> list[ChunkRecord]:
    source_type = config.source_type.lower()
    if source_type != "paired_jsonl":
        return []
    if not config.chunks_path:
        return []

    documents_by_id = {doc.doc_id: doc for doc in (documents or [])}
    chunks: list[ChunkRecord] = []
    with _coerce_path(config.chunks_path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks.append(
                _build_chunk(
                    row=row,
                    config=config,
                    fallback_id=f"chunk_{idx + 1}",
                    documents_by_id=documents_by_id,
                )
            )
    return chunks


def load_manifest(config: DatasetConfig) -> dict:
    if not config.manifest_path:
        return {}
    path = _coerce_path(config.manifest_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
