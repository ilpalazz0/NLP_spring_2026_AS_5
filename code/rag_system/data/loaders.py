from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from rag_system.schemas import DatasetConfig, DocumentRecord
from rag_system.utils.text import safe_text


def _build_document(
    row: dict,
    config: DatasetConfig,
    fallback_id: str,
) -> DocumentRecord:
    doc_id = safe_text(row.get(config.document_id_field)) if config.document_id_field else fallback_id
    title = safe_text(row.get(config.title_field)) if config.title_field else doc_id
    text = safe_text(row.get(config.text_field)) if config.text_field else ""
    metadata = {field: row.get(field) for field in config.metadata_fields}
    return DocumentRecord(doc_id=doc_id, title=title or doc_id, text=text, metadata=metadata)


def load_csv(config: DatasetConfig) -> list[DocumentRecord]:
    df = pd.read_csv(config.path)
    documents: list[DocumentRecord] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        documents.append(_build_document(row, config, fallback_id=f"doc_{idx+1}"))
    return documents


def load_jsonl(config: DatasetConfig) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    with open(config.path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents.append(_build_document(row, config, fallback_id=f"doc_{idx+1}"))
    return documents


def load_json(config: DatasetConfig) -> list[DocumentRecord]:
    with open(config.path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if "documents" in payload and isinstance(payload["documents"], list):
            payload = payload["documents"]
        else:
            payload = [payload]
    documents: list[DocumentRecord] = []
    for idx, row in enumerate(payload):
        documents.append(_build_document(row, config, fallback_id=f"doc_{idx+1}"))
    return documents


def load_text_dir(config: DatasetConfig) -> list[DocumentRecord]:
    base = Path(config.path)
    extensions = {".txt", ".md"}
    files = sorted([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in extensions])
    documents: list[DocumentRecord] = []
    for idx, path in enumerate(files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(base).as_posix()
        documents.append(
            DocumentRecord(
                doc_id=f"doc_{idx+1}",
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
    raise ValueError(f"Unsupported source_type: {config.source_type}")
