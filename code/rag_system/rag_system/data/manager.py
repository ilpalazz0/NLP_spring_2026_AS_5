from __future__ import annotations

from pathlib import Path

from rag_system.data.loaders import load_chunks, load_documents, load_manifest
from rag_system.schemas import DatasetConfig
from rag_system.utils.io import read_json


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    payload = read_json(Path(config_path))
    return DatasetConfig(**payload)


def ingest_dataset(config_path: str | Path):
    config = load_dataset_config(config_path)
    documents = load_documents(config)
    chunks = load_chunks(config, documents=documents)
    manifest = load_manifest(config)
    return config, documents, chunks, manifest


def ingest_documents(config_path: str | Path):
    config, documents, _, _ = ingest_dataset(config_path)
    return config, documents
