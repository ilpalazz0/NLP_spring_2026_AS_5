from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    dataset_name: str = Field(default="azerbaijani_rag_dataset")
    source_type: str = Field(description="csv | json | jsonl | text_dir")
    path: str
    document_id_field: str | None = None
    title_field: str | None = None
    text_field: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)
    lowercase: bool = False
    strip_html: bool = True
    remove_extra_whitespace: bool = True


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    token_count: int
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class AskResponse(BaseModel):
    question: str
    rag_answer: str
    baseline_answer: str
    retrieved_chunks: list[RetrievedChunk]
    top_k: int


class EvaluationExample(BaseModel):
    question: str
    gold_answer: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
