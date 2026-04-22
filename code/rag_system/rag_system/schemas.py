from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    dataset_name: str = Field(default="azerbaijani_rag_dataset")
    source_type: str = Field(description="csv | json | jsonl | text_dir | paired_jsonl")
    path: str | None = None

    # Paired / pre-chunked dataset support
    documents_path: str | None = None
    chunks_path: str | None = None
    manifest_path: str | None = None
    keep_existing_chunks: bool = False

    # Raw document fields
    document_id_field: str | None = None
    title_field: str | None = None
    text_field: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)

    # Pre-chunked fields
    chunk_id_field: str = "chunk_id"
    chunk_doc_id_field: str = "doc_id"
    chunk_title_field: str = "title"
    chunk_text_field: str = "text"
    chunk_index_field: str = "chunk_index"
    chunk_token_count_field: str | None = "token_count"
    chunk_section_title_field: str | None = "section_title"
    chunk_language_field: str | None = "language"
    chunk_source_field: str | None = "source"
    chunk_source_title_field: str | None = "source_title"
    chunk_metadata_fields: list[str] = Field(default_factory=list)

    # Cleaning
    lowercase: bool = False
    strip_html: bool = True
    remove_extra_whitespace: bool = True

    @model_validator(mode="after")
    def validate_paths(self) -> "DatasetConfig":
        source_type = self.source_type.strip().lower()

        if source_type == "paired_jsonl":
            if not self.documents_path:
                raise ValueError("documents_path is required when source_type='paired_jsonl'")
            if not self.chunks_path:
                raise ValueError("chunks_path is required when source_type='paired_jsonl'")
            if self.text_field and not self.title_field:
                # harmless, but likely indicates a partially copied config
                self.title_field = self.title_field
            return self

        if not self.path:
            raise ValueError("path is required unless source_type='paired_jsonl'")
        return self


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
    section_title: str | None = None
    source: str | None = None
    source_title: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    token_count: int
    chunk_index: int = 0
    section_title: str | None = None
    source: str | None = None
    source_title: str | None = None
    language: str | None = None
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
    gold_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
