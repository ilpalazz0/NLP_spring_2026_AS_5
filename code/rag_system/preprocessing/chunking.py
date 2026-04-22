from __future__ import annotations

import re

from transformers import AutoTokenizer

from rag_system.schemas import ChunkRecord, DocumentRecord

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


class TextChunker:
    def __init__(
        self,
        tokenizer_name: str,
        chunk_size_tokens: int = 320,
        chunk_overlap_tokens: int = 60,
        min_chunk_tokens: int = 120,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_units(self, text: str) -> list[str]:
        units = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
        return units or [text.strip()]

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        units = self._split_units(document.text)
        chunks: list[ChunkRecord] = []
        current_units: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            if unit_tokens > self.chunk_size_tokens:
                encoded = self.tokenizer.encode(unit, add_special_tokens=False)
                for start in range(0, len(encoded), self.chunk_size_tokens - self.chunk_overlap_tokens):
                    segment_ids = encoded[start : start + self.chunk_size_tokens]
                    segment_text = self.tokenizer.decode(segment_ids, skip_special_tokens=True).strip()
                    token_count = len(segment_ids)
                    if token_count >= self.min_chunk_tokens or not chunks:
                        chunks.append(
                            ChunkRecord(
                                chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                                doc_id=document.doc_id,
                                title=document.title,
                                text=segment_text,
                                token_count=token_count,
                                chunk_index=chunk_index,
                                metadata=document.metadata,
                            )
                        )
                        chunk_index += 1
                continue

            projected_tokens = current_tokens + unit_tokens
            if projected_tokens <= self.chunk_size_tokens:
                current_units.append(unit)
                current_tokens = projected_tokens
                continue

            if current_units:
                chunk_text = " ".join(current_units).strip()
                token_count = self.count_tokens(chunk_text)
                if token_count >= self.min_chunk_tokens or not chunks:
                    chunks.append(
                        ChunkRecord(
                            chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                            doc_id=document.doc_id,
                            title=document.title,
                            text=chunk_text,
                            token_count=token_count,
                            chunk_index=chunk_index,
                            metadata=document.metadata,
                        )
                    )
                    chunk_index += 1

                overlap_units: list[str] = []
                overlap_tokens = 0
                for prior in reversed(current_units):
                    prior_tokens = self.count_tokens(prior)
                    if overlap_tokens + prior_tokens > self.chunk_overlap_tokens:
                        break
                    overlap_units.insert(0, prior)
                    overlap_tokens += prior_tokens
                current_units = overlap_units + [unit]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current_units = [unit]
                current_tokens = unit_tokens

        if current_units:
            chunk_text = " ".join(current_units).strip()
            token_count = self.count_tokens(chunk_text)
            if token_count > 0:
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                        doc_id=document.doc_id,
                        title=document.title,
                        text=chunk_text,
                        token_count=token_count,
                        chunk_index=chunk_index,
                        metadata=document.metadata,
                    )
                )
        return chunks
