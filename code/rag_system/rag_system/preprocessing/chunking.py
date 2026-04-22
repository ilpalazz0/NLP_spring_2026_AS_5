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
        chunk_overlap_tokens: int = 64,
        min_chunk_tokens: int = 180,
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

    def _build_chunk_record(
        self,
        document: DocumentRecord,
        chunk_index: int,
        chunk_text: str,
        token_count: int,
    ) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
            doc_id=document.doc_id,
            title=document.title,
            text=chunk_text,
            token_count=token_count,
            chunk_index=chunk_index,
            section_title=str(document.metadata.get("section_title", "") or "") or None,
            source=str(document.metadata.get("source", "") or "") or None,
            source_title=str(document.metadata.get("source_title", "") or "") or None,
            language=str(document.metadata.get("language", "") or "") or None,
            metadata=document.metadata,
        )

    def _maybe_merge_small_tail(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        if len(chunks) < 2:
            return chunks

        last = chunks[-1]
        if last.token_count >= self.min_chunk_tokens:
            return chunks

        previous = chunks[-2]
        merged_text = f"{previous.text} {last.text}".strip()
        merged_tokens = self.count_tokens(merged_text)
        chunks[-2] = ChunkRecord(
            chunk_id=previous.chunk_id,
            doc_id=previous.doc_id,
            title=previous.title,
            text=merged_text,
            token_count=merged_tokens,
            chunk_index=previous.chunk_index,
            section_title=previous.section_title,
            source=previous.source,
            source_title=previous.source_title,
            language=previous.language,
            metadata=previous.metadata,
        )
        chunks.pop()
        return chunks

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        units = self._split_units(document.text)
        chunks: list[ChunkRecord] = []
        current_units: list[str] = []
        current_tokens = 0
        chunk_index = 0
        stride = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            if unit_tokens > self.chunk_size_tokens:
                encoded = self.tokenizer.encode(unit, add_special_tokens=False)
                for start in range(0, len(encoded), stride):
                    segment_ids = encoded[start : start + self.chunk_size_tokens]
                    segment_text = self.tokenizer.decode(segment_ids, skip_special_tokens=True).strip()
                    token_count = len(segment_ids)
                    if token_count >= self.min_chunk_tokens or not chunks:
                        chunks.append(self._build_chunk_record(document, chunk_index, segment_text, token_count))
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
                    chunks.append(self._build_chunk_record(document, chunk_index, chunk_text, token_count))
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
                chunks.append(self._build_chunk_record(document, chunk_index, chunk_text, token_count))

        return self._maybe_merge_small_tail(chunks)
