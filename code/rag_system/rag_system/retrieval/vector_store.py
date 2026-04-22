from __future__ import annotations

import json
import re
import string
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from rag_system.config import settings
from rag_system.schemas import ChunkRecord, RetrievedChunk

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class ChromaVectorStore:
    def __init__(self, persist_directory: str | None = None, collection_name: str | None = None) -> None:
        resolved_dir = persist_directory or str(settings.chroma_dir)
        resolved_collection = collection_name or settings.collection_name
        self.client = chromadb.PersistentClient(path=resolved_dir)
        self.collection_name = resolved_collection
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._lexical_cache: tuple[list[str], list[str], list[dict]] | None = None

    def _safe_batch_size(self) -> int:
        default_batch = 5000

        getter = getattr(self.client, "get_max_batch_size", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, int) and value > 0:
                    return min(value, default_batch)
            except Exception:
                pass

        value = getattr(self.client, "max_batch_size", None)
        if isinstance(value, int) and value > 0:
            return min(value, default_batch)

        return default_batch

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def rebuild(self, chunks: Sequence[ChunkRecord], embeddings) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._lexical_cache = None
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "section_title": chunk.section_title or "",
                "source": chunk.source or "",
                "source_title": chunk.source_title or "",
                "language": chunk.language or "",
                "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
            }
            metadatas.append(metadata)

        if hasattr(embeddings, "tolist"):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [list(e) for e in embeddings]

        batch_size = self._safe_batch_size()
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
            )

    def search(self, query_embedding, top_k: int = 4) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        query_embeddings = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        result = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            score = float(1.0 - distance) if distance is not None else 0.0
            metadata_json = meta.get("metadata_json")
            extra_meta = {}
            if isinstance(metadata_json, str):
                try:
                    extra_meta = json.loads(metadata_json)
                except Exception:
                    extra_meta = {}

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta.get("doc_id", "")),
                    title=str(meta.get("title", "")),
                    text=text,
                    score=score,
                    token_count=int(meta.get("token_count", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    section_title=str(meta.get("section_title", "") or "") or None,
                    source=str(meta.get("source", "") or "") or None,
                    source_title=str(meta.get("source_title", "") or "") or None,
                    language=str(meta.get("language", "") or "") or None,
                    metadata=extra_meta,
                )
            )
        return retrieved

    def search_lexical(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        query_norm = " ".join((query or "").lower().split())
        query_tokens = {
            tok.lower()
            for tok in TOKEN_RE.findall(query_norm.translate(str.maketrans("", "", string.punctuation)))
            if len(tok) > 1
        }
        if not query_tokens and query_norm:
            query_tokens = set(TOKEN_RE.findall(query_norm))

        if self._lexical_cache is None:
            result = self.collection.get(include=["documents", "metadatas"])
            self._lexical_cache = (
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        ids, docs, metas = self._lexical_cache

        scored: list[RetrievedChunk] = []
        for chunk_id, text, meta in zip(ids, docs, metas):
            title = str(meta.get("title", ""))
            title_norm = " ".join(title.lower().split())
            text_norm = " ".join((text or "").lower().split())

            title_tokens = {
                tok.lower()
                for tok in TOKEN_RE.findall(title_norm.translate(str.maketrans("", "", string.punctuation)))
                if len(tok) > 1
            }
            text_tokens = {
                tok.lower()
                for tok in TOKEN_RE.findall(text_norm.translate(str.maketrans("", "", string.punctuation)))
                if len(tok) > 1
            }

            title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
            text_overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            phrase_bonus = 1.0 if query_norm and query_norm in title_norm else 0.0
            text_phrase_bonus = 1.0 if query_norm and query_norm in text_norm else 0.0

            score = 0.60 * title_overlap + 0.30 * text_overlap + 0.07 * phrase_bonus + 0.03 * text_phrase_bonus
            if score <= 0:
                continue

            metadata_json = meta.get("metadata_json")
            extra_meta = {}
            if isinstance(metadata_json, str):
                try:
                    extra_meta = json.loads(metadata_json)
                except Exception:
                    extra_meta = {}

            scored.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta.get("doc_id", "")),
                    title=title,
                    text=text,
                    score=float(score),
                    token_count=int(meta.get("token_count", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    section_title=str(meta.get("section_title", "") or "") or None,
                    source=str(meta.get("source", "") or "") or None,
                    source_title=str(meta.get("source_title", "") or "") or None,
                    language=str(meta.get("language", "") or "") or None,
                    metadata=extra_meta,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(1, int(top_k))]
