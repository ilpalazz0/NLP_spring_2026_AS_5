from __future__ import annotations

import json
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from rag_system.config import settings
from rag_system.schemas import ChunkRecord, RetrievedChunk


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

    def rebuild(self, chunks: Sequence[ChunkRecord], embeddings) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
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
                    metadata=extra_meta,
                )
            )
        return retrieved
