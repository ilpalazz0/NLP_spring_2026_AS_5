from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_system.embeddings.embedder import SentenceEmbedder
from rag_system.retrieval.vector_store import ChromaVectorStore
from rag_system.schemas import RetrievedChunk


class Retriever:
    def __init__(
        self,
        embedder: SentenceEmbedder,
        vector_store: ChromaVectorStore,
        use_reranker: bool = False,
        reranker_model_name: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.reranker = CrossEncoder(reranker_model_name) if use_reranker and reranker_model_name else None

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []

        top_k = max(1, int(top_k))
        candidate_k = top_k * 3 if self.reranker else top_k

        query_embedding = self.embedder.encode([query])[0]
        candidates = self.vector_store.search(query_embedding=query_embedding, top_k=candidate_k)

        if not self.reranker:
            return candidates[:top_k]

        pairs = [(query, item.text) for item in candidates]
        scores = self.reranker.predict(pairs)

        rescored: list[RetrievedChunk] = []
        for item, rerank_score in zip(candidates, scores):
            item.score = float(rerank_score)
            rescored.append(item)

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
