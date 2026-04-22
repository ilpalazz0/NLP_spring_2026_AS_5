from __future__ import annotations

import re
import string

from sentence_transformers import CrossEncoder

from rag_system.embeddings.embedder import SentenceEmbedder
from rag_system.retrieval.vector_store import ChromaVectorStore
from rag_system.schemas import RetrievedChunk

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
AZ_STOPWORDS = {
    "kim",
    "kimdir",
    "nə",
    "nedir",
    "nədir",
    "haqqında",
    "barədə",
    "bu",
    "hansı",
    "necə",
    "harada",
    "ilə",
    "və",
    "da",
    "də",
    "mi",
    "mı",
    "mu",
    "mü",
}
AZ_CHAR_MAP = str.maketrans(
    {
        "ə": "e",
        "Ə": "E",
        "ş": "s",
        "Ş": "S",
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ü": "u",
        "Ü": "U",
    }
)


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
        self.reranker = CrossEncoder(reranker_model_name, trust_remote_code=True) if use_reranker and reranker_model_name else None

    def _normalize(self, text: str) -> str:
        normalized = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(normalized.split())

    def _normalize_ascii(self, text: str) -> str:
        return self._normalize((text or "").translate(AZ_CHAR_MAP))

    def _tokens(self, text: str) -> list[str]:
        return [tok.lower() for tok in TOKEN_RE.findall(text or "")]

    def _keywords(self, text: str) -> list[str]:
        keywords = [tok for tok in self._tokens(text) if tok not in AZ_STOPWORDS and len(tok) > 1]
        return keywords or self._tokens(text)

    def _query_variants(self, query: str) -> list[str]:
        variants: list[str] = []
        raw = (query or "").strip()
        if raw:
            variants.append(raw)
        keyword_query = " ".join(self._keywords(raw))
        if keyword_query and keyword_query not in variants:
            variants.append(keyword_query)
        ascii_query = raw.translate(AZ_CHAR_MAP)
        if ascii_query and ascii_query not in variants:
            variants.append(ascii_query)
        ascii_keyword_query = " ".join(self._keywords(ascii_query))
        if ascii_keyword_query and ascii_keyword_query not in variants:
            variants.append(ascii_keyword_query)
        return variants

    def _lexical_boost(self, query: str, item: RetrievedChunk) -> float:
        query_keywords = self._keywords(query)
        if not query_keywords:
            return 0.0

        title_norm = self._normalize(item.title)
        text_norm = self._normalize(item.text)
        query_norm = self._normalize(query)

        title_ascii = self._normalize_ascii(item.title)
        text_ascii = self._normalize_ascii(item.text)
        query_ascii = self._normalize_ascii(query)

        keyword_set = set(query_keywords)
        title_tokens = set(self._tokens(title_norm))
        text_tokens = set(self._tokens(text_norm))

        title_overlap = len(keyword_set & title_tokens) / max(1, len(keyword_set))
        text_overlap = len(keyword_set & text_tokens) / max(1, len(keyword_set))

        phrase_match = 1.0 if (query_norm and query_norm in title_norm) else 0.0
        phrase_match_ascii = 1.0 if (query_ascii and query_ascii in title_ascii) else 0.0
        text_phrase_match = 1.0 if (query_norm and query_norm in text_norm) else 0.0
        text_phrase_match_ascii = 1.0 if (query_ascii and query_ascii in text_ascii) else 0.0

        return (
            0.50 * title_overlap
            + 0.25 * text_overlap
            + 0.15 * max(phrase_match, phrase_match_ascii)
            + 0.10 * max(text_phrase_match, text_phrase_match_ascii)
        )

    def _with_score(self, item: RetrievedChunk, score: float) -> RetrievedChunk:
        if hasattr(item, "model_copy"):
            return item.model_copy(update={"score": float(score)})
        payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)  # fallback safety
        payload["score"] = float(score)
        return RetrievedChunk(**payload)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []

        top_k = max(1, int(top_k))
        candidate_k = min(max(top_k * (8 if self.reranker else 6), 20), 120)

        # Hybrid candidate generation: semantic retrieval over several query variants.
        merged_candidates: dict[str, RetrievedChunk] = {}
        for variant in self._query_variants(query):
            query_embedding = self.embedder.encode([variant])[0]
            retrieved = self.vector_store.search(query_embedding=query_embedding, top_k=candidate_k)
            for item in retrieved:
                existing = merged_candidates.get(item.chunk_id)
                if existing is None or item.score > existing.score:
                    merged_candidates[item.chunk_id] = item

        # Lexical backstop catches named entities that dense retrieval may miss.
        lexical_candidates = self.vector_store.search_lexical(query=query, top_k=min(candidate_k, 80))
        for item in lexical_candidates:
            existing = merged_candidates.get(item.chunk_id)
            if existing is None:
                merged_candidates[item.chunk_id] = item
            else:
                # Keep higher confidence between dense and lexical channels.
                if float(item.score) > float(existing.score):
                    merged_candidates[item.chunk_id] = item

        candidates = list(merged_candidates.values())
        if not candidates:
            return []

        if not self.reranker:
            rescored = []
            for item in candidates:
                lexical = self._lexical_boost(query, item)
                combined = 0.72 * float(item.score) + 0.28 * lexical
                rescored.append(self._with_score(item, combined))
            rescored.sort(key=lambda x: x.score, reverse=True)
            return rescored[:top_k]

        pairs = [(query, item.text) for item in candidates]
        scores = self.reranker.predict(pairs)

        rescored: list[RetrievedChunk] = []
        for item, rerank_score in zip(candidates, scores):
            lexical = self._lexical_boost(query, item)
            combined = 0.70 * float(rerank_score) + 0.20 * float(item.score) + 0.10 * lexical
            rescored.append(self._with_score(item, combined))

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
