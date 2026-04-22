from __future__ import annotations

from rag_system.llm.base import BaseGenerator
from rag_system.rag.prompts import build_baseline_prompt, build_rag_prompt
from rag_system.retrieval.retriever import Retriever
from rag_system.schemas import AskResponse


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: BaseGenerator,
        default_top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.default_top_k = default_top_k

    def answer_with_retrieval(self, question: str, top_k: int | None = None) -> AskResponse:
        actual_top_k = top_k or self.default_top_k
        retrieved_chunks = self.retriever.search(question, top_k=actual_top_k)
        rag_prompt = build_rag_prompt(question, retrieved_chunks)
        baseline_prompt = build_baseline_prompt(question)
        rag_answer = self.generator.generate(rag_prompt)
        baseline_answer = self.generator.generate(baseline_prompt)
        return AskResponse(
            question=question,
            rag_answer=rag_answer,
            baseline_answer=baseline_answer,
            retrieved_chunks=retrieved_chunks,
            top_k=actual_top_k,
        )
