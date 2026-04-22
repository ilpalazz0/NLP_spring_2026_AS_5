from __future__ import annotations

import json

from rag_system.llm.base import BaseGenerator
from rag_system.rag.prompts import build_baseline_prompt, build_rag_prompt
from rag_system.retrieval.retriever import Retriever
from rag_system.schemas import AskResponse, GeneratedAnswer, GeneratedCitation


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
        actual_top_k = max(1, int(top_k or self.default_top_k))
        retrieved_chunks = self.retriever.search(question, top_k=actual_top_k)
        rag_prompt = build_rag_prompt(question, retrieved_chunks)
        baseline_prompt = build_baseline_prompt(question)
        rag_raw = self.generator.generate(rag_prompt)
        baseline_raw = self.generator.generate(baseline_prompt)

        rag_output = self._parse_generated_output(rag_raw)
        baseline_output = self._parse_generated_output(baseline_raw)
        rag_answer = rag_output.answer
        baseline_answer = baseline_output.answer

        return AskResponse(
            question=question,
            rag_answer=rag_answer,
            baseline_answer=baseline_answer,
            retrieved_chunks=retrieved_chunks,
            top_k=actual_top_k,
            rag_output=rag_output,
            baseline_output=baseline_output,
        )

    def _parse_generated_output(self, raw_text: str) -> GeneratedAnswer:
        text = (raw_text or "").strip()
        if not text:
            return GeneratedAnswer(answer="", citations=[], abstained=True, confidence=0.0)
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                citations = payload.get("citations", [])
                return GeneratedAnswer(
                    answer=str(payload.get("answer", "")).strip(),
                    citations=[GeneratedCitation.model_validate(c) for c in citations if isinstance(c, dict)],
                    abstained=bool(payload.get("abstained", False)),
                    confidence=float(payload.get("confidence", 0.0)),
                )
        except Exception:
            pass
        return GeneratedAnswer(answer=text, citations=[], abstained=False, confidence=0.0)
