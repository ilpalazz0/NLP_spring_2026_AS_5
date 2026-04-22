from __future__ import annotations

from typing import Sequence

from rag_system.schemas import RetrievedChunk


def build_rag_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        section_line = f"\nBölmə: {chunk.section_title}" if chunk.section_title else ""
        source_title_line = f"\nMənbə başlığı: {chunk.source_title}" if chunk.source_title else ""
        source_line = f"\nURL: {chunk.source}" if chunk.source else ""

        context_blocks.append(
            f"[Mənbə {idx}]"
            f"\nBaşlıq: {chunk.title}"
            f"{section_line}"
            f"{source_title_line}"
            f"{source_line}"
            f"\nSənəd ID: {chunk.doc_id}"
            f"\nParça ID: {chunk.chunk_id}"
            f"\nMətn: {chunk.text}"
        )
    context = "\n\n".join(context_blocks)

    return f"""Sən Azərbaycan dilində sual-cavab köməkçisisən.

QAYDALAR:
1. Yalnız verilən kontekstdən istifadə et.
2. Kənar bilik və ya ehtimal əsasında məlumat uydurma.
3. Cavab kontekstdə yoxdursa `abstained=true` et və cavabda dəqiq bu cümləni yaz:
"Bu məlumat verilən kontekstdə yoxdur."
4. Cavab verdikdə istifadə etdiyin mənbələri citation kimi qaytar.
5. Mütləq JSON qaytar. JSON-dan kənar mətn yazma.

Sual:
{question}

Kontekst:
{context}

Çıxış formatı (JSON):
{{
  "answer": "string",
  "citations": [
    {{
      "source_index": 1,
      "title": "string",
      "section_title": "string",
      "url": "string",
      "chunk_id": "string",
      "doc_id": "string"
    }}
  ],
  "abstained": false,
  "confidence": 0.0
}}
"""


def build_baseline_prompt(question: str) -> str:
    return f"""Sən Azərbaycan dilində sual-cavab köməkçisisən.
Kənar bilikdən istifadə edə bilərsən, amma əmin deyilsənsə açıq qeyd et.
Mütləq JSON qaytar. JSON-dan kənar mətn yazma.

Sual:
{question}

Çıxış formatı (JSON):
{{
  "answer": "string",
  "citations": [],
  "abstained": false,
  "confidence": 0.0
}}
"""
