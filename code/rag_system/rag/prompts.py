from __future__ import annotations

from typing import Sequence

from rag_system.schemas import RetrievedChunk


def build_rag_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Mənbə {idx}]\nBaşlıq: {chunk.title}\nSənəd ID: {chunk.doc_id}\nMətn: {chunk.text}"
        )
    context = "\n\n".join(context_blocks)

    return f"""Sən Azərbaycan dilində sual-cavab köməkçisisən.
Yalnız verilən kontekstdən istifadə et.
Kontekstdə olmayan məlumatı uydurma.
Əgər cavab kontekstdə yoxdursa, belə yaz:
"Bu məlumat verilən kontekstdə yoxdur."

Sual:
{question}

Kontekst:
{context}

Qısa, aydın və dəqiq cavab ver.
"""


def build_baseline_prompt(question: str) -> str:
    return f"""Sən Azərbaycan dilində sual-cavab köməkçisisən.
Aşağıdakı suala qısa və aydın cavab ver.

Sual:
{question}
"""
