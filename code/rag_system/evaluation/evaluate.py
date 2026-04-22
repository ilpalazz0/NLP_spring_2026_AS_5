from __future__ import annotations

from statistics import mean

from rag_system.evaluation.metrics import exact_match, hit_at_k, reciprocal_rank, token_f1
from rag_system.rag.pipeline import RAGPipeline
from rag_system.schemas import EvaluationExample


def evaluate_examples(pipeline: RAGPipeline, examples: list[EvaluationExample], top_k: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    rag_em_scores: list[float] = []
    rag_f1_scores: list[float] = []
    base_em_scores: list[float] = []
    base_f1_scores: list[float] = []
    retrieval_hits: list[float] = []
    retrieval_rr: list[float] = []

    for example in examples:
        response = pipeline.answer_with_retrieval(example.question, top_k=top_k)
        retrieved_doc_ids = [chunk.doc_id for chunk in response.retrieved_chunks]

        rag_em = exact_match(response.rag_answer, example.gold_answer)
        rag_f1 = token_f1(response.rag_answer, example.gold_answer)
        base_em = exact_match(response.baseline_answer, example.gold_answer)
        base_f1 = token_f1(response.baseline_answer, example.gold_answer)
        rag_em_scores.append(rag_em)
        rag_f1_scores.append(rag_f1)
        base_em_scores.append(base_em)
        base_f1_scores.append(base_f1)

        row = {
            "question": example.question,
            "gold_answer": example.gold_answer,
            "rag_answer": response.rag_answer,
            "baseline_answer": response.baseline_answer,
            "rag_em": rag_em,
            "rag_f1": rag_f1,
            "baseline_em": base_em,
            "baseline_f1": base_f1,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in response.retrieved_chunks],
        }

        if example.gold_doc_ids:
            hit = hit_at_k(retrieved_doc_ids, example.gold_doc_ids)
            rr = reciprocal_rank(retrieved_doc_ids, example.gold_doc_ids)
            retrieval_hits.append(hit)
            retrieval_rr.append(rr)
            row["retrieval_hit_at_k"] = hit
            row["retrieval_mrr"] = rr
            row["gold_doc_ids"] = example.gold_doc_ids

        rows.append(row)

    metrics = {
        "num_examples": len(examples),
        "rag": {
            "exact_match": round(mean(rag_em_scores), 4) if rag_em_scores else 0.0,
            "token_f1": round(mean(rag_f1_scores), 4) if rag_f1_scores else 0.0,
        },
        "baseline": {
            "exact_match": round(mean(base_em_scores), 4) if base_em_scores else 0.0,
            "token_f1": round(mean(base_f1_scores), 4) if base_f1_scores else 0.0,
        },
        "improvement": {
            "exact_match_delta": round((mean(rag_em_scores) - mean(base_em_scores)), 4) if rag_em_scores else 0.0,
            "token_f1_delta": round((mean(rag_f1_scores) - mean(base_f1_scores)), 4) if rag_f1_scores else 0.0,
        },
    }

    if retrieval_hits:
        metrics["retrieval"] = {
            "hit_at_k": round(mean(retrieval_hits), 4),
            "mrr": round(mean(retrieval_rr), 4),
        }

    return rows, metrics
