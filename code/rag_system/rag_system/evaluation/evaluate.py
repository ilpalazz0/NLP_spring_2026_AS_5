from __future__ import annotations

from statistics import mean, median, pstdev

from rag_system.evaluation.metrics import exact_match, hit_at_k, reciprocal_rank, token_f1
from rag_system.rag.pipeline import RAGPipeline
from rag_system.schemas import EvaluationExample


def evaluate_examples(pipeline: RAGPipeline, examples: list[EvaluationExample], top_k: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []

    rag_em_scores: list[float] = []
    rag_f1_scores: list[float] = []
    base_em_scores: list[float] = []
    base_f1_scores: list[float] = []

    retrieval_doc_hits: list[float] = []
    retrieval_doc_rr: list[float] = []
    retrieval_chunk_hits: list[float] = []
    retrieval_chunk_rr: list[float] = []
    retrieval_doc_hits_at: dict[int, list[float]] = {3: [], 5: [], 10: []}
    retrieval_chunk_hits_at: dict[int, list[float]] = {3: [], 5: [], 10: []}
    rag_answer_lengths: list[int] = []
    baseline_answer_lengths: list[int] = []
    rag_better_count = 0
    baseline_better_count = 0
    tied_count = 0
    rag_empty_count = 0
    baseline_empty_count = 0

    for example in examples:
        response = pipeline.answer_with_retrieval(example.question, top_k=top_k)
        retrieved_doc_ids = [chunk.doc_id for chunk in response.retrieved_chunks]
        retrieved_chunk_ids = [chunk.chunk_id for chunk in response.retrieved_chunks]

        rag_em = exact_match(response.rag_answer, example.gold_answer)
        rag_f1 = token_f1(response.rag_answer, example.gold_answer)
        base_em = exact_match(response.baseline_answer, example.gold_answer)
        base_f1 = token_f1(response.baseline_answer, example.gold_answer)

        rag_em_scores.append(rag_em)
        rag_f1_scores.append(rag_f1)
        base_em_scores.append(base_em)
        base_f1_scores.append(base_f1)
        rag_answer_lengths.append(len((response.rag_answer or "").split()))
        baseline_answer_lengths.append(len((response.baseline_answer or "").split()))

        if rag_f1 > base_f1:
            rag_better_count += 1
        elif base_f1 > rag_f1:
            baseline_better_count += 1
        else:
            tied_count += 1

        if not (response.rag_answer or "").strip():
            rag_empty_count += 1
        if not (response.baseline_answer or "").strip():
            baseline_empty_count += 1

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
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }

        if example.gold_doc_ids:
            hit = hit_at_k(retrieved_doc_ids, example.gold_doc_ids)
            rr = reciprocal_rank(retrieved_doc_ids, example.gold_doc_ids)
            retrieval_doc_hits.append(hit)
            retrieval_doc_rr.append(rr)
            for k in (3, 5, 10):
                retrieval_doc_hits_at[k].append(hit_at_k(retrieved_doc_ids[:k], example.gold_doc_ids))
            row["retrieval_doc_hit_at_k"] = hit
            row["retrieval_doc_mrr"] = rr
            row["gold_doc_ids"] = example.gold_doc_ids

        if example.gold_chunk_ids:
            hit = hit_at_k(retrieved_chunk_ids, example.gold_chunk_ids)
            rr = reciprocal_rank(retrieved_chunk_ids, example.gold_chunk_ids)
            retrieval_chunk_hits.append(hit)
            retrieval_chunk_rr.append(rr)
            for k in (3, 5, 10):
                retrieval_chunk_hits_at[k].append(hit_at_k(retrieved_chunk_ids[:k], example.gold_chunk_ids))
            row["retrieval_chunk_hit_at_k"] = hit
            row["retrieval_chunk_mrr"] = rr
            row["gold_chunk_ids"] = example.gold_chunk_ids

        if example.metadata:
            row["example_metadata"] = example.metadata

        rows.append(row)

    def _summary_stats(scores: list[float]) -> dict:
        if not scores:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(mean(scores), 4),
            "median": round(median(scores), 4),
            "std": round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    def _length_stats(lengths: list[int]) -> dict:
        if not lengths:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
        return {
            "mean": round(mean(lengths), 2),
            "median": round(median(lengths), 2),
            "min": min(lengths),
            "max": max(lengths),
        }

    rag_em_mean = mean(rag_em_scores) if rag_em_scores else 0.0
    rag_f1_mean = mean(rag_f1_scores) if rag_f1_scores else 0.0
    base_em_mean = mean(base_em_scores) if base_em_scores else 0.0
    base_f1_mean = mean(base_f1_scores) if base_f1_scores else 0.0

    metrics = {
        "num_examples": len(examples),
        "rag": {
            "exact_match": round(rag_em_mean, 4),
            "token_f1": round(rag_f1_mean, 4),
            "exact_match_stats": _summary_stats(rag_em_scores),
            "token_f1_stats": _summary_stats(rag_f1_scores),
            "answer_length_tokens": _length_stats(rag_answer_lengths),
            "empty_answer_rate": round((rag_empty_count / len(examples)), 4) if examples else 0.0,
        },
        "baseline": {
            "exact_match": round(base_em_mean, 4),
            "token_f1": round(base_f1_mean, 4),
            "exact_match_stats": _summary_stats(base_em_scores),
            "token_f1_stats": _summary_stats(base_f1_scores),
            "answer_length_tokens": _length_stats(baseline_answer_lengths),
            "empty_answer_rate": round((baseline_empty_count / len(examples)), 4) if examples else 0.0,
        },
        "improvement": {
            "exact_match_delta": round((rag_em_mean - base_em_mean), 4),
            "token_f1_delta": round((rag_f1_mean - base_f1_mean), 4),
            "relative_token_f1_lift_pct": round(((rag_f1_mean - base_f1_mean) / base_f1_mean) * 100, 2)
            if base_f1_mean > 0
            else 0.0,
        },
        "pairwise_outcomes": {
            "rag_better_count": rag_better_count,
            "baseline_better_count": baseline_better_count,
            "tied_count": tied_count,
            "rag_better_rate": round((rag_better_count / len(examples)), 4) if examples else 0.0,
            "baseline_better_rate": round((baseline_better_count / len(examples)), 4) if examples else 0.0,
            "tied_rate": round((tied_count / len(examples)), 4) if examples else 0.0,
        },
    }

    if retrieval_doc_hits:
        metrics["retrieval_docs"] = {
            "hit_at_k": round(mean(retrieval_doc_hits), 4),
            "mrr": round(mean(retrieval_doc_rr), 4),
            "hit_at_k_stats": _summary_stats(retrieval_doc_hits),
            "mrr_stats": _summary_stats(retrieval_doc_rr),
            "hit_at_3": round(mean(retrieval_doc_hits_at[3]), 4) if retrieval_doc_hits_at[3] else 0.0,
            "hit_at_5": round(mean(retrieval_doc_hits_at[5]), 4) if retrieval_doc_hits_at[5] else 0.0,
            "hit_at_10": round(mean(retrieval_doc_hits_at[10]), 4) if retrieval_doc_hits_at[10] else 0.0,
        }

    if retrieval_chunk_hits:
        metrics["retrieval_chunks"] = {
            "hit_at_k": round(mean(retrieval_chunk_hits), 4),
            "mrr": round(mean(retrieval_chunk_rr), 4),
            "hit_at_k_stats": _summary_stats(retrieval_chunk_hits),
            "mrr_stats": _summary_stats(retrieval_chunk_rr),
            "hit_at_3": round(mean(retrieval_chunk_hits_at[3]), 4) if retrieval_chunk_hits_at[3] else 0.0,
            "hit_at_5": round(mean(retrieval_chunk_hits_at[5]), 4) if retrieval_chunk_hits_at[5] else 0.0,
            "hit_at_10": round(mean(retrieval_chunk_hits_at[10]), 4) if retrieval_chunk_hits_at[10] else 0.0,
        }

    # Backward-compatibility for older frontend keys.
    if "retrieval_docs" in metrics:
        metrics["retrieval"] = {
            "hit_at_k": metrics["retrieval_docs"]["hit_at_k"],
            "mrr": metrics["retrieval_docs"]["mrr"],
        }

    return rows, metrics
