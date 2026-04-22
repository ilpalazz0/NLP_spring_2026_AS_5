from __future__ import annotations

import argparse

from rag_system.config import settings
from rag_system.embeddings.embedder import SentenceEmbedder
from rag_system.evaluation.evaluate import evaluate_examples
from rag_system.llm.local_hf import LocalHFGenerator
from rag_system.llm.openai_provider import OpenAIGenerator
from rag_system.rag.pipeline import RAGPipeline
from rag_system.retrieval.retriever import Retriever
from rag_system.retrieval.vector_store import ChromaVectorStore
from rag_system.schemas import EvaluationExample
from rag_system.utils.io import read_jsonl, write_json, write_jsonl_dicts


def build_generator():
    if settings.generation_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when GENERATION_PROVIDER=openai")
        return OpenAIGenerator(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return LocalHFGenerator(
        model_name=settings.local_llm_name,
        max_new_tokens=settings.local_llm_max_new_tokens,
        temperature=settings.local_llm_temperature,
        top_p=settings.local_llm_top_p,
        use_4bit=settings.local_llm_use_4bit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG vs baseline.")
    parser.add_argument("--eval-file", required=True, help="Path to evaluation JSONL")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Retriever top-k")
    args = parser.parse_args()

    examples = [EvaluationExample(**row) for row in read_jsonl(args.eval_file)]

    embedder = SentenceEmbedder(settings.embedding_model_name)
    vector_store = ChromaVectorStore(str(settings.chroma_dir))
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        use_reranker=settings.use_reranker,
        reranker_model_name=settings.reranker_model_name,
    )
    generator = build_generator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator, default_top_k=args.top_k)

    rows, metrics = evaluate_examples(pipeline=pipeline, examples=examples, top_k=args.top_k)
    write_jsonl_dicts(settings.eval_results_path, rows)
    write_json(settings.metrics_path, metrics)

    print("Evaluation complete.")
    print(f"Results: {settings.eval_results_path}")
    print(f"Metrics: {settings.metrics_path}")


if __name__ == "__main__":
    main()
