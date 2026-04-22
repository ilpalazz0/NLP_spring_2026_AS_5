from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_system.config import settings
from rag_system.embeddings.embedder import SentenceEmbedder
from rag_system.llm.local_hf import LocalHFGenerator
from rag_system.llm.ollama_provider import OllamaGenerator
from rag_system.llm.openai_provider import OpenAIGenerator
from rag_system.rag.pipeline import RAGPipeline
from rag_system.retrieval.retriever import Retriever
from rag_system.retrieval.vector_store import ChromaVectorStore
from rag_system.schemas import ChunkRecord, DocumentRecord
from rag_system.summaries.dataset_summary import build_dataset_summary
from rag_system.summaries.library_summary import build_library_summary
from rag_system.utils.device import describe_device
from rag_system.utils.io import read_json, read_jsonl


@dataclass
class AppState:
    pipeline: RAGPipeline
    generator: Any
    summary: dict
    library: dict
    metrics: dict
    manifest: dict
    device_info: dict
    collection_count: int
    generator_info: dict


def build_generator():
    provider = getattr(settings, "generation_provider", "ollama").strip().lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when GENERATION_PROVIDER=openai")
        return OpenAIGenerator(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ), {"provider": "openai", "model": settings.openai_model}

    if provider in {"ollama", "qwen"}:
        generator = OllamaGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_predict=settings.ollama_num_predict,
            timeout=settings.ollama_timeout,
        )
        return generator, {
            "provider": "ollama",
            "model": settings.ollama_model,
            "base_url": settings.ollama_base_url,
        }

    if provider not in {"local", "local_hf"}:
        raise ValueError("Unsupported GENERATION_PROVIDER. Use one of: local, local_hf, ollama, qwen, openai")

    generator = LocalHFGenerator(
        model_name=settings.local_llm_name,
        max_new_tokens=settings.local_llm_max_new_tokens,
        temperature=settings.local_llm_temperature,
        top_p=settings.local_llm_top_p,
        use_4bit=settings.local_llm_use_4bit,
    )
    return generator, {"provider": "local_hf", "model": settings.local_llm_name}


def load_state() -> AppState:
    missing = []
    for required in [settings.summary_path, settings.manifest_path, settings.documents_path, settings.chunks_path]:
        if not required.exists():
            missing.append(str(required))

    if missing:
        raise FileNotFoundError(
            "Required build artifacts are missing. Run the offline build first. Missing: " + ", ".join(missing)
        )

    summary = read_json(settings.summary_path)
    manifest = read_json(settings.manifest_path)
    metrics = read_json(settings.metrics_path) if settings.metrics_path.exists() else {}

    document_rows = read_jsonl(settings.documents_path)
    chunk_rows = read_jsonl(settings.chunks_path)
    documents = [DocumentRecord.model_validate(row) for row in document_rows]
    chunks = [ChunkRecord.model_validate(row) for row in chunk_rows]

    if not documents:
        raise ValueError("No processed documents were found. Build the knowledge base first.")
    if not chunks:
        raise ValueError("No processed chunks were found. Build the knowledge base first.")

    if not isinstance(summary, dict) or "document_count" not in summary:
        summary = build_dataset_summary(manifest.get("dataset_name", "dataset"), documents, chunks)

    if settings.library_summary_path.exists():
        library = read_json(settings.library_summary_path)
    else:
        library = build_library_summary(documents)

    embedder = SentenceEmbedder(settings.embedding_model_name)
    vector_store = ChromaVectorStore(str(settings.chroma_dir))
    collection_count = vector_store.count()
    if collection_count == 0:
        raise ValueError("The vector store is empty. Run build_knowledge_base.py first.")

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        use_reranker=settings.use_reranker,
        reranker_model_name=settings.reranker_model_name,
    )

    generator, generator_info = build_generator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator, default_top_k=settings.top_k)

    return AppState(
        pipeline=pipeline,
        generator=generator,
        summary=summary,
        library=library,
        metrics=metrics,
        manifest=manifest,
        device_info=describe_device(),
        collection_count=collection_count,
        generator_info=generator_info,
    )
