from __future__ import annotations

import argparse
from datetime import datetime, timezone

from rag_system.config import settings
from rag_system.data.manager import ingest_dataset
from rag_system.embeddings.embedder import SentenceEmbedder
from rag_system.preprocessing.pipeline import build_chunks, clean_documents, normalize_existing_chunks
from rag_system.retrieval.vector_store import ChromaVectorStore
from rag_system.summaries.dataset_summary import build_dataset_summary
from rag_system.summaries.library_summary import build_library_summary
from rag_system.utils.io import write_json, write_jsonl_models


def _validate_task_requirements(document_count: int, total_words: int) -> None:
    if document_count < 50 and total_words < 10000:
        raise ValueError(
            "Dataset does not satisfy the assignment minimum. "
            "Need at least 50 documents or at least 10,000 words."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base offline.")
    parser.add_argument("--config", required=True, help="Path to dataset_config.json")
    args = parser.parse_args()

    config, documents, existing_chunks, source_manifest = ingest_dataset(args.config)
    cleaned_documents = clean_documents(config, documents)

    total_words = sum(len(doc.text.split()) for doc in cleaned_documents)
    _validate_task_requirements(len(cleaned_documents), total_words)

    if config.keep_existing_chunks and existing_chunks:
        chunks = normalize_existing_chunks(
            config=config,
            chunks=existing_chunks,
            documents=cleaned_documents,
            tokenizer_name=settings.embedding_model_name,
        )
    else:
        chunks = build_chunks(
            documents=cleaned_documents,
            tokenizer_name=settings.embedding_model_name,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            min_chunk_tokens=settings.min_chunk_tokens,
        )

    if not cleaned_documents:
        raise ValueError("No documents remained after cleaning. Check the dataset config and text field.")
    if not chunks:
        raise ValueError("No chunks were created. Inspect the dataset or reduce min_chunk_tokens.")

    write_jsonl_models(settings.documents_path, cleaned_documents)
    write_jsonl_models(settings.chunks_path, chunks)

    embedder = SentenceEmbedder(settings.embedding_model_name)
    embeddings = embedder.encode([chunk.text for chunk in chunks], batch_size=settings.embedding_batch_size)

    vector_store = ChromaVectorStore(str(settings.chroma_dir))
    vector_store.rebuild(chunks=chunks, embeddings=embeddings)

    summary = build_dataset_summary(
        dataset_name=config.dataset_name,
        documents=cleaned_documents,
        chunks=chunks,
    )
    library_summary = build_library_summary(cleaned_documents)

    write_json(settings.summary_path, summary)
    write_json(settings.library_summary_path, library_summary)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config.dataset_name,
        "source_type": config.source_type,
        "used_existing_chunks": bool(config.keep_existing_chunks and existing_chunks),
        "embedding_model_name": settings.embedding_model_name,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "min_chunk_tokens": settings.min_chunk_tokens,
        "document_count": len(cleaned_documents),
        "chunk_count": len(chunks),
        "total_words": total_words,
        "chroma_dir": str(settings.chroma_dir),
        "source_manifest": source_manifest,
    }
    write_json(settings.manifest_path, manifest)

    print("Knowledge base build complete.")
    print(f"Documents: {len(cleaned_documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Summary: {settings.summary_path}")
    print(f"Library summary: {settings.library_summary_path}")
    print(f"Vector store: {settings.chroma_dir}")


if __name__ == "__main__":
    main()
