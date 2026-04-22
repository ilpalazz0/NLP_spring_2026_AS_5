from __future__ import annotations

import argparse
from datetime import datetime

from rag_system.config import settings
from rag_system.data.manager import ingest_documents
from rag_system.embeddings.embedder import SentenceEmbedder
from rag_system.preprocessing.pipeline import build_chunks, clean_documents
from rag_system.retrieval.vector_store import ChromaVectorStore
from rag_system.summaries.dataset_summary import build_dataset_summary
from rag_system.utils.io import write_json, write_jsonl_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base offline.")
    parser.add_argument("--config", required=True, help="Path to dataset_config.json")
    args = parser.parse_args()

    config, documents = ingest_documents(args.config)
    cleaned_documents = clean_documents(config, documents)

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
        raise ValueError("No chunks were created. Reduce min_chunk_tokens or inspect the dataset.")

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
    write_json(settings.summary_path, summary)

    manifest = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "dataset_name": config.dataset_name,
        "embedding_model_name": settings.embedding_model_name,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "document_count": len(cleaned_documents),
        "chunk_count": len(chunks),
        "chroma_dir": str(settings.chroma_dir),
    }
    write_json(settings.manifest_path, manifest)

    print("Knowledge base build complete.")
    print(f"Documents: {len(cleaned_documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Summary: {settings.summary_path}")
    print(f"Vector store: {settings.chroma_dir}")


if __name__ == "__main__":
    main()
