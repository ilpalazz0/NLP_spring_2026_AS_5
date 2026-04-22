from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    artifacts_dir: Path = ROOT_DIR / "data" / "artifacts"
    evaluation_dir: Path = ROOT_DIR / "data" / "evaluation"

    generation_provider: str = os.getenv("GENERATION_PROVIDER", "local").strip().lower()

    # Local HF generation
    local_llm_name: str = os.getenv("LOCAL_LLM_NAME", "Qwen/Qwen2.5-3B-Instruct")
    local_llm_max_new_tokens: int = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
    local_llm_temperature: float = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))
    local_llm_top_p: float = float(os.getenv("LOCAL_LLM_TOP_P", "0.9"))
    local_llm_use_4bit: bool = _to_bool(os.getenv("LOCAL_LLM_USE_4BIT"), False)

    # Ollama generation
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))

    # OpenAI generation
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

    # Embeddings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # Retrieval / chunking
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "320"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
    min_chunk_tokens: int = int(os.getenv("MIN_CHUNK_TOKENS", "180"))

    # Optional reranker
    use_reranker: bool = _to_bool(os.getenv("USE_RERANKER"), False)
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    # API
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

    # Storage / artifacts
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    chroma_dir: Path = ROOT_DIR / "data" / "artifacts" / "chroma_db"
    documents_path: Path = ROOT_DIR / "data" / "processed" / "documents.jsonl"
    chunks_path: Path = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
    summary_path: Path = ROOT_DIR / "data" / "artifacts" / "dataset_summary.json"
    library_summary_path: Path = ROOT_DIR / "data" / "artifacts" / "library_summary.json"
    manifest_path: Path = ROOT_DIR / "data" / "artifacts" / "build_manifest.json"
    metrics_path: Path = ROOT_DIR / "data" / "evaluation" / "metrics.json"
    eval_results_path: Path = ROOT_DIR / "data" / "evaluation" / "results.jsonl"


settings = Settings()
