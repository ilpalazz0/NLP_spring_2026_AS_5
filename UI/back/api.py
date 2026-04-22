from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from UI.back.bootstrap import ROOT_DIR  # noqa: F401
from UI.back.state import AppState, load_state
from rag_system.config import settings
from rag_system.schemas import AskRequest


app_state: dict[str, AppState] = {}


def _build_allowed_origins() -> list[str]:
    candidates = [
        getattr(settings, "frontend_origin", None),
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    origins: list[str] = []
    for origin in candidates:
        if isinstance(origin, str) and origin.strip() and origin not in origins:
            origins.append(origin)
    return origins


def _safe_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _safe_dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_dump(v) for v in value]
    return value


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    dumped = _safe_dump(value)
    if isinstance(dumped, dict):
        for key in (
            "answer",
            "rag_answer",
            "plain_answer",
            "baseline_answer",
            "text",
            "content",
            "response",
            "output",
            "generated_text",
        ):
            candidate = dumped.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return str(dumped).strip()


def _format_sources(retrieved_chunks: list[dict]) -> list[dict]:
    sources = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        sources.append(
            {
                "rank": idx,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("title"),
                "section_title": chunk.get("section_title"),
                "source": chunk.get("source") or chunk.get("metadata", {}).get("source"),
                "source_title": chunk.get("source_title") or chunk.get("metadata", {}).get("source_title"),
                "language": chunk.get("language") or chunk.get("metadata", {}).get("language"),
                "score": chunk.get("score", 0.0),
                "text": chunk.get("text", ""),
            }
        )
    return sources


def _build_plain_prompt(question: str) -> str:
    return (
        "Sen Azerbaycan dilinde qisa ve faydali cavab veren komekcisen.\n"
        "Asagidaki suala birbasa cavab ver.\n"
        "Hec bir retrieval ve xarici kontekst istifade etme.\n\n"
        f"Sual: {question}\n\n"
        "Cavab:"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["state"] = load_state()
    yield
    app_state.clear()


app = FastAPI(
    title="Azerbaijani Wikipedia RAG Backend",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    state = app_state.get("state")
    return {
        "status": "ok" if state is not None else "loading",
        "ready": state is not None,
        "generation_provider": settings.generation_provider,
        "collection_name": settings.collection_name,
        "collection_count": state.collection_count if state else 0,
    }


@app.get("/manifest")
async def manifest():
    state = app_state["state"]
    return {
        "manifest": state.manifest,
        "device": state.device_info,
        "generator": state.generator_info,
        "generation_provider": settings.generation_provider,
        "collection_count": state.collection_count,
    }


@app.get("/dataset/summary")
async def dataset_summary():
    return app_state["state"].summary


@app.get("/metrics")
async def metrics():
    return app_state["state"].metrics


@app.get("/library")
async def library():
    return app_state["state"].library


@app.post("/ask")
async def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    state = app_state.get("state")
    if state is None:
        raise HTTPException(status_code=503, detail="Application state is not loaded yet.")

    top_k = payload.top_k if payload.top_k is not None else settings.top_k
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")

    rag_payload: dict[str, Any] = {}
    rag_answer = ""
    plain_answer = ""
    rag_error = None
    plain_error = None
    sources: list[dict] = []

    try:
        rag_result = state.pipeline.answer_with_retrieval(question, top_k=top_k)
        dumped = _safe_dump(rag_result)
        if isinstance(dumped, dict):
            rag_payload = dumped
            sources = _format_sources(dumped.get("retrieved_chunks", []))
        else:
            rag_payload = {"rag_result": dumped}
        rag_answer = _extract_text(rag_result) or _extract_text(rag_payload)
        if not rag_answer:
            rag_answer = "RAG generation returned no text."
            rag_error = rag_answer
    except Exception as exc:
        rag_error = str(exc)
        rag_answer = f"RAG generation failed: {exc}"

    try:
        plain_answer = state.generator.generate(_build_plain_prompt(question))
        if isinstance(plain_answer, str):
            plain_answer = plain_answer.strip()
        if not plain_answer:
            plain_answer = "Direct generation returned no text."
            plain_error = plain_answer
    except Exception as exc:
        plain_error = str(exc)
        plain_answer = f"Direct generation failed: {exc}"

    response: dict[str, Any] = {}
    response.update(rag_payload)
    response.update(
        {
            "success": True,
            "question": question,
            "top_k": top_k,
            "rag_answer": rag_answer,
            "plain_answer": plain_answer,
            "baseline_answer": plain_answer,
            "with_rag_answer": rag_answer,
            "without_rag_answer": plain_answer,
            "answer_with_retrieval": rag_answer,
            "answer_without_retrieval": plain_answer,
            "retrieval_answer": rag_answer,
            "direct_answer": plain_answer,
            "rag": rag_answer,
            "no_rag": plain_answer,
            "sources": sources,
            "source_count": len(sources),
            "answers": {
                "with_retrieval": rag_answer,
                "without_retrieval": plain_answer,
            },
            "errors": {
                "with_retrieval": rag_error,
                "without_retrieval": plain_error,
            },
            "rag_error": rag_error,
            "plain_error": plain_error,
            "generator": state.generator_info,
        }
    )
    return response
