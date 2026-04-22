# Azerbaijani RAG System Architecture

This document describes the high-level architecture, data flow, and technical components of the Azerbaijani Wikipedia RAG (Retrieval-Augmented Generation) system.

## 1. System Overview
The system is designed to provide accurate, context-aware answers to questions about Azerbaijani literature by combining a semantic search engine (retrieval) with a Large Language Model (generation).

## 2. Core Components

### A. Data Ingestion & Preprocessing
The ingestion engine converts raw source data into a structured knowledge base.
- **Tools used**: `Pandas`, `Transformers` (Tokenizer), `Regex`.
- **Cleaning**: `TextCleaner` handles HTML stripping and whitespace normalization while preserving Azerbaijani characters (ə, ş, ç, ğ, ı, ö, ü).
- **Chunking**: `TextChunker` implements a sliding-window strategy.
    - **Default size**: 320 tokens.
    - **Default overlap**: 64 tokens.
    - **Min chunk size**: 180 tokens (to avoid "fragment" chunks).

### B. Vector Storage & Semantic Search
- **Vector Database**: **ChromaDB** (Persistent).
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
    - **Dimension**: 384.
    - **Metric**: Cosine Similarity.
- **Hybrid Retrieval**: The system combines dense vector search with a lexical backstop to ensure high recall for named entities.

### C. Retrieval Pipeline (`Retriever`)
The retrieval process is multi-staged:
1. **Query Normalization**: Queries are converted to lower-case and stripped of punctuation.
2. **Variant Generation**: The system generates query variants (Raw, Keywords, ASCII-only) to overcome spelling variations.
3. **Semantic Search**: Parallel vector searches across variants.
4. **Reranking**: (Optional) Uses `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` to perform deep relevance scoring on the top candidates.

### D. Generation Pipeline (`RAGPipeline`)
Coordinates the transition from "retrieved chunks" to "natural language answer".
- **Prompt Engineering**: Uses specialized Azerbaijani prompts that enforce strict adherence to context and JSON output formatting.
- **LLM Providers**:
    - **Ollama**: Local generation using models like Qwen 2.5.
    - **Local HF**: Direct HuggingFace integration via `transformers` and `bitsandbytes` (4-bit quantization).
    - **OpenAI**: Support for GPT-series models via API.

## 3. Configuration & Environment

The system is controlled via a `.env` file. Key parameters include:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/...` | Model used for vectorization. |
| `GENERATION_PROVIDER` | `ollama` | Choice of LLM backend. |
| `USE_RERANKER` | `true` | Enables/Disables the Cross-Encoder stage. |
| `TOP_K` | `10` | Number of documents passed to the LLM. |
| `CHUNK_SIZE_TOKENS` | `320` | Max tokens per chunk during build. |

## 4. Evaluation Framework
Located in `rag_system.evaluation`, this module calculates:
- **Retrieval Metrics**: MRR (Mean Reciprocal Rank), Hit@k.
- **Generation Metrics**: Token-level F1, Exact Match.
- **Performance Analysis**: Comparison between RAG (with context) and Baseline (without context).

## 5. Technical Tools Summary
- **Database**: ChromaDB
- **NLP Frameworks**: Sentence-Transformers, Transformers (HuggingFace)
- **Runtime**: Python 3.x
- **Inference**: Ollama (preferred) or Local PyTorch
- **Data Handling**: Pydantic (for schemas), Pandas
