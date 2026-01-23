# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the API server
uvicorn app.main:app --reload

# Run the SQS worker (for async document processing)
python worker.py

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Install dependencies
pip install -r requirements.txt
```

## Architecture

RAG API built with FastAPI, PostgreSQL/pgvector, and sentence-transformers.

### Data Flow

1. **Document Upload** (`POST /documents`): File uploaded → saved to storage (local or S3) → ingestion via background task or SQS
2. **Ingestion** (`app/rag/ingest.py`): Extract text → chunk (3500 chars, 300 overlap) → embed → store in PostgreSQL
3. **Query** (`POST /query`): Embed question → cosine similarity search → optionally generate answer with LLM → return citations

### API Endpoints

- `POST /documents` - Upload document (multipart form: file, source?, project?)
- `GET /documents/{id}` - Get document metadata and status
- `GET /documents/{id}/chunks` - List document chunks
- `POST /query` - Query documents (body: question, top_k, project?, source?, use_llm)
- `GET /health` - Health check

### Key Components

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims), lazy singleton in `app/rag/embedder.py`
- **Vector Storage**: pgvector with `Vector(384)`, cosine distance search
- **LLM Generation** (`app/rag/llm.py`): OpenAI or Gemini with automatic fallback
- **Storage** (`app/storage/base.py`): Abstraction layer supporting local filesystem or S3
- **Queue** (`app/queue/sqs.py`): Optional SQS for async ingestion, consumed by `worker.py`
- **Database**: Async SQLAlchemy with `AsyncSession` (`app/db/session.py`)

### Database Models

- `Document`: id, title, source, project, tags (JSONB), status (PENDING→INDEXING→INDEXED/FAILED)
- `Chunk`: id, document_id, chunk_index, content, chunk_meta (JSONB), embedding (Vector 384)

### Configuration

See `.env.example`. Key settings:

- `DATABASE_URL`: PostgreSQL connection string (format: `postgresql+psycopg://user:pass@host:port/db`)
- `LLM_PROVIDER`: `openai` or `gemini` (requires corresponding API key)
- `STORAGE_BACKEND`: `local` or `s3`
- `USE_SQS`: Enable SQS queue for document processing (requires `SQS_QUEUE_URL`)