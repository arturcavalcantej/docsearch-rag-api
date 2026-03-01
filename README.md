# DocSearch RAG API

REST API for document ingestion and semantic search powered by RAG (Retrieval-Augmented Generation). Upload documents (PDF, images, text), extract and chunk content, generate vector embeddings, and query with natural language — optionally enhanced by LLM-generated answers with citations.

## Tech Stack

- **Framework:** FastAPI
- **Database:** PostgreSQL 16 + pgvector
- **ORM:** SQLAlchemy 2.0 (async)
- **Embeddings:** sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims)
- **LLM:** OpenAI / Google Gemini (with automatic fallback)
- **OCR:** Tesseract via pytesseract + pdfplumber
- **Queue:** AWS SQS (optional, for async ingestion)
- **Storage:** Local filesystem or AWS S3

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Upload  │────▶│  Extract     │────▶│  Chunk + Embed │────▶│  PostgreSQL  │
│  (POST)  │     │  (PDF/OCR/   │     │  (3500 chars,  │     │  + pgvector  │
│          │     │   text)       │     │   300 overlap) │     │              │
└──────────┘     └──────────────┘     └────────────────┘     └──────┬───────┘
                                                                    │
┌──────────┐     ┌──────────────┐     ┌────────────────┐           │
│  Query   │────▶│  Embed       │────▶│  Cosine        │◀──────────┘
│  (POST)  │     │  question    │     │  similarity    │
│          │     │              │     │  search        │
└──────────┘     └──────────────┘     └───────┬────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │  LLM answer      │
                                    │  (optional)      │
                                    │  + citations     │
                                    └──────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- (Optional) Tesseract OCR for scanned PDFs and images

### 1. Start the database

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# System dependencies for OCR (Ubuntu/WSL)
sudo apt-get install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng poppler-utils
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run migrations

```bash
alembic upgrade head
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### Upload a document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@document.pdf" \
  -F "project=my-project" \
  -F "source=manual"
```

### Check document status

```bash
curl http://localhost:8000/documents/{document_id}
```

Status flow: `PENDING` → `INDEXING` → `INDEXED` | `FAILED`

### Query documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "top_k": 5,
    "project": "my-project",
    "use_llm": true
  }'
```

### List document chunks

```bash
curl http://localhost:8000/documents/{document_id}/chunks
```

### Health check

```bash
curl http://localhost:8000/health
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg://rag:rag@localhost:5432/rag` |
| `LLM_PROVIDER` | `openai` or `gemini` | `gemini` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o-mini` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `GEMINI_MODEL` | Gemini model | `gemini-2.0-flash` |
| `STORAGE_BACKEND` | `local` or `s3` | `local` |
| `USE_SQS` | Enable SQS for async processing | `false` |
| `SQS_QUEUE_URL` | AWS SQS queue URL | — |
| `OCR_ENABLED` | Enable OCR for scanned documents | `true` |
| `OCR_LANGUAGE` | Tesseract language codes | `por+eng` |

## SQS Worker (Optional)

For async document processing via AWS SQS:

```bash
python worker.py
```

Requires `USE_SQS=true` and `SQS_QUEUE_URL` configured. When SQS is disabled, documents are processed via FastAPI background tasks.

## Project Structure

```
app/
├── api/routes/          # FastAPI route handlers
├── core/config.py       # Pydantic settings
├── db/                  # Database session and CRUD
├── models/              # SQLAlchemy models (Document, Chunk)
├── rag/
│   ├── extractors/      # PDF, image, and text extractors
│   ├── chunking.py      # Text chunking (3500 chars, 300 overlap)
│   ├── embedder.py      # Sentence-transformers embeddings
│   ├── ingest.py        # Document ingestion pipeline
│   ├── llm.py           # OpenAI/Gemini generation
│   └── retrieve.py      # Vector similarity search
├── schemas/             # Pydantic request/response models
├── storage/             # Local and S3 storage backends
└── queue/               # SQS integration
```
