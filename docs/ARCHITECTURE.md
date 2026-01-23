# RAG API - Arquitetura e Fluxos

## Visão Geral

Este projeto implementa uma API de **Retrieval-Augmented Generation (RAG)** que permite:
- Upload e processamento de documentos (TXT, PDF, imagens)
- Extração de texto com suporte a OCR
- Indexação semântica usando embeddings vetoriais
- Busca por similaridade semântica
- Geração de respostas usando LLM (opcional)

---

## Arquitetura Geral

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  CLIENTE                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI (REST)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ POST        │  │ GET         │  │ POST        │  │ GET                 │ │
│  │ /documents  │  │ /documents  │  │ /query      │  │ /health             │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐           ┌─────────────────────────────────────────┐
│   Document          │           │            Query Pipeline               │
│   Processing        │           │  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│   Pipeline          │           │  │ Embed   │→ │ Search   │→ │ LLM    │ │
│                     │           │  │ Query   │  │ Vectors  │  │ Answer │ │
└─────────────────────┘           │  └─────────┘  └──────────┘  └────────┘ │
         │                        └─────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PostgreSQL + pgvector                                │
│  ┌───────────────────────────────┐  ┌─────────────────────────────────────┐ │
│  │ documents                     │  │ chunks                              │ │
│  │ - id, title, status          │  │ - id, document_id, content          │ │
│  │ - source, project, tags      │  │ - embedding (Vector 384)            │ │
│  └───────────────────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fluxo 1: Document Processing (Upload e Indexação)

### Diagrama de Fluxo

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload  │ ──▶ │  Storage │ ──▶ │  Extraction  │ ──▶ │   Chunking   │
│  (API)   │     │ (S3/Local)│    │  (OCR/PDF)   │     │  (3500 chars)│
└──────────┘     └──────────┘     └──────────────┘     └──────────────┘
                                                               │
                                                               ▼
┌──────────┐     ┌──────────────┐     ┌──────────────────────────────┐
│  Indexed │ ◀── │   Embedding  │ ◀── │  paraphrase-multilingual-    │
│   (DB)   │     │   (384 dims) │     │  MiniLM-L12-v2               │
└──────────┘     └──────────────┘     └──────────────────────────────┘
```

### Etapas Detalhadas

#### 1. Upload do Documento
```
POST /documents
Content-Type: multipart/form-data

file: <arquivo>
source: "manual" (opcional)
project: "meu-projeto" (opcional)
```

- Documento salvo com status `PENDING`
- Arquivo enviado para storage (local ou S3)
- Ingestão iniciada (background task ou SQS)

#### 2. Extração de Texto (`app/rag/extractor.py`)

O orquestrador detecta o tipo de arquivo e seleciona o extrator apropriado:

```
Arquivo recebido
       │
       ▼
┌──────────────────┐
│ Detectar tipo    │ (magic bytes + extensão)
└──────────────────┘
       │
       ├─── PDF? ──────────────────────────────────────┐
       │    │                                          │
       │    ▼                                          │
       │   ┌────────────────────────┐                  │
       │   │ pdfplumber.extract()   │                  │
       │   └────────────────────────┘                  │
       │    │                                          │
       │    ▼                                          │
       │   texto < 50 chars E OCR_ENABLED?             │
       │    │                                          │
       │    ├── SIM ─▶ ┌─────────────────────────┐     │
       │    │          │ pdf2image → pytesseract │     │
       │    │          └─────────────────────────┘     │
       │    │                                          │
       │    └── NÃO ─▶ retorna texto ─────────────────▶│
       │                                               │
       ├─── Imagem (PNG/JPG/etc)? ─────────────────────┤
       │    │                                          │
       │    ▼                                          │
       │   ┌─────────────────────────┐                 │
       │   │ PIL → pytesseract OCR   │                 │
       │   └─────────────────────────┘                 │
       │                                               │
       ├─── Texto (.txt, .md, .csv, etc)? ─────────────┤
       │    │                                          │
       │    ▼                                          │
       │   ┌─────────────────────────┐                 │
       │   │ decode UTF-8            │                 │
       │   └─────────────────────────┘                 │
       │                                               │
       └─── Desconhecido? ─────────────────────────────┤
            │                                          │
            ▼                                          │
           ┌─────────────────────────┐                 │
           │ fallback: decode UTF-8  │                 │
           └─────────────────────────┘                 │
                                                       │
                        ◀──────────────────────────────┘
                        │
                        ▼
                   TEXTO EXTRAÍDO
```

#### 3. Chunking (`app/rag/chunking.py`)

O texto é dividido em chunks para melhor granularidade na busca:

```
Parâmetros:
- Tamanho máximo: 3500 caracteres
- Overlap: 300 caracteres (contexto entre chunks)

Texto Original (10.000 chars)
│
├── Chunk 0: chars 0-3500
├── Chunk 1: chars 3200-6700  (overlap de 300)
├── Chunk 2: chars 6400-9900  (overlap de 300)
└── Chunk 3: chars 9600-10000
```

#### 4. Embedding (`app/rag/embedder.py`)

Cada chunk é transformado em um vetor de 384 dimensões:

```
┌─────────────────────────────────────────────────────────────────┐
│  Modelo: paraphrase-multilingual-MiniLM-L12-v2                  │
│                                                                  │
│  Características:                                                │
│  - 384 dimensões                                                 │
│  - Multilíngue (50+ idiomas incluindo português)                │
│  - Otimizado para similaridade semântica                        │
│  - Modelo leve (~120MB)                                         │
└─────────────────────────────────────────────────────────────────┘

"Como resetar minha senha?"
            │
            ▼
    ┌───────────────┐
    │ Sentence      │
    │ Transformer   │
    └───────────────┘
            │
            ▼
[0.023, -0.156, 0.089, ..., 0.012]  (384 floats)
```

#### 5. Armazenamento

```sql
-- Tabela chunks com pgvector
INSERT INTO chunks (document_id, chunk_index, content, embedding)
VALUES (1, 0, 'texto do chunk...', '[0.023, -0.156, ...]');
```

---

## Fluxo 2: Information Retrieval (Query)

### Diagrama de Fluxo

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Query   │ ──▶ │   Embed      │ ──▶ │   Cosine     │ ──▶ │  Top K   │
│  (texto) │     │   Question   │     │   Similarity │     │  Chunks  │
└──────────┘     └──────────────┘     └──────────────┘     └──────────┘
                                                                  │
                                                                  ▼
                                                           ┌──────────────┐
                                               (opcional)  │     LLM      │
                                                           │   Response   │
                                                           └──────────────┘
```

### Etapas Detalhadas

#### 1. Receber Query

```
POST /query
{
    "question": "Como faço para resetar minha senha?",
    "top_k": 5,
    "project": "docs-suporte",  // opcional
    "source": null,             // opcional
    "use_llm": true            // opcional
}
```

#### 2. Embed da Pergunta

A pergunta é convertida no mesmo espaço vetorial dos chunks:

```
"Como faço para resetar minha senha?"
            │
            ▼
    ┌───────────────┐
    │ Mesmo modelo  │
    │ (MiniLM)      │
    └───────────────┘
            │
            ▼
[0.045, -0.134, 0.067, ..., 0.023]  (384 floats)
```

#### 3. Busca por Similaridade (pgvector)

```sql
-- Busca usando distância cosseno
SELECT
    c.id,
    c.content,
    c.document_id,
    1 - (c.embedding <=> query_embedding) AS similarity
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE d.project = 'docs-suporte'  -- filtro opcional
ORDER BY c.embedding <=> query_embedding
LIMIT 5;
```

**Distância Cosseno Explicada:**

```
Similaridade = cos(θ) = (A · B) / (||A|| × ||B||)

Valores:
- 1.0 = vetores idênticos (mesma direção)
- 0.0 = vetores ortogonais (sem relação)
- -1.0 = vetores opostos

Exemplo:
Query:    "resetar senha"     → [0.8, 0.2, 0.1]
Chunk A:  "trocar password"   → [0.7, 0.3, 0.1]  similarity: 0.95
Chunk B:  "preço do produto"  → [0.1, 0.1, 0.9]  similarity: 0.23
```

#### 4. Retorno dos Resultados

Sem LLM (`use_llm: false`):
```json
{
    "answer": null,
    "citations": [
        {
            "chunk_id": 42,
            "document_id": 7,
            "content": "Para resetar sua senha, acesse...",
            "score": 0.89,
            "document_title": "FAQ Suporte"
        }
    ]
}
```

#### 5. Geração com LLM (opcional)

Se `use_llm: true`, os chunks são enviados como contexto para o LLM:

```
┌─────────────────────────────────────────────────────────────────┐
│  Prompt para LLM                                                │
│                                                                  │
│  Contexto:                                                       │
│  [Chunk 1]: Para resetar sua senha, acesse Configurações...     │
│  [Chunk 2]: O processo de recuperação de senha leva 24h...      │
│  [Chunk 3]: Senhas devem ter no mínimo 8 caracteres...          │
│                                                                  │
│  Pergunta: Como faço para resetar minha senha?                  │
│                                                                  │
│  Responda baseado apenas no contexto acima.                     │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │ OpenAI/Gemini │
    └───────────────┘
            │
            ▼
"Para resetar sua senha, siga estes passos:
1. Acesse Configurações no menu principal
2. Clique em 'Segurança'
3. Selecione 'Resetar senha'
..."
```

---

## Componentes Principais

### Storage (`app/storage/`)

```
┌─────────────────────────────────────────┐
│           StorageBackend                │
│              (Interface)                │
├─────────────────────────────────────────┤
│  + save_file(content, path)             │
│  + load_file(path) → bytes              │
│  + delete_file(path)                    │
└─────────────────────────────────────────┘
            ▲                 ▲
            │                 │
┌───────────────────┐  ┌───────────────────┐
│  LocalStorage     │  │    S3Storage      │
│  (filesystem)     │  │    (AWS S3)       │
└───────────────────┘  └───────────────────┘
```

### Queue (`app/queue/`) - Opcional

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   API        │ ──▶ │   SQS        │ ──▶ │   Worker     │
│ (enqueue)    │     │   Queue      │     │ (process)    │
└──────────────┘     └──────────────┘     └──────────────┘

Mensagem SQS:
{
    "document_id": 123,
    "file_path": "uploads/doc_123.pdf"
}
```

### LLM (`app/rag/llm.py`)

```
┌─────────────────────────────────────────┐
│           LLM Provider                   │
├─────────────────────────────────────────┤
│  Configuração: LLM_PROVIDER             │
│                                          │
│  ┌─────────┐        ┌─────────┐         │
│  │ OpenAI  │   ou   │ Gemini  │         │
│  │ gpt-4o  │        │ 2.0     │         │
│  └─────────┘        └─────────┘         │
│                                          │
│  Fallback automático em caso de erro    │
└─────────────────────────────────────────┘
```

---

## Configurações

### Variáveis de Ambiente

```bash
# Database
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/rag_db

# LLM
LLM_PROVIDER=gemini          # ou "openai"
GEMINI_API_KEY=xxx
OPENAI_API_KEY=xxx

# Storage
STORAGE_BACKEND=local        # ou "s3"
S3_BUCKET=my-bucket

# OCR
OCR_ENABLED=true
OCR_LANGUAGE=por+eng
OCR_MIN_TEXT_LENGTH=50
OCR_TIMEOUT=300

# Queue (opcional)
USE_SQS=false
SQS_QUEUE_URL=xxx
```

---

## Diagrama de Banco de Dados

```
┌─────────────────────────────────┐
│           documents             │
├─────────────────────────────────┤
│ id          UUID (PK)           │
│ title       VARCHAR             │
│ source      VARCHAR             │
│ project     VARCHAR             │
│ tags        JSONB               │
│ status      ENUM                │
│             (PENDING, INDEXING, │
│              INDEXED, FAILED)   │
│ created_at  TIMESTAMP           │
│ updated_at  TIMESTAMP           │
└─────────────────────────────────┘
              │
              │ 1:N
              ▼
┌─────────────────────────────────┐
│            chunks               │
├─────────────────────────────────┤
│ id          UUID (PK)           │
│ document_id UUID (FK)           │
│ chunk_index INTEGER             │
│ content     TEXT                │
│ chunk_meta  JSONB               │
│ embedding   VECTOR(384)         │ ◀── pgvector
│ created_at  TIMESTAMP           │
└─────────────────────────────────┘
```

---

## Extensões Futuras

1. **Mais formatos**: DOCX, XLSX, HTML
2. **Reranking**: Cross-encoder para melhorar relevância
3. **Hybrid Search**: Combinar BM25 + vetorial
4. **Streaming**: Respostas LLM em stream
5. **Cache**: Redis para embeddings frequentes
