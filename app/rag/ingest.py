import os
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import set_document_status
from app.models.chunk import Chunk
from app.rag.chunking import smart_chunk_text
from app.rag.embedder import embed_texts
from app.rag.extractor import extract_text
from app.storage.base import load_file
from app.core.config import settings
from app.models.enums import DocumentStatus

async def ingest_document(db: AsyncSession, document_id, file_path: str) -> None:
    """Processa documento: chunking, embedding e salva no banco."""
    try:
        await set_document_status(db, document_id, DocumentStatus.INDEXING)

        content = load_file(file_path)

        # Extrai nome do arquivo para ajudar na deteccao de tipo
        filename = os.path.basename(file_path)
        text = extract_text(content, filename)
        chunks = smart_chunk_text(
            text, strategy=settings.CHUNKING_STRATEGY,
            max_chars=settings.CHUNK_MAX_CHARS,overlap=settings.CHUNK_OVERLAP
        )

        if not chunks:
            await set_document_status(db, document_id,DocumentStatus.FAILED)
            return


        vectors = embed_texts(chunks)
        rows = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            rows.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=c,
                    chunk_meta={"source_path": file_path},
                    embedding=v,
                )
            )

        db.add_all(rows)
        await db.commit()

        await set_document_status(db, document_id, DocumentStatus.INDEXED)
    except Exception:
        await set_document_status(db, document_id, DocumentStatus.FAILED)
        raise
