from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import set_document_status
from app.models.chunk import Chunk
from app.rag.chunking import chunk_text
from app.rag.embedder import embed_texts
from app.storage.base import load_file
from app.core.config import settings


def extract_text(content: bytes) -> str:
    """Extrai texto do conteudo do arquivo."""
    return content.decode("utf-8", errors="ignore")


async def ingest_document(db: AsyncSession, document_id, file_path: str) -> None:
    """Processa documento: chunking, embedding e salva no banco."""
    try:
        await set_document_status(db, document_id, "INDEXING")

        # Carrega arquivo do storage (local ou S3)
        content = load_file(file_path)
        text = extract_text(content)

        chunks = chunk_text(text)

        if not chunks:
            await set_document_status(db, document_id, "FAILED")
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

        await set_document_status(db, document_id, "INDEXED")
    except Exception:
        await set_document_status(db, document_id, "FAILED")
        raise
