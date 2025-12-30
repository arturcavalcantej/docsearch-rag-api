from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import set_document_status
from app.models.chunk import Chunk
from app.rag.chunking import chunk_text
from app.rag.embedder import embed_texts

def extract_text_from_file(path: Path) -> str:
    # MVP: txt/md
    return path.read_text(encoding="utf-8", errors="ignore")

async def ingest_document(db: AsyncSession, document_id, file_path: Path) -> None:
    try:
        await set_document_status(db, document_id, "INDEXING")

        text = extract_text_from_file(file_path)
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
                    chunk_meta={"source_path": str(file_path)},
                    embedding=v,
                )
            )

        db.add_all(rows)
        await db.commit()

        await set_document_status(db, document_id, "INDEXED")
    except Exception:
        await set_document_status(db, document_id, "FAILED")
        raise
