import os
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import set_document_status
from app.models.chunk import Chunk
from app.rag.chunking import chunk_text
from app.rag.embedder import embed_texts
from app.rag.extractor import extract_text
from app.storage.base import load_file


async def ingest_document(db: AsyncSession, document_id, file_path: str) -> None:
    """Processa documento: chunking, embedding e salva no banco."""
    try:
        await set_document_status(db, document_id, "INDEXING")

        # Carrega arquivo do storage (local ou S3)
        content = load_file(file_path)

        # Extrai nome do arquivo para ajudar na deteccao de tipo
        filename = os.path.basename(file_path)
        text = extract_text(content, filename)
        print(text[:2500])  # printa os primeiros 2500 caracteres
        chunks = chunk_text(text)
        print(f" {(chunks)} chunks.")

        if not chunks:
            await set_document_status(db, document_id, "FAILED")
            return


        vectors = embed_texts(chunks)
        print(vectors)
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
