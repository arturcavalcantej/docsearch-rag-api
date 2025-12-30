from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.chunk import Chunk
from app.models.document import Document

async def retrieve_top_chunks(
    db: AsyncSession,
    query_vec: list[float],
    top_k: int = 5,
    project: str | None = None,
    source: str | None = None,
):
    stmt = (
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.document_id)
    )

    if project:
        stmt = stmt.where(Document.project == project)
    if source:
        stmt = stmt.where(Document.source == source)

    # cosine_distance funciona bem com embeddings normalizados
    stmt = stmt.order_by(Chunk.embedding.cosine_distance(query_vec)).limit(top_k)

    res = await db.execute(stmt)
    return res.all()  # lista de tuplas (Chunk, Document)
