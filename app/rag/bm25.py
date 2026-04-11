import logging
from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.chunk import Chunk
from app.models.document import Document

logger = logging.getLogger(__name__)

async def bm25_search(
    db: AsyncSession,
    query: str,
    top_k: int = 20,
    project: str | None = None,
    source: str | None = None,
) -> list[tuple[Chunk, Document, float]]:
    """Busca por keywords usando BM25."""
    stmt = select(Chunk, Document).join(Document, Document.id == Chunk.document_id)
    if project:
        stmt = stmt.where(Document.project == project)
    if source:
        stmt = stmt.where(Document.source == source)
    res = await db.execute(stmt)
    all_results = res.all()

    if not all_results:
        return []
    
    chunks_list = [row[0] for row in all_results]
    docs_list = [row[1] for row in all_results]

    corpus = [chunk.content.lower().split() for chunk in chunks_list]
    bm25 = BM25Okapi(corpus)

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)
    scored = list(zip(chunks_list,docs_list, scores))
    scored.sort(key=lambda x: x[2], reverse=True)

    return scored[:top_k]
