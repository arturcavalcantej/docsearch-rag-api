from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.bm25 import bm25_search
from app.rag.reranker import rerank

async def retrieve_top_chunks(
    db: AsyncSession,
    query_vec: list[float],
    top_k: int = 5,
    project: str | None = None,
    source: str | None = None,
):
    """"Busca vetorial para (cosine distance)"""
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

#RRF
def reciprocal_rank_fusion(
    rankings: list[list[tuple]],
    k: int = 60,
) -> list[tuple]:
    """
    Combina múltiplos rankings usando RRF.
    Cada ranking é lista de (Chunk, Document, ...).
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, tuple] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking):
            chunk = item[0]
            doc = item[1]
            chunk_id = str(chunk.id)

            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = (chunk, doc)
            
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]

#VECTOR + BM25 - TOP20
async def retrieve_hybrid(
    db: AsyncSession,
    query_text: str,
    query_vec: list[float],
    top_k: int = 5,
    project: str | None = None,
    source: str | None = None,
    use_reranking: bool = False
):
    """Hybrid search: Vector + BM25 com Reciprocal Rank Fusion."""
    # ESTÁGIO 1: Retrieval amplo

    # Vector search (top 20 candidatos)
    vector_results = await retrieve_top_chunks(db, query_vec, top_k=20, project=project, source=source)
    vector_ranked = [(chunk, doc) for chunk, doc in vector_results]

    # BM25 (top 20 candidatos)
    bm25_results = await bm25_search(db, query_text, top_k=20, project=project, source=source)
    bm25_ranked = [(chunk, doc) for chunk, doc, score in bm25_results]

    # ESTAGIO 2:
    # Fusão com RRF
    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])
    candidates = fused[:20]

    # ESTAGIO 3 Reranking (cross-enconder):
    if use_reranking and candidates:
        return rerank(query_text, candidates, top_k)

    return candidates[:top_k]

