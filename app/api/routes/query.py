from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.query import QueryRequest, QueryResponse, Citation
from app.rag.embedder import embed_query
from app.rag.retrieve import retrieve_top_chunks
from app.rag.llm import generate_answer
from app.core.config import settings

router = APIRouter(prefix="/query", tags=["query"])

@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    qvec = embed_query(req.question)

    hits = await retrieve_top_chunks(
        db=db,
        query_vec=qvec,
        top_k=req.top_k,
        project=req.project,
        source=req.source,
    )

    if not hits:
        return QueryResponse(
            answer="Nao encontrei contexto suficiente nos documentos indexados.",
            citations=[],
            retrieved_context_preview="",
        )

    context_parts = []
    citations = []
    for chunk, doc in hits:
        context_parts.append(f"[doc={doc.id} chunk={chunk.chunk_index}] {chunk.content[:500]}")
        citations.append(
            Citation(document_id=doc.id, chunk_id=chunk.id, chunk_index=chunk.chunk_index)
        )

    preview = "\n\n".join(context_parts)[:2000]

    # Gera resposta com LLM se habilitado e configurado
    if req.use_llm and (settings.OPENAI_API_KEY or settings.GEMINI_API_KEY):
        try:
            context_for_llm = "\n\n".join([chunk.content for chunk, doc in hits])
            answer = await generate_answer(req.question, context_for_llm)
        except Exception as e:
            # Fallback para resposta sem LLM em caso de erro
            answer = (
                f"[LLM indisponivel: {type(e).__name__}] "
                "Encontrei estes trechos relevantes nos documentos.\n\n"
                + preview
            )
    else:
        answer = (
            "Encontrei estes trechos relevantes nos documentos.\n\n"
            + preview
        )

    return QueryResponse(
        answer=answer,
        citations=citations,
        retrieved_context_preview=preview,
    )
