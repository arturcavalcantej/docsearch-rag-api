import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embedder import embed_query
from app.rag.retrieve import retrieve_top_chunks
from app.core.config import settings

logger = logging.getLogger(__name__)

def _get_llm():
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.3, max_tokens=1024)
    elif settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        from langchain_gemini import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL, api_key=settings.GEMINI_API_KEY, temperature=0.3, max_tokens=1024)
    
    return None

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Voce eh um assistente que responde perguntas com base no contexto fornecido.
- Responda APENAS com base no contexto fornecido
- Se o contexto nao contiver informacao suficiente, diga isso claramente
- Cite as fontes usando [doc_id, chunk_index]
- Seja conciso e direto
- Responda no mesmo idioma da pergunta"""),
    ("human", "Contexto:{context} Pergunta: {question} Resposta:")])

async def retrieve_context(question: str, db: AsyncSession, project: str | None = None, source: str | None = None, top_k: int = 5):
    qvec = embed_query(question)
    hits = await retrieve_top_chunks(db=db, query_vec=qvec, top_k=top_k, project=project, source=source)
    context_parts = []
    citations = []
    for chunk, doc in hits:
        context_parts.append(f"[doc={doc.id} chunk={chunk.chunk_index}] {chunk.content[:500]}")
        citations.append(f"[{doc.id}, {chunk.chunk_index}]")
    return "\n\n".join(context_parts), citations, hits

async def langchain_query(question:str, db: AsyncSession, top_k: int = 5, project:str | None = None, source:str | None = None, use_llm: bool = True) -> dict:
    """Pipeline RAG completo usando LangChain LCEL."""
    retrieval = await retrieve_context(question, db, top_k,project, source)

    if not retrieval["context"]:
        return {
            "answer": "Nao encontrei contexto suficiente nos documentos indexados.",
            "citations": [],
            "retrieved_context_preview": "",
        }
    preview = retrieval["context"][:2000]
    if use_llm:
        try:
            llm = _get_llm()
            chain = RAG_PROMPT | llm | StrOutputParser()
            answer = await chain.ainvoke({"context": retrieval["context"], "question": question})
        except Exception as e:
            logger.error(f"LLM generation failed: {type(e).__name__} - {str(e)}")
            answer = f"[LLM indisponivel: {type(e).__name__}] Encontrei estes trechos relevantes nos documentos.\n\n{preview}"
    else:
        answer = f"Encontrei estes trechos relevantes nos documentos.\n\n{preview}"
    return {
        "answer": answer,
        "citations": retrieval["citations"],    
        "retrieved_context_preview": preview,
    }