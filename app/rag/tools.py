"""Tools que o LLM pode chamar."""
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embedder import embed_query
from app.rag.retrieve import retrieve_top_chunks, retrieve_hybrid
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


# === Descrições das tools em formato OpenAI ===

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Busca trechos relevantes nos documentos indexados. "
                "Use quando a pergunta precisar de contexto da base de conhecimento."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A pergunta ou termo de busca",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Quantos chunks retornar (padrão: 5)",
                        "default": 5,
                    },
                    "use_hybrid": {
                        "type": "boolean",
                        "description": "Se True, usa BM25+vector. Melhor para termos exatos.",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_without_context",
            "description": (
                "Responde diretamente quando a pergunta é simples (saudação, agradecimento) "
                "ou quando não há documentos relevantes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "A resposta direta",
                    },
                },
                "required": ["response"],
            },
        },
    },
]

async def execute_tool(name:str, args: dict) -> str:
    """Executa a tool pedida pelo LLM e retorna o resultado em string."""
    logger.info(f"Executando tool {name}", extra={"args": args})
    if name == "search_documents":
        async with AsyncSessionLocal() as db:
            qvec = embed_query(args["query"])
            if args.get("use_hybrid"):
                hits = await retrieve_hybrid(
                    db, args["query"], query_vec=qvec, top_k=args.get("top_k", 5),

                )
            else:
                hits = await retrieve_top_chunks(
                    db=db, query_vec=qvec, top_k=args.get("top_k", 5),

                )
            if not hits:
                return "Nenhum documento relevante encontrado."
            return "\n\n".join(
                f"[doc={doc.id} chunk={chunk.chunk_index}] {chunk.content[:500]}"
                for chunk, doc in hits
            )
    elif name == "answer_without_context":
        return args["response"]
    
    else:
        return f"Tool desconhecida {name}"