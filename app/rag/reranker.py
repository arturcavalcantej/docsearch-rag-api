import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Singleton do cross-encoder para reranking."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder carregado")
    return _reranker


def rerank(query: str, candidates: list[tuple], top_k: int = 5) -> list[tuple]:
    """
    Reranqueia candidatos usando cross-encoder.
    
    Recebe: lista de (Chunk, Document)
    Retorna: mesma lista reordenada por relevância (cross-encoder score)
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Cross-encoder precisa de pares (query, texto_do_chunk)
    pairs = [(query, chunk.content) for chunk, doc in candidates]

    # Gera score para cada par — processa query+doc JUNTOS
    scores = reranker.predict(pairs)

    # Ordena por score do cross-encoder (maior = mais relevante)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info("Reranking concluido", extra={
        "candidates": len(candidates),
        "top_k": top_k,
        "top_score": float(scored[0][1]) if scored else 0,
    })

    return [item for item, score in scored[:top_k]]
