import logging
from typing import Optional, List

from app.rag.extractors.base import BaseExtractor
from app.rag.extractors.pdf import PDFExtractor
from app.rag.extractors.image import ImageExtractor
from app.rag.extractors.text import TextExtractor

logger = logging.getLogger(__name__)


class TextExtractorOrchestrator:
    """Orquestrador que seleciona o extrator apropriado para cada arquivo."""

    def __init__(self):
        # Ordem importa: PDF e imagem antes de texto (mais especificos primeiro)
        self._extractors: List[BaseExtractor] = [
            PDFExtractor(),
            ImageExtractor(),
            TextExtractor(),
        ]

    def extract(self, content: bytes, filename: Optional[str] = None) -> str:
        """
        Extrai texto do conteudo usando o extrator apropriado.

        Args:
            content: Bytes do arquivo
            filename: Nome do arquivo (opcional, ajuda na deteccao de tipo)

        Returns:
            Texto extraido

        Raises:
            ValueError: Se nenhum extrator puder processar o conteudo
        """
        for extractor in self._extractors:
            if extractor.can_handle(content, filename):
                extractor_name = extractor.__class__.__name__
                logger.info(f"Usando {extractor_name} para extrair texto de {filename or 'arquivo'}")
                return extractor.extract(content, filename)

        # Fallback: tenta decodificar como UTF-8
        logger.warning(f"Nenhum extrator especifico encontrado para {filename or 'arquivo'}, tentando UTF-8")
        return content.decode("utf-8", errors="ignore")


# Singleton para reutilizacao
_orchestrator: Optional[TextExtractorOrchestrator] = None


def get_extractor() -> TextExtractorOrchestrator:
    """Retorna instancia singleton do orquestrador."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TextExtractorOrchestrator()
    return _orchestrator


def extract_text(content: bytes, filename: Optional[str] = None) -> str:
    """
    Funcao de conveniencia para extrair texto.

    Args:
        content: Bytes do arquivo
        filename: Nome do arquivo (opcional)

    Returns:
        Texto extraido
    """
    return get_extractor().extract(content, filename)
