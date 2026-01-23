from abc import ABC, abstractmethod
from typing import Optional


class BaseExtractor(ABC):
    """Interface base para extratores de texto."""

    @abstractmethod
    def extract(self, content: bytes, filename: Optional[str] = None) -> str:
        """
        Extrai texto do conteudo.

        Args:
            content: Bytes do arquivo
            filename: Nome do arquivo (opcional, para deteccao de tipo)

        Returns:
            Texto extraido
        """
        pass

    @abstractmethod
    def can_handle(self, content: bytes, filename: Optional[str] = None) -> bool:
        """
        Verifica se este extrator pode processar o conteudo.

        Args:
            content: Bytes do arquivo
            filename: Nome do arquivo (opcional)

        Returns:
            True se pode processar, False caso contrario
        """
        pass
