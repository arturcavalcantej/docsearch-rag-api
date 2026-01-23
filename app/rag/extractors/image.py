import io
import logging
from typing import Optional

from app.core.config import settings
from app.rag.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """Extrator para imagens usando OCR."""

    # Magic bytes para formatos de imagem comuns
    IMAGE_SIGNATURES = {
        b"\x89PNG\r\n\x1a\n": "png",
        b"\xff\xd8\xff": "jpeg",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
        b"BM": "bmp",
        b"RIFF": "webp",  # WebP comeca com RIFF
        b"II*\x00": "tiff",  # TIFF little-endian
        b"MM\x00*": "tiff",  # TIFF big-endian
    }

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

    def can_handle(self, content: bytes, filename: Optional[str] = None) -> bool:
        """Verifica se e uma imagem."""
        # Verifica magic bytes
        for signature in self.IMAGE_SIGNATURES:
            if content[:len(signature)] == signature:
                return True

        # Verifica extensao
        if filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            return ext in self.IMAGE_EXTENSIONS

        return False

    def extract(self, content: bytes, filename: Optional[str] = None) -> str:
        """Extrai texto da imagem usando OCR."""
        if not settings.OCR_ENABLED:
            logger.warning("OCR desabilitado, nao e possivel extrair texto de imagem")
            return ""

        try:
            from PIL import Image
            import pytesseract

            image = Image.open(io.BytesIO(content))

            # Converte para RGB se necessario (ex: RGBA para JPG)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            text = pytesseract.image_to_string(
                image,
                lang=settings.OCR_LANGUAGE,
                timeout=settings.OCR_TIMEOUT,
            )

            return text.strip()

        except Exception as e:
            logger.error(f"Erro ao extrair texto da imagem: {e}")
            return ""
