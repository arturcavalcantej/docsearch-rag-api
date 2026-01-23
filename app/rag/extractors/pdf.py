import io
import logging
from typing import Optional

from app.core.config import settings
from app.rag.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extrator para arquivos PDF com fallback para OCR."""

    # Magic bytes para PDF
    PDF_MAGIC = b"%PDF"

    def can_handle(self, content: bytes, filename: Optional[str] = None) -> bool:
        """Verifica se e um arquivo PDF."""
        # Verifica magic bytes
        if content[:4] == self.PDF_MAGIC:
            return True

        # Verifica extensao
        if filename:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            return ext == "pdf"

        return False

    def extract(self, content: bytes, filename: Optional[str] = None) -> str:
        """Extrai texto do PDF, usando OCR se necessario."""
        import pdfplumber

        text_parts = []

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"Erro ao extrair texto do PDF com pdfplumber: {e}")

        full_text = "\n".join(text_parts).strip()

        # Se texto extraido e muito curto e OCR esta habilitado, tenta OCR
        if len(full_text) < settings.OCR_MIN_TEXT_LENGTH and settings.OCR_ENABLED:
            logger.info("Texto insuficiente no PDF, tentando OCR...")
            ocr_text = self._extract_with_ocr(content)
            if len(ocr_text) > len(full_text):
                return ocr_text

        return full_text

    def _extract_with_ocr(self, content: bytes) -> str:
        """Converte PDF em imagens e aplica OCR."""
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(
                content,
                dpi=300,
                timeout=settings.OCR_TIMEOUT,
            )

            text_parts = []
            for i, image in enumerate(images):
                logger.debug(f"Processando pagina {i + 1} com OCR...")
                page_text = pytesseract.image_to_string(
                    image,
                    lang=settings.OCR_LANGUAGE,
                    timeout=settings.OCR_TIMEOUT,
                )
                text_parts.append(page_text)

            return "\n".join(text_parts).strip()

        except Exception as e:
            logger.error(f"Erro ao aplicar OCR no PDF: {e}")
            return ""
