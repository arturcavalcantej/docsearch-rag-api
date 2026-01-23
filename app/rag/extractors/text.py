from typing import Optional

from app.rag.extractors.base import BaseExtractor


class TextExtractor(BaseExtractor):
    """Extrator para arquivos de texto plano."""

    TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".rst", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd"}

    def can_handle(self, content: bytes, filename: Optional[str] = None) -> bool:
        """Verifica se e um arquivo de texto."""
        if filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext in self.TEXT_EXTENSIONS:
                return True

        # Tenta detectar se e texto UTF-8 valido
        try:
            content[:1024].decode("utf-8")
            # Verifica se nao tem muitos bytes nulos (indicativo de binario)
            null_count = content[:1024].count(b"\x00")
            return null_count < 10
        except UnicodeDecodeError:
            return False

    def extract(self, content: bytes, filename: Optional[str] = None) -> str:
        """Decodifica texto UTF-8."""
        return content.decode("utf-8", errors="ignore")
