from uuid import UUID
from pathlib import Path
from app.core.config import settings


def save_file(document_id: UUID, filename: str, content: bytes) -> str:
    """Salva arquivo e retorna o path/key."""
    if settings.STORAGE_BACKEND == "s3":
        from app.storage.s3 import upload_to_s3
        return upload_to_s3(document_id, filename, content)
    else:
        from app.storage.local import save_upload_file
        return str(save_upload_file(document_id, filename, content))


def load_file(path_or_key: str) -> bytes:
    """Carrega arquivo do storage."""
    if settings.STORAGE_BACKEND == "s3":
        from app.storage.s3 import download_from_s3
        return download_from_s3(path_or_key)
    else:
        return Path(path_or_key).read_bytes()


def delete_file(path_or_key: str) -> None:
    """Deleta arquivo do storage."""
    if settings.STORAGE_BACKEND == "s3":
        from app.storage.s3 import delete_from_s3
        delete_from_s3(path_or_key)
    else:
        Path(path_or_key).unlink(missing_ok=True)
