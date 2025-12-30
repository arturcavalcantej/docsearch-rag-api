import os
from pathlib import Path
from uuid import UUID

BASE_DIR = Path("data/uploads")

def save_upload_file(document_id: UUID, filename: str, content: bytes) -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    path = BASE_DIR / f"{document_id}_{safe_name}"
    path.write_bytes(content)
    return path
