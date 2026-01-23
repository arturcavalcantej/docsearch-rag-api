from pydantic import BaseModel
from typing import Any, Optional
from uuid import UUID
from datetime import datetime

class DocumentCreateResponse(BaseModel):
    id: UUID
    status: str

class DocumentRead(BaseModel):
    id: UUID
    title: str
    source: Optional[str] = None
    project: Optional[str] = None
    tags: dict[str, Any]
    status: str
    created_at: datetime

class ChunkRead(BaseModel):
    id: UUID
    chunk_index: int
    content: str
    chunk_meta: dict[str, Any]
