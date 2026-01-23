from pydantic import BaseModel
from typing import Any, Optional
from uuid import UUID

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    project: Optional[str] = None
    source: Optional[str] = None
    use_llm: bool = True

class Citation(BaseModel):
    document_id: UUID
    chunk_id: UUID
    chunk_index: int

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_context_preview: str
