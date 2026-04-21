"""Schemas de resposta do agent."""
from pydantic import BaseModel, Field


class CitationRef(BaseModel):
    """Referência a um chunk específico."""
    document_id: str
    chunk_index: int


class AgentAnswer(BaseModel):
    """Resposta estruturada do agent."""
    answer: str = Field(..., description="A resposta em linguagem natural")
    citations: list[CitationRef] = Field(
        default_factory=list,
        description="Documentos e chunks citados na resposta",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confiança na resposta (0 baixa, 1 alta)",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Nomes das tools usadas para responder",
    )
    reasoning: str = Field(
        default="",
        description="Breve explicação de como chegou na resposta",
    )
