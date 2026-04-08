from enum import StrEnum


class DocumentStatus(StrEnum):
    """Status do documento no pipeline de ingestão."""
    PENDING = "PENDING"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
