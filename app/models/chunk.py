import uuid
from datetime import datetime

from sqlalchemy import Integer, ForeignKey, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import text

from pgvector.sqlalchemy import Vector

from app.models.base import Base


class Chunk(Base):
    __tablename__="chunks"

    id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
    server_default=text("gen_random_uuid()"),
)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    chunk_meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    document = relationship("Document", back_populates="chunks")