import uuid
from datetime import datetime

from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import text
from app.models.enums import DocumentStatus
from app.models.base import Base

class Document(Base):
    __tablename__="documents"

    id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
    server_default=text("gen_random_uuid()"),
)
    title: Mapped[str] = mapped_column(String(255), default="Untitled")
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    project: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tags: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[DocumentStatus] = mapped_column(String(32), default=DocumentStatus.PENDING.value)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")