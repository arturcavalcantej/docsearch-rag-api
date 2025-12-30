from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from uuid import UUID

from app.models.document import Document
from app.models.chunk import Chunk


async def create_document(db: AsyncSession, title: str, source: str, project: str, tags:dict) -> Document:
    doc = Document(title=title,source=source,project=project,tags=tags,status='PENDING')
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc

async def get_document(db: AsyncSession, document_id: UUID) -> Document | None:
    res = await db.execute(select(Document).where(Document.id==document_id))
    return res.scalar_one_or_none()

async def set_document_status(db: AsyncSession, document_id: UUID, status: str) -> None:
    await db.execute(update(Document).where(Document.id == document_id).values(status=status))
    await db.commit()

async def list_chunks_by_document(db: AsyncSession, document_id: UUID, limit: int = 50) -> list[Chunk]:
    res = await db.execute(select(Chunk).where(Chunk.document.id==document_id).limit(limit))
    return list(res.scalars().all())

    