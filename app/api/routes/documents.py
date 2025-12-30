from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_db, AsyncSessionLocal
from app.db.crud import create_document, get_document, list_chunks_by_document
from app.schemas.documents import DocumentCreateResponse, DocumentRead, ChunkRead
from app.storage.local import save_upload_file
from app.rag.ingest import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("", response_model=DocumentCreateResponse)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...), source: str | None = Form(None), project: str | None = Form(None), db: AsyncSession = Depends(get_db)):
    content = await file.read()
    doc = await create_document(
        db=db,
        title=file.filename,
        source=source,
        project=project,
        tags={}
    )
    path = save_upload_file(doc.id, file.filename, content)
    async def _run():
        async with AsyncSessionLocal() as task_db:
            await ingest_document(task_db, doc.id, path)

    background.add_task(_run)
    return DocumentCreateResponse(id=doc.id,status=doc.status)

@router.get("/{document_id}", response_model=DocumentRead)
async def read_document(document_id: UUID, db: AsyncSession = Depends(get_db)):
    doc = await get_document(db, document_id)
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail='Dcoument not found')
    return DocumentRead(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        project=doc.project,
        tags=doc.tags,
        status=doc.status,
        created_at=doc.created_at,
    )


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def read_chunks(document_id: UUID, db: AsyncSession = Depends(get_db)):
    chunks = await list_chunks_by_document(db, document_id, limit=100)
    return [
        ChunkRead(
            id=c.id,
            chunk_index=c.chunk_index,
            content=c.content,
            chunk_meta=c.metadata,
        )
        for c in chunks
    ]
