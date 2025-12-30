from fastapi import FastAPI
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router

app = FastAPI(title="RAG Docs API")

app.include_router(documents_router)
app.include_router(query_router)

@app.get("/health")
def health():
    return {"status": "ok"}