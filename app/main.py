from fastapi import FastAPI

app = FastAPI(title="RAG Docs API")

@app.get("/health")
def health():
    return {"status": "ok"}