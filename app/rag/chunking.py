from typing import Iterable

def chunk_text(text: str, max_chars: int = 3500, overlap: int = 300) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == len(text):
            break
    return chunks

