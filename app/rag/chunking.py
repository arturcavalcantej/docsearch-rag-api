from typing import Iterable, Literal

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

def recursive_chunk_text(text:str, max_chars: int = 3500, overlap: int = 200, separators: list[str] | None = None) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    text = text.strip()
    if not text:
        return []
    
    if len(text) < max_chars:
        return [text]
    
    # Encontra o melhor separador (primeiro que existe no texto)
    best_sep = separators[-1]
    for sep in separators:
        if sep in text:
            best_sep = sep
            break


    chunks = []
    current = ""

    parts = text.split(best_sep)
    for part in parts:
        candidate = current + best_sep + part if current else part 
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Se a part sozinha é maior que max_chars, recurse com proximo separador
            if len(part) > max_chars:
                remaining_sep = separators[separators.index(best_sep) + 1:]
                if remaining_sep:
                    sub_chunks = recursive_chunk_text(part, max_chars, overlap, remaining_sep)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    chunks.append(part[:max_chars].strip())
                    current = ""
            else:
                current = part
        
    if current.strip():
        chunks.append(current.strip())

    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    return [c for c in chunks if c]

def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + " " + chunks[i])
    return result

def smart_chunk_text(
    text: str,
    strategy: Literal["fixed", "recursive"] = "recursive",
    max_chars: int = 3500,
    overlap: int = 200,
) -> list[str]:
    """Chunking com strategy configurável."""
    if strategy == "fixed":
        return chunk_text(text, max_chars, overlap)
    return recursive_chunk_text(text, max_chars, overlap)