"""Guardrails para input e output do agent."""
import re
import logging

logger = logging.getLogger(__name__)


# ===== INPUT GUARDRAILS ===== #

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|prior|earlier)\s+(instructions?|prompts?|messages?)",
    r"disregard\s+(previous|above|prior)",
    r"system\s+prompt",
    r"you\s+are\s+now\s+(a|an)",
    r"new\s+instructions?\s*:",
    r"forget\s+everything",
    r"reveal\s+(your|the)\s+(prompt|instructions|system)",
    # PT-BR
    r"ignore\s+(as\s+)?instru[çc][õo]es\s+anteriores",
    r"esque[çc]a\s+(tudo|as\s+instru[çc][õo]es)",
    r"revele\s+(o|seu)\s+(prompt|sistema|instru[çc][õo]es)",
]


class PromptInjectionDetected(Exception):
    """Prompt injection detectado no input."""
    pass


def detect_prompt_injection(text: str) -> bool:
    """Detecta padrões comuns de prompt injection."""
    text_lower = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning("Prompt injection detectado", extra={"pattern": pattern})
            return True
    return False


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """Sanitiza input do usuário."""
    # Truncar
    if len(text) > max_length:
        text = text[:max_length]
    # Remover caracteres de controle
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# ===== OUTPUT GUARDRAILS ===== #

def extract_citations(answer: str) -> list[tuple[str, int]]:
    """Extrai pares (doc_id, chunk_idx) da resposta."""
    # Padrões: [doc=abc chunk=3] ou [doc_id: abc, chunk_idx: 3]
    patterns = [
        r'\[doc=([^\s]+)\s+chunk=(\d+)\]',
        r'\[doc_id[:\s]+([^,\]]+)[,\s]+chunk[_\s]*idx?[:\s]+(\d+)\]',
    ]
    citations = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer):
            citations.append((match.group(1).strip(), int(match.group(2))))
    return citations


def validate_grounding(answer: str, retrieved_doc_ids: set[str]) -> dict:
    """
    Verifica se todas as citações na resposta referenciam chunks realmente recuperados.
    Retorna dict com {"valid": bool, "invalid_citations": list, "reason": str}
    """
    citations = extract_citations(answer)
    invalid = [
        (doc_id, chunk_idx)
        for doc_id, chunk_idx in citations
        if doc_id not in retrieved_doc_ids
    ]
    if invalid:
        return {
            "valid": False,
            "invalid_citations": invalid,
            "reason": f"Resposta cita {len(invalid)} documento(s) que não estavam no contexto",
        }
    return {"valid": True, "invalid_citations": [], "reason": ""}


def check_pii_leakage(text: str) -> list[str]:
    """Detecta possível vazamento de PII."""
    patterns = {
        "CPF": r"\d{3}\.\d{3}\.\d{3}-\d{2}",
        "Email": r"\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b",
        "Telefone": r"\(\d{2}\)\s*\d{4,5}-\d{4}",
        "Cartão": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }
    found = []
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            found.append(name)
    return found
