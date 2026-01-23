from app.core.config import settings

# Clients singleton
_openai_client = None
_gemini_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


async def generate_with_openai(question: str, context: str) -> str:
    """Gera resposta usando OpenAI."""
    client = get_openai_client()

    system_prompt = """Voce eh um assistente que responde perguntas com base no contexto fornecido.
- Responda APENAS com base no contexto fornecido
- Se o contexto nao contiver informacao suficiente, diga isso claramente
- Seja conciso e direto
- Responda no mesmo idioma da pergunta"""

    user_prompt = f"""Contexto:
{context}

Pergunta: {question}

Resposta:"""

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()


async def generate_with_gemini(question: str, context: str) -> str:
    """Gera resposta usando Gemini."""
    import asyncio
    client = get_gemini_client()

    prompt = f"""Voce eh um assistente que responde perguntas com base no contexto fornecido.
- Responda APENAS com base no contexto fornecido
- Se o contexto nao contiver informacao suficiente, diga isso claramente
- Seja conciso e direto
- Responda no mesmo idioma da pergunta

Contexto:
{context}

Pergunta: {question}

Resposta:"""

    # Gemini sync API - run in thread pool
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.3, "max_output_tokens": 1000}
    )

    return response.text.strip()


async def generate_answer(question: str, context: str) -> str:
    """Gera resposta usando o provider configurado, com fallback."""
    providers = []

    # Provider principal conforme config
    if settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        providers.append(("gemini", generate_with_gemini))
    elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        providers.append(("openai", generate_with_openai))

    # Fallback: adiciona o outro provider se disponivel
    if settings.GEMINI_API_KEY and not any(p[0] == "gemini" for p in providers):
        providers.append(("gemini", generate_with_gemini))
    if settings.OPENAI_API_KEY and not any(p[0] == "openai" for p in providers):
        providers.append(("openai", generate_with_openai))

    if not providers:
        raise ValueError("Nenhum LLM configurado (OPENAI_API_KEY ou GEMINI_API_KEY)")

    last_error = None
    for name, generate_fn in providers:
        try:
            return await generate_fn(question, context)
        except Exception as e:
            last_error = e
            continue

    raise last_error
