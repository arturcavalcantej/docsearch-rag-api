"""Agent simples com function calling — sem LangGraph ainda."""
import json
import logging
from google import genai
from google.genai import types
from app.core.config import settings
from app.rag.tools import TOOLS_SPEC, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas usando a base de conhecimento quando necessário.\n"
    "- Para saudações e perguntas triviais: use answer_without_context\n"
    "- Para perguntas que precisam de informação: use search_documents\n"
    "- Para termos exatos (códigos de erro, IDs): use search_documents com use_hybrid=true\n"
    "- Depois de receber o contexto, responda baseado nele com citações."
)

def _convert_tools_spec_to_gemini(tools_spec: list) -> list[types.Tool]:
    """Converte spec no formato OpenAI para formato Gemini."""
    function_declarations = []
    for tool in tools_spec:
        fn = tool["function"]
        function_declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=fn["parameters"],
        ))
    return [types.Tool(function_declarations=function_declarations)]


async def basic_agent(user_question: str, max_iterations: int = 5)-> str:
    """
    Agent que pode chamar tools em loop até decidir responder.
    Máximo de iterações para evitar loops infinitos.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    tools = _convert_tools_spec_to_gemini(TOOLS_SPEC)
    # Histórico da conversa
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_question)])
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        temperature=0.3,
    )

    for iteration in range(max_iterations):
        logger.info(f"Iteração {iteration + 1}")

        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=config
        )
        function_calls = response.function_calls

        if not function_calls:
            return response.text
        
        contents.append(response.candidates[0].content)

        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            logger.info(f"Tool: {tool_name}, args: {tool_args}")
            result = await execute_tool(tool_name,tool_args)

            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )]
            ))
    return "Limite de iterações atingido. Não consegui completar a tarefa."