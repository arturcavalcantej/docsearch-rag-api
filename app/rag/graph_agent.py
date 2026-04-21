"""Agent com LangGraph — evolução do basic_agent."""
import asyncio
import logging
from typing import TypedDict, Annotated, Literal
from operator import add
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool as lc_tool

from app.core.config import settings
from app.rag.tools import execute_tool
from app.rag.agent_config import TOOL_TIMEOUT_SECONDS, MAX_ITERATIONS
from app.rag.guardrails import (
    detect_prompt_injection,
    sanitize_input,
    validate_grounding,
    check_pii_leakage,
    PromptInjectionDetected,
)
from app.schemas.agent import AgentAnswer

logger = logging.getLogger(__name__)

# State compartilhado entre nodes
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]  # messages acumulam
    iterations: int
    final_answer: dict

# Tools em formato LangChain
@lc_tool
async def search_documents(query: str, top_k: int = 5, use_hybrid: bool = False) -> str:
    """Busca trechos relevantes nos documentos indexados.
    Use quando a pergunta precisar de contexto da base de conhecimento."""
    return await execute_tool("search_documents", {
        "query": query, "top_k": top_k, "use_hybrid": use_hybrid
    })

@lc_tool
async def count_documents(project: str | None = None) -> str:
    """Conta quantos documentos estão indexados. Opcional: filtrar por projeto.
    Use quando o usuário perguntar 'quantos documentos você tem?'."""
    return await execute_tool("count_documents", {"project": project})


@lc_tool
async def summary_doc(query: str, top_k: int = 5) -> str:
    """Busca chunks relacionados e retorna o conteúdo para você (LLM) resumir.
    Use quando o usuário pede um 'resumo' ou 'visão geral' de um tema."""
    return await execute_tool("summary_doc", {"query": query, "top_k": top_k})

TOOLS = [search_documents, count_documents, summary_doc]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def _get_llm_with_tools():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )
    return llm.bind_tools(TOOLS)

# ======================= NODES ==================== #
#Cada nó é uma etapa#

async def format_answer_node(state: AgentState) -> dict:
    """Último node antes do output_guard: extrai resposta final em formato estruturado."""
    # Achar a última AIMessage que não tem tool_calls (resposta final do LLM)
    final_ai_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_ai_msg = msg
            break

    if not final_ai_msg:
        return {"final_answer": AgentAnswer(
            answer="Não foi possível gerar resposta.",
            confidence=0.0,
        ).model_dump()}

    # Coletar nomes de tools usadas no histórico
    tools_used = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tools_used.extend(tc["name"] for tc in msg.tool_calls)

    # Pedir ao LLM para estruturar (with_structured_output garante validação)
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,  # zero para ser determinístico
    )
    structured_llm = llm.with_structured_output(AgentAnswer)

    prompt = f"""Formate esta resposta de forma estruturada:

Resposta bruta: {final_ai_msg.content}
Tools usadas: {tools_used}

Extraia:
- answer: resposta clara para o usuário
- citations: documentos e chunks citados (formato [doc={{id}} chunk={{idx}}])
- confidence: sua confiança (0-1) com base na qualidade do contexto
- reasoning: breve justificativa
"""
    structured = await structured_llm.ainvoke(prompt)
    structured.tools_used = list(set(tools_used))

    return {"final_answer": structured.model_dump()}

async def output_guard_node(state: AgentState) -> dict:
    """Último node: valida a resposta antes de retornar."""
    final = state.get("final_answer", {})
    answer_text = final.get("answer", "")

    # 1. Check PII leakage
    pii_found = check_pii_leakage(answer_text)
    if pii_found:
        logger.warning("PII detectado na resposta", extra={"types": pii_found})
        # Opção: redatar ou rejeitar
        for pii_type in pii_found:
            answer_text = re.sub(r"\S+", "[REDACTED]", answer_text, count=1)
        final["answer"] = answer_text

    # 2. Validar grounding (citations existem no contexto)
    retrieved_ids = set()
    for msg in state["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            ids_in_msg = re.findall(r'doc=([^\s\]]+)', msg.content)
            retrieved_ids.update(ids_in_msg)

    grounding = validate_grounding(answer_text, retrieved_ids)
    if not grounding["valid"]:
        logger.warning("Grounding inválido", extra=grounding)
        final["confidence"] = min(final.get("confidence", 1.0), 0.3)
        final["reasoning"] = (
            final.get("reasoning", "") +
            f" [AVISO: {grounding['reason']}]"
        )

    return {"final_answer": final}

async def call_model(state: AgentState) -> dict:
    """Node: chama o LLM com o histórico."""

    llm = _get_llm_with_tools()
    response = await llm.ainvoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }



async def _run_single_tool(tool_call: dict) -> ToolMessage:
    """Executa UMA tool com timeout e error handling."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]

    logger.info("Executando tool", extra={"tool": tool_name, "args": tool_args})

    tool_fn = TOOLS_BY_NAME.get(tool_name)
    if not tool_fn:
        return ToolMessage(
            content=f"Tool desconhecida: {tool_name}",
            tool_call_id=tool_id,
        )

    try:
        result = await asyncio.wait_for(
            tool_fn.ainvoke(tool_args),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Tool timeout", extra={"tool": tool_name})
        result = f"Tool {tool_name} excedeu timeout de {TOOL_TIMEOUT_SECONDS}s"
    except Exception as e:
        logger.error("Tool falhou", extra={"tool": tool_name, "error": str(e)})
        result = f"Falha na tool {tool_name}: {type(e).__name__}: {e}"

    return ToolMessage(content=str(result), tool_call_id=tool_id)

async def execute_tools_node(state: AgentState) -> dict:
    """Node: executa tools em PARALELO, com timeout e retry."""
    last_message = state["messages"][-1]

    # Executar todas as tool_calls em paralelo
    tool_messages = await asyncio.gather(
        *[_run_single_tool(tc) for tc in last_message.tool_calls]
    )

    return {"messages": list(tool_messages)}


# ============== CONDITIONAL EDGES ============== #
#Arestas são transições condicionais#

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide: chamar tools ou terminar?"""
    last_message = state["messages"][-1]

    #Limite de iterações(segurança)
    if state["iterations"] >= MAX_ITERATIONS:
        logger.warning("Max iterations atingido")
        return "end"
    
    # Se o LLM pediu tools → executar
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Senão, terminou
    return "end"

# ========== BUILD GRAPH ========== #

def build_agent_graph():
    """Constrói o StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", execute_tools_node)
    graph.add_node("format_answer", format_answer_node)
    graph.add_node("output_guard", output_guard_node)

    graph.set_entry_point("call_model")

    graph.add_conditional_edges( 
        "call_model",
        should_continue,
        {"tools": "tools", "end": "format_answer"}
    )

    graph.add_edge("tools", "call_model") # depois de tools, volta ao modelo
    graph.add_edge("format_answer", "output_guard") # format → guard
    graph.add_edge("output_guard", END)  # guard → END

    return graph.compile()



# ========================================================== #

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph

# ========== RUN GRAPH - CHAMADA PRINCIPAL ========== #

SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre documentos indexados.
Use as tools disponíveis para buscar informação quando necessário.
Cite fontes usando [doc_id, chunk_idx]."""


async def run_graph_agent(user_question: str) -> dict:
    graph = get_graph()

    if detect_prompt_injection(user_question):
        return {
            "answer": "Desculpe, não posso processar essa solicitação.",
            "confidence": 0.0,
            "citations": [],
            "tools_used": [],
            "reasoning": "Input rejeitado por guardrail",
        }
    user_question = sanitize_input(user_question)
    initial_state = {
        "messages": [
            HumanMessage(content=f"{SYSTEM_PROMPT}\n\nPergunta: {user_question}")
        ],
        "iterations": 0,
        "final_answer": {},
    }

    final_state = await graph.ainvoke(initial_state)

    return final_state.get("final_answer", {
        "answer": "Não foi possível gerar resposta.",
        "confidence": 0.0,
        "citations": [],
        "tools_used": [],
        "reasoning": "",
    })
