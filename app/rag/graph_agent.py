"""Agent com LangGraph — evolução do basic_agent."""
import logging
from typing import TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool as lc_tool

from app.core.config import settings
from app.rag.tools import execute_tool

logger = logging.getLogger(__name__)

# State compartilhado entre nodes
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]  # messages acumulam
    iterations: int

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
    pass


@lc_tool
async def summary_doc(query: str, top_k: int = 5, use_hybrid: bool = False) -> str:
    pass

TOOLS = [search_documents, count_documents, summary_doc]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def _get_llm_with_tools():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )
    return llm.bind_tools(TOOLS)

# ========== NODES ========== #
    """Cada nó é uma etapa"""

async def call_model(state: AgentState) -> dict:
    """Node: chama o LLM com o histórico."""

    llm = _get_llm_with_tools()
    response = await llm.ainvoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

async def execute_tools_node(state: AgentState) -> dict:
    """Node: executa as tools que o LLM pediu."""

    last_message = state["messages"][-1]
    tool_message = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f"Node: execute_tools", extra={"tool": tool_name, "args": tool_args})

        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if not tool_fn:
            result = f"Tool desconhecida: {tool_name}"
        else:
            try:
                result = await tool_fn.ainvoke(tool_args)
            except Exception as e:
                result = f"Falha na tool: {tool_name}: {e}"

        tool_message.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages":tool_message}


# ========== CONDITIONAL EDGES ========== #
    """Arestas são transições condicionais"""

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide: chamar tools ou terminar?"""
    last_message = state["messages"][-1]

    #Limite de iterações(segurança)
    if state["iterations"] >= 5:
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

    graph.set_entry_point("call_model")

    graph.add_conditional_edge(
        "call_model",
        should_continue,
        {"tools": "tools", "end": END}
    )

    graph.add_edge("tools", "call_model") # depois de tools, volta ao modelo

    return graph.compile()


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


async def run_graph_agent(user_question: str) -> str:
    """Executa o agent via LangGraph."""
    graph = get_graph()

    initial_state = {
        "messages": [
            HumanMessage(content=f"{SYSTEM_PROMPT}\n\nPergunta: {user_question}")
        ],
        "iterations": 0,
    }

    final_state = await graph.aivoke(initial_state)

    # Última AIMessage sem tool_calls é a resposta final
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content
        
    return "Não foi possivel gerar resposta"
