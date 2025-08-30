from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from .email_service import email_service


class WorkflowState(TypedDict):
    question: str
    top_k: int
    answer: str
    sources: List[str]
    email_status: Dict[str, Any]


def rag_node(state: WorkflowState) -> WorkflowState:
    """
    RAG node - performs retrieval and answer synthesis
    """
    # Import here to avoid circular imports
    from .main import retrieve, synthesize_answer, aggregate_sources_with_pages

    question = state["question"]
    top_k = state["top_k"]

    # Perform RAG retrieval and synthesis
    hits = retrieve(question, top_k)

    if not hits:
        answer = "I don't know based on the current index."
        sources = []
    else:
        answer = synthesize_answer(question, hits)
        sources = aggregate_sources_with_pages(hits)

    return {**state, "answer": answer, "sources": sources}


def email_node(state: WorkflowState) -> WorkflowState:
    """
    Email node - sends notification email with question and answer
    """
    question = state["question"]
    answer = state["answer"]
    sources = state["sources"]

    # Send email notification
    email_result = email_service.send_rag_notification(question, answer, sources)

    return {**state, "email_status": email_result}


def create_workflow() -> CompiledGraph:
    """
    Create and compile the LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("rag", rag_node)
    workflow.add_node("email", email_node)

    # Define the flow
    workflow.set_entry_point("rag")
    workflow.add_edge("rag", "email")
    workflow.add_edge("email", END)

    # Compile the graph
    return workflow.compile()


# Global workflow instance
rag_workflow = create_workflow()
