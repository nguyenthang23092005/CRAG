from langgraph.graph import START, StateGraph, END
from function import retrieve, generate, evaluate_documents, transform_query, web_search, decide_to_generate, transform_query_to_search
from define_state import GraphState
from IPython.display import Image, display
from build_vectostores import add_conversation

workflow = StateGraph(GraphState)

workflow.add_node("tranform_query", transform_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", evaluate_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query_to_search", transform_query_to_search)
workflow.add_node("web_search", web_search)
workflow.add_node("add_conversation", add_conversation)


workflow.add_edge(START, "tranform_query")
workflow.add_edge("tranform_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_condition_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query_to_search",
        "generate": "generate",
    }
)
workflow.add_edge("transform_query_to_search", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "add_conversation")
workflow.add_edge("add_conversation", END)

app = workflow.compile()

def display_graph():
    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        pass

