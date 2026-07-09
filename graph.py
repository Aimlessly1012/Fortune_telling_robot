from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from nodes import (
    calculate_bazi_node,
    collect_birth_info,
    detect_intent,
    extract_birth_info,
    fortune_agent_node,
    normal_chat_node,
    retrieve_knowledge,
    route_after_extract,
    route_after_intent,
)
from state import FortuneState


def build_graph():
    graph = StateGraph(FortuneState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("normal_chat", normal_chat_node)
    graph.add_node("extract_birth_info", extract_birth_info)
    graph.add_node("collect_birth_info", collect_birth_info)
    graph.add_node("calculate_bazi", calculate_bazi_node)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("fortune_agent", fortune_agent_node)

    graph.add_edge(START, "detect_intent")
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "normal_chat": "normal_chat",
            "extract_birth_info": "extract_birth_info",
        },
    )
    graph.add_conditional_edges(
        "extract_birth_info",
        route_after_extract,
        {
            "collect_birth_info": "collect_birth_info",
            "calculate_bazi": "calculate_bazi",
        },
    )
    graph.add_edge("collect_birth_info", "extract_birth_info")
    graph.add_edge("calculate_bazi", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "fortune_agent")
    graph.add_edge("normal_chat", END)
    graph.add_edge("fortune_agent", END)
    return graph


checkpointer = InMemorySaver()
app = build_graph().compile(checkpointer=checkpointer)


def get_config(user_id: str) -> dict:
    return {"configurable": {"thread_id": user_id}}


def chat(user_id: str, content: str):
    return app.invoke(
        {
            "messages": [{"role": "user", "content": content}],
            "input_mode": "chat",
        },
        get_config(user_id),
    )


def resume_with_birth_info(user_id: str, form_data: dict):
    return app.invoke(
        Command(resume=form_data),
        get_config(user_id),
    )
