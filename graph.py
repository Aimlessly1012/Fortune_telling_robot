"""组装并编译命理助手的 LangGraph 工作流。"""

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
    """定义普通聊天与命理分析两条执行路径。"""
    graph = StateGraph(FortuneState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("normal_chat", normal_chat_node)
    graph.add_node("extract_birth_info", extract_birth_info)
    graph.add_node("collect_birth_info", collect_birth_info)
    graph.add_node("calculate_bazi", calculate_bazi_node)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("fortune_agent", fortune_agent_node)

    graph.add_edge(START, "detect_intent")
    # 普通问题直接聊天；命理请求进入出生信息收集与分析流程。
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
    # collect_birth_info 恢复后再次校验日期，直到信息完整且合法。
    graph.add_edge("collect_birth_info", "extract_birth_info")
    graph.add_edge("calculate_bazi", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "fortune_agent")
    graph.add_edge("normal_chat", END)
    graph.add_edge("fortune_agent", END)
    return graph

# 内存检查点用于保存每个 thread_id 的中断状态；进程重启后不会持久化。
checkpointer = InMemorySaver()
app = build_graph().compile(checkpointer=checkpointer)


def get_config(user_id: str) -> dict:
    """把用户 ID 映射为 LangGraph 会话线程 ID。"""
    return {"configurable": {"thread_id": user_id}}


def chat(user_id: str, content: str):
    """向指定用户会话发送一条消息并执行工作流。"""
    return app.invoke(
        {
            "messages": [{"role": "user", "content": content}],
            "input_mode": "chat",
        },
        get_config(user_id),
    )


def resume_with_birth_info(user_id: str, form_data: dict):
    """提交补充的出生信息，从人工中断位置继续执行。"""
    return app.invoke(
        Command(resume=form_data),
        get_config(user_id),
    )
