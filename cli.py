"""命令行交互层：负责收集输入和展示结果，不承载业务判断。"""

from langchain_core.messages import AIMessage

from embeddings import knowledge_base
from graph import chat, resume_with_birth_info


def read_form(action: dict) -> dict:
    """根据 LangGraph 中断节点提供的字段定义收集用户信息。"""
    print(f"\nAI：{action['message']}")
    form_data = {}

    for field in action["fields"]:
        name = field["name"]
        prompt = field["label"]
        placeholder = field.get("placeholder", "")
        old_value = field.get("value", "")

        if placeholder:
            prompt += f"（{placeholder}）"
        if old_value:
            prompt += f"［当前：{old_value}］"

        value = input(f"{prompt}：").strip()
        form_data[name] = value or old_value

    return form_data


def print_result(result: dict) -> None:
    """把图执行结果转换成适合终端阅读的文本。"""
    messages = result.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        print(f"\nAI：{messages[-1].content}")

    bazi_info = result.get("bazi_info", {})
    if bazi_info.get("success"):
        print("\n八字计算：")
        print("年柱：", bazi_info["year_pillar"])
        print("月柱：", bazi_info["month_pillar"])
        print("日柱：", bazi_info["day_pillar"])
        print("时柱：", bazi_info["time_pillar"] or "缺少出生时间")

    sources = result.get("rag_sources", [])
    if sources:
        print("\n参考资料：")
        for source in sources:
            print(f"- {source}")


def run_cli(user_id: str = "user-1") -> None:
    """启动单用户命令行会话。"""
    knowledge_base.ensure_ready()
    print("AI：你好，有什么想聊的吗？")
    print("输入“退出”结束程序。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"退出", "exit", "quit"}:
            print("AI：再见。")
            return
        if not user_input:
            continue

        result = chat(user_id, user_input)
        # 图可能因缺少出生信息暂停；提交表单后从同一线程继续执行。
        while result.get("__interrupt__"):
            action = result["__interrupt__"][0].value
            result = resume_with_birth_info(user_id, read_form(action))

        print_result(result)
