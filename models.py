"""集中初始化大模型、结构化提取器和对话 Agent。"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from prompts import SYSTEM_PROMPT
from state import ExtractedBirthInfo, IntentResult

# 在创建模型客户端之前加载本地环境变量。
load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 结构化输出将模型结果约束为 Pydantic 数据，减少节点内的字符串解析。
intent_detector = model.with_structured_output(IntentResult)
birth_info_extractor = model.with_structured_output(ExtractedBirthInfo)

# 普通聊天和命理分析使用不同系统提示，避免职责相互污染。
normal_agent = create_agent(
    model=model,
    system_prompt=(
        "你是一个友好的聊天助手。"
        "正常回答用户的问题，不要编造个人信息。"
    ),
)

fortune_agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)
