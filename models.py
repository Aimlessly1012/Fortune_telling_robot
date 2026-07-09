import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from prompts import SYSTEM_PROMPT
from state import ExtractedBirthInfo, IntentResult

load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

intent_detector = model.with_structured_output(IntentResult)
birth_info_extractor = model.with_structured_output(ExtractedBirthInfo)

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
