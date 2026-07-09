"""工作流状态以及大模型结构化输出的数据模型。"""

from typing import Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class FortuneState(MessagesState):
    """在 LangGraph 节点之间共享的会话状态。"""

    # 路由状态
    is_fortune_request: bool

    # 用户出生日期；允许分多轮逐步补齐。
    birth_year: str
    birth_month: str
    birth_day: str
    birth_date: str

    # 人工表单和日期校验状态
    input_mode: str
    missing_fields: list
    can_analyze: bool

    # 确定性八字计算结果
    bazi_info: dict

    # 检索增强生成所需的上下文和引用来源
    rag_context: str
    rag_sources: list


class IntentResult(BaseModel):
    """意图识别模型的结构化返回值。"""

    is_fortune_request: bool = Field(
        description="是否请求算命、八字、运势等命理服务"
    )


class ExtractedBirthInfo(BaseModel):
    """出生日期提取模型的结构化返回值。"""

    birth_year: Optional[str] = Field(
        default=None,
        description="明确提供的四位出生年份，没有则返回 null",
    )
    birth_month: Optional[str] = Field(
        default=None,
        description="明确提供的出生月份，没有则返回 null",
    )
    birth_day: Optional[str] = Field(
        default=None,
        description="明确提供的出生日期，没有则返回 null",
    )
