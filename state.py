from typing import Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class FortuneState(MessagesState):
    is_fortune_request: bool

    birth_year: str
    birth_month: str
    birth_day: str
    birth_date: str

    input_mode: str
    missing_fields: list
    can_analyze: bool

    bazi_info: dict

    rag_context: str
    rag_sources: list


class IntentResult(BaseModel):
    is_fortune_request: bool = Field(
        description="是否请求算命、八字、运势等命理服务"
    )


class ExtractedBirthInfo(BaseModel):
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
