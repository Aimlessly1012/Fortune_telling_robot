from langchain_core.tools import tool

from utils import get_bazi, is_valid_date


@tool
def calculate_bazi(birth_date: str) -> dict:
    """根据 YYYY-MM-DD 格式的公历出生日期计算年柱、月柱和日柱。"""
    if not is_valid_date(birth_date):
        return {
            "success": False,
            "error": "出生日期格式错误",
        }

    try:
        return {
            "success": True,
            **get_bazi(birth_date),
        }
    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }
