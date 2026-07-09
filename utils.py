"""与模型无关的日期校验和八字计算函数。"""

from datetime import datetime

from lunar_python import Solar


def is_valid_date(s: str) -> bool:
    """检查字符串是否为真实存在的 YYYY-MM-DD 日期。"""
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_bazi(birth_date: str, birth_time: str | None = None) -> dict:
    """把公历出生时间转换为年、月、日、时四柱。"""
    if birth_time:
        birth_datetime = datetime.strptime(
            f"{birth_date} {birth_time}",
            "%Y-%m-%d %H:%M",
        )
    else:
        # 中午只用于稳定计算年月日柱；没有出生时间时不返回虚假的时柱。
        birth_datetime = datetime.strptime(
            birth_date,
            "%Y-%m-%d",
        ).replace(hour=12)

    # lunar_python 负责公历、农历以及干支历法之间的转换。
    solar = Solar.fromDate(birth_datetime)
    lunar = solar.getLunar()
    eight_char = lunar.getEightChar()

    return {
        "birth_date": birth_date,
        "birth_time": birth_time,
        "year_pillar": eight_char.getYear(),
        "month_pillar": eight_char.getMonth(),
        "day_pillar": eight_char.getDay(),
        "time_pillar": eight_char.getTime() if birth_time else None,
        "complete": birth_time is not None,
        "message": (
            "已计算完整四柱"
            if birth_time
            else "缺少出生时间，目前只计算年柱、月柱和日柱"
        ),
    }
