from pathlib import Path

# ========== 默认常量 ==========
# 生肖
ZODIACS = "鼠牛虎兔龙蛇马羊猴鸡狗猪"

# 星座边界：每月的分界日
CONSTELLATIONS = [
    (120, "摩羯座"), (219, "水瓶座"), (321, "双鱼座"), (420, "白羊座"),
    (521, "金牛座"), (622, "双子座"), (723, "巨蟹座"), (823, "狮子座"),
    (923, "处女座"), (1024, "天秤座"), (1123, "天蝎座"), (1222, "射手座"),
    (1232, "摩羯座"),
]

# 文件路径
BASE_DIR = Path(__file__).resolve().parent
BOOKS_DIR = BASE_DIR / "books"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "books"
