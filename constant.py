"""项目内共享的文件路径和持久化名称。"""

from pathlib import Path

# 以源码目录为基准，避免程序从不同工作目录启动时路径漂移。
BASE_DIR = Path(__file__).resolve().parent
BOOKS_DIR = BASE_DIR / "books"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "books"
