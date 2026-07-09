from pathlib import Path

# 文件路径
BASE_DIR = Path(__file__).resolve().parent
BOOKS_DIR = BASE_DIR / "books"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "books"
