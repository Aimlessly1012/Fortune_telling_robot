"""把 books 目录中的文本批量导入本地 Chroma 向量库。"""

import argparse
import sys
from pathlib import Path

# 兼容 VSCode 直接运行当前文件：python Fortune_telling_robot/ingest.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from embeddings import knowledge_base


def main() -> None:
    """解析导入参数并执行知识库构建。"""
    parser = argparse.ArgumentParser(description="导入本地 txt 到 Chroma 向量数据库。")
    parser.add_argument("--batch-size", type=int, default=64, help="每批写入的 chunk 数。")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先删除 Chroma collection 再重新导入；换 embedding 模型时必须使用。",
    )
    args = parser.parse_args()

    print(f"Embedding model: {knowledge_base.embedding_model}")
    print(f"Books path: {knowledge_base.books_dir}")
    print(f"Chroma path: {knowledge_base.chroma_dir}")
    knowledge_base.ingest(batch_size=args.batch_size, reset=args.reset)


if __name__ == "__main__":
    main()
