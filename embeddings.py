"""本地命理知识库的加载、切分、向量化和检索封装。"""

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from constant import BOOKS_DIR, CHROMA_DIR, COLLECTION_NAME


@dataclass
class KnowledgeBaseConfig:
    """知识库配置；dataclass 会自动生成 __init__ 等基础方法。"""

    books_dir: Path = BOOKS_DIR
    chroma_dir: Path = CHROMA_DIR
    collection_name: str = COLLECTION_NAME
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    retriever_k: int = 4


class KnowledgeBase:
    """封装本地 txt 知识库的导入、连接和检索能力。"""

    def __init__(
            self,
            config: KnowledgeBaseConfig | None = None,
            books_dir: Path | None = None,
            chroma_dir: Path | None = None,
            collection_name: str | None = None,
            embedding_model: str | None = None,
            ollama_base_url: str | None = None,
            retriever_k: int | None = None,
    ) -> None:
        # 显式参数优先于配置对象，便于测试或脚本临时覆盖默认值。
        base_config = config or KnowledgeBaseConfig()
        self.config = KnowledgeBaseConfig(
            books_dir=books_dir or base_config.books_dir,
            chroma_dir=chroma_dir or base_config.chroma_dir,
            collection_name=collection_name or base_config.collection_name,
            embedding_model=embedding_model or base_config.embedding_model,
            ollama_base_url=ollama_base_url or base_config.ollama_base_url,
            retriever_k=retriever_k or base_config.retriever_k,
        )
        self.books_dir = self.config.books_dir
        self.chroma_dir = self.config.chroma_dir
        self.collection_name = self.config.collection_name
        # 当前已有库是用 nomic-embed-text 写入的；换模型时请先执行 ingest.py --reset。
        self.embedding_model = self.config.embedding_model
        self.ollama_base_url = self.config.ollama_base_url
        self.retriever_k = self.config.retriever_k

        # 向量写入和查询必须使用同一个 embedding 模型。
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retriever_k}
        )

    def _create_vectorstore(self) -> Chroma:
        """创建 Chroma 连接；这里只连接，不会自动导入文档。"""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_dir),
        )

    def refresh_vectorstore(self) -> None:
        """重置 collection 后，刷新当前对象持有的 Chroma 和 retriever。"""
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retriever_k}
        )

    def count(self) -> int:
        """返回当前 Chroma collection 里的向量数量。"""
        return self.vectorstore._collection.count()

    def ensure_ready(self) -> None:
        """主程序启动前检查；知识库为空时提示先导入。"""
        count = self.count()
        if count == 0:
            raise RuntimeError(
                "Chroma 向量库为空，请先执行：python -m Fortune_telling_robot.ingest"
            )
        print(f"当前向量库数量: {count}")

    def load_txt_documents(self) -> list[Document]:
        """读取 books 目录下所有 txt，转成 LangChain Document。"""
        raw_docs = []
        for path in sorted(self.books_dir.rglob("*.txt")):
            raw_docs.append(
                Document(
                    page_content=path.read_text(encoding="utf-8", errors="ignore"),
                    metadata={"source": str(path)},
                )
            )
        return raw_docs

    def split_documents(self, raw_docs: list[Document]) -> list[Document]:
        """把长文本切成适合做向量检索的小块。"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        return text_splitter.split_documents(raw_docs)

    def build_ids(self, docs: list[Document]) -> list[str]:
        """生成稳定 ID，重复导入会覆盖同 ID 文档，而不是无限追加。"""
        ids = []
        source_counts: dict[str, int] = {}
        for doc in docs:
            source = Path(doc.metadata["source"]).stem
            index = source_counts.get(source, 0)
            source_counts[source] = index + 1
            ids.append(f"{source}-{index}")
        return ids

    def reset_collection(self) -> None:
        """删除当前 collection；换 embedding 模型或切分参数时需要重建。"""
        try:
            self.vectorstore.delete_collection()
        except ValueError:
            # collection 尚不存在时也允许继续创建空库。
            pass
        self.refresh_vectorstore()

    def ingest(self, batch_size: int = 64, reset: bool = False) -> int:
        """读取文本、切分、生成向量，并批量写入 Chroma。"""
        if reset:
            self.reset_collection()

        raw_docs = self.load_txt_documents()
        docs = self.split_documents(raw_docs)
        ids = self.build_ids(docs)

        print(f"读取文件数: {len(raw_docs)}")
        print(f"切分 chunk 数: {len(docs)}")
        print(f"导入前数量: {self.count()}")

        for start in range(0, len(docs), batch_size):
            end = start + batch_size
            # 分批写入，避免一次生成全部向量造成内存和服务压力。
            self.vectorstore.add_documents(docs[start:end], ids=ids[start:end])
            print(f"已写入: {min(end, len(docs))}/{len(docs)}", flush=True)

        print(f"导入后数量: {self.count()}")
        return len(docs)

# 应用进程共享同一个知识库连接，避免每个节点重复初始化 Chroma。
knowledge_base = KnowledgeBase()
