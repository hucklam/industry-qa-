import chromadb
from chromadb.config import Settings
import httpx
import os
from typing import List, Dict

class VectorStore:
    """向量存储服务"""

    def __init__(self, persist_directory: str = "./data"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com")

    def _get_embedding(self, text: str) -> List[float]:
        """调用 MiniMax API 获取嵌入向量"""
        url = f"{self.base_url}/v1/text/embeddings"

        response = httpx.post(
            url,
            json={
                "model": "embedding-001",
                "input": text
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

        if response.status_code != 200:
            raise Exception(f"嵌入API调用失败: {response.status_code} - {response.text}")

        result = response.json()
        return result["data"][0]["embedding"]

    def get_or_create_collection(self, name: str = "knowledge_base"):
        """获取或创建集合"""
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"description": "行业知识库"}
        )
        return self.collection

    def add_documents(self, doc_id: str, chunks: List[str], filename: str):
        """添加文档到向量库"""
        if not self.collection:
            self.get_or_create_collection()

        # 调用 MiniMax 嵌入模型
        embeddings = []
        for chunk in chunks:
            embedding = self._get_embedding(chunk)
            embeddings.append(embedding)

        # 存储到 Chroma
        self.collection.add(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
        )

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """相似度搜索"""
        if not self.collection:
            self.get_or_create_collection()

        # 查询嵌入
        query_embedding = self._get_embedding(query)

        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 格式化返回
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        return [
            {"content": doc, "source": meta.get("filename", "未知")}
            for doc, meta in zip(documents, metadatas)
        ]

    def delete_document(self, doc_id: str):
        """删除文档"""
        if not self.collection:
            return

        ids_to_delete = [f"{doc_id}_{i}" for i in range(1000)]
        try:
            self.collection.delete(ids=ids_to_delete)
        except:
            pass
