"""
智谱 BigModel Embedding — 符合 LangChain Embeddings 标准接口

使用方式:
    from embedding import BigModelEmbeddingFunc
    embeddings = BigModelEmbeddingFunc(api_key="your-key")
    vec = embeddings.embed_query("你好")
"""

import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from typing import List


class BigModelEmbeddingFunc(Embeddings, BaseModel):
    """自定义智谱 Embedding 嵌入类，符合 LangChain 标准"""

    api_key: str = Field(..., description="API Key")
    base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/embeddings",
        description="向量接口地址",
    )
    model: str = Field(default="embedding-3", description="嵌入模型名称")

    def embed_query(self, text: str) -> List[float]:
        """单条文本向量化：用户提问时调用（检索用）"""
        try:
            resp = requests.post(
                url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"向量化失败: {e}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化：Chroma.from_documents 入库时批量调用"""
        try:
            resp = requests.post(
                url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            # 按照输入顺序对齐向量
            embeds = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeds]
        except Exception as e:
            print(f"向量化失败: {e}")
            return []
