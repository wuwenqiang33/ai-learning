
import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from typing import List

class BigModelEmbeddingFunc(Embeddings, BaseModel):

    """自定义DeepSeek Embedding 嵌入类，符合LangChain标准"""
    api_key: str = Field(..., description=" API Key") # ... 是 Python 内置常量 Ellipsis，Pydantic 里用来表示：此字段必填，没有默认值，实例化时必须手动传参
    base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4/embeddings", description="向量接口地址")
    model: str = Field(default="embedding-3", description="嵌入模型名称")
    def embed_query(self,text:str) -> List[float]:
        """
        单条文本向量化：用户提问时调用（检索用）
        """
        resp = requests.post(
            url = self.base_url,
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json = {
                "model":self.model,
                "input":text
            }
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    
    def embed_documents(self, texts:List[str]) -> List[List[float]]:
        """
        批量文档向量化：Chroma.from_documents 入库时批量调用
        :param texts:字符串列表，多条文档切片
        :return: 二维向量列表，每条文本对应一个向量 
        """
        resp = requests.post(
            url = self.base_url,
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json = {
                "model":self.model,
                "input":texts
            }
        )
        resp.raise_for_status()
        data = resp.json()
        # 按照输入顺序对齐向量
        embeds = sorted(data["data"],key = lambda x: x["index"])
        return [item["embedding"] for item in embeds]