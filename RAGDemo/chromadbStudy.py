
# chromadb 学习

import chromadb
import requests
import os
import json
from chromadb import Documents,EmbeddingFunction,Embeddings

class DeepSeekEmbeddingFunc(EmbeddingFunction):
    def __init__(self,api_key:str):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    def __call__(self,texts:Documents) -> Embeddings:
        try:
            # 调用DeepSeek云端向量接口
            resp = requests.post(
                url = self.base_url,
                headers = self.headers,
                json = {
                    "model":"embedding-3",
                    "input":texts
                }
            )
            print("API调用状态码:", resp.status_code)
            # print("响应内容:", resp.text)
            resp.raise_for_status()
            resp_data = resp.json()
            # 提取向量列表返回给Chroma
            embeds = [item["embedding"] for item in resp_data["data"]]
            return embeds
        except Exception as e:
            print(f"调用失败: {e}")
            return None

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), "config_bigmodel.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    # ========== 配置区 ==========
    DEEPSEEK_KEY = config["api_key"]
    # 持久化向量库文件夹
    CHROMA_DB_PATH = "./my_knowledge_base"
    COLLECTION_NAME = "article"
# ============================
    # 持久化客户端
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 传入自定义向量化函数，创建/获取集合
    ds_embed = DeepSeekEmbeddingFunc(api_key=DEEPSEEK_KEY)
    # 创建/获取集合
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ds_embed,
        # 推荐开启余弦相似度匹配，更适配Embedding检索
        metadata={"hnsw:space": "cosine"}
    )

    # 写入数据
    # coll.add(
    #     documents=[
    #         "Chroma是轻量级向量数据库",
    #         "DeepSeek API可以生成文本Embedding向量",
    #         "RAG架构依靠向量库检索知识库"
    #     ],
    #     metadatas=[{"source": "note1"}, {"source": "note2"}, {"source": "note3"}],
    #     ids=["id1", "id2", "id3"]
    # )
    # 更新数据
    coll.update(
        ids=["id1"],
        documents=["Chroma是轻量级向量数据库，支持多种存储后端和向量化方法"],
        metadatas=[{"source": "note1"}]
    )
    # 删除数据
    coll.delete(ids=["id3"])
    # 返回集合中文档数量
    print("集合中文档数量:", coll.count())
    
    # 查询note1的条数
    print("note1的条数:", len(coll.get(where={"source":"note1"})["ids"]))

    # 语义查询（提问自动转向量，相似度召回）
    query_result = coll.query(
        query_texts=["向量数据库能用来做什么"],
        where = {"source": "note1"},  # 可选，按元数据过滤
        n_results=2  # 返回最相似2条
    )

    # 打印结果
    print("检索到的原文：")
    print(query_result["documents"])
    print("对应相似度距离：")
    print(query_result["distances"])