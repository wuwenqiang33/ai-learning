# Chunking 策略实验脚本

"""
对比不同 chunk_size / chunk_overlap 对检索结果的影响。
运行方式: python chunk_experiment.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from lib.embedding import BigModelEmbeddingFunc
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 配置 ==========
with open(os.path.join(os.path.dirname(__file__), "config_bigmodel.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]

DOC_PATH = os.path.join(os.path.dirname(__file__), "doc", "kkfile.md")
TEST_QUERIES = [
    "kkFileView的官网地址？",
    "支持哪些3D模型格式？",
    "v5.0.0版本修复了哪些问题？",
]

# ========== 实验配置 ==========
# 每组: (splitter_name, splitter_instance)
splitters = [
    ("500_50",
     RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)),
    ("800_100",
     RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)),
    ("1000_200",
     RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
]


def load_and_split(splitter, name):
    """加载文档并用指定 splitter 分割，返回 chunks 统计"""
    docs = TextLoader(DOC_PATH, encoding="utf-8").load()
    chunks = splitter.split_documents(docs)

    # 统计信息
    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    print(f"\n{'='*60}")
    print(f"策略: {name}")
    print(f"  分块数: {len(chunks)}")
    print(f"  平均长度: {avg_len:.0f} 字符")
    print(f"  最小长度: {min(len(c.page_content) for c in chunks)}")
    print(f"  最大长度: {max(len(c.page_content) for c in chunks)}")

    # 展示第一个 chunk 的内容预览
    if chunks:
        print(f"  首个 chunk 预览: {chunks[0].page_content[:100]}...")

    return chunks


def evaluate_chunking(chunks, name, embeddings):
    """用简单检索+LLM判断来评估不同策略"""
    safe_name = "".join(name)
    db_path = f"./experiment_db_{safe_name}"
    
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0,
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_URL,
    )
    prompt = PromptTemplate.from_template("""请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：
参考资料：{context}
用户问题：{input}
""")
    # 构建 RAG 管道
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n  {name}检索给大模型的回答结果:")
    for query in TEST_QUERIES:
        result = rag_chain.invoke(query)
        print("回答结果:", result)


if __name__ == "__main__":
    print("=" * 60)
    print("Chunking 策略实验")
    print("=" * 60)

    embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

    for name, splitter in splitters:
        chunks = load_and_split(splitter, name)
        evaluate_chunking(chunks, name, embeddings)

    print("\n" + "=" * 60)
    print("实验完成")
    # print("  - chunk_size 太小 → 上下文碎片化，LLM 难以连贯理解")
    # print("  - chunk_size 太大 → 检索精度下降，混入无关信息")
    # print("  - chunk_overlap 太小 → 相邻块语义断裂")
    # print("  - chunk_overlap 太大 → 冗余增加，浪费 token")
    # print("  - 经验值: chunk_size=500~1000, overlap=50~200")
    # print("=" * 60)
