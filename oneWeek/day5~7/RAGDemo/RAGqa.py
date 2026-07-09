
# 个人文档问答系统

import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from BigModelEmbedding import BigModelEmbeddingFunc
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

if __name__ == '__main__':
    # 配置加载
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config_bigmodel.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件不存在: {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件格式异常: {e}")
        raise

    # ========== 配置区 ==========
    DEEPSEEK_KEY = config["deepseek"]["api_key"]
    EMBEDDING_KEY = config["embedding"]["api_key"]
    DEEPSEEK_URL = config["deepseek"]["base_url"]
    CHROMA_DB_PATH = config["chroma"]["persist_directory"]
    # ============================

    # 1. 加载文档
    doc_path = os.path.join(os.path.dirname(__file__), "doc", "kkfile.md")
    print("当前文档路径:", os.path.dirname(__file__))
    try:
        loader = TextLoader(doc_path, encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        print(f"错误: 文档加载失败: {e}")
        raise

    if not docs:
        print("警告: 文档为空，请检查文件路径和内容")
        raise SystemExit(1)
    print(f"加载文档 {len(docs)} 篇")

    # 2. 文本分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"分块数: {len(chunks)}")

    # 3. 向量化 + 存储
   
    try:
        embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)
        # vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
        # vectorstore.persist()  # 持久化 虽然会自动持久化，为确保持久化成功，这里手动持久化
        # 优化 先尝试加载已有库 找不到则创建
        if os.path.exists(CHROMA_DB_PATH):
            vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            print(f"加载已有向量库，共 {vectorstore._collection.count()} 条记录")
        else:
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
            vectorstore.persist()
            print(f"新建向量库，存入 {len(chunks)} 个分块")
    except Exception as e:
        print(f"错误: 向量化失败: {e}")
        raise

    # 4. 构建问答链
    # 构建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 自定义提示词
    prompt = PromptTemplate.from_template("""请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：
参考资料：{context}
用户问题：{input}
""")

    # 初始化LLM
    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0.7,
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_URL
    )

    # 构建 RAG 管道
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("========== RAG问答系统 ==========")
    try:
        result = rag_chain.invoke("历史更新记录")
        print("回答结果:", result)
        
        print("========== 第二次问答 ==========")

        result = rag_chain.invoke("kkFileView的官网地址？")
        print("回答结果:", result)
    except Exception as e:
        print(f"错误: LLM 调用失败: {e}")
        raise