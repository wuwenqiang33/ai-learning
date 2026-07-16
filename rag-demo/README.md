# RAG Demo — AI 学习笔记

> 从 Day 1 到 Day X 持续迭代 的完整 RAG 学习路径，每一步都是可运行的独立脚本。

## 快速开始

```bash
# 1. 复制配置模板
cp config.example.json config.json

# 2. 编辑 config.json，填入你的 API Key

# 3. 按顺序运行
python steps/step01_basic.py       # Day 1-2: 最简 RAG
python steps/step02_chroma.py      # Day 3-4: ChromaDB 基础
python steps/step03_full_rag.py    # Day 5-7: 端到端 RAG
python steps/step04_chunk_exp.py   # Day 8:   Chunking 实验
python steps/step05_rerank.py      # Day 9:   Rerank 重排序
python steps/step06_multiformat.py # Day 10:  多格式最终版
python steps/step07_pgvector.py    # Day 11-12: PgVector 生产级 RAG
python steps/step08_ensemble.py    # Day 13:  混合检索（向量+BM25）
python steps/step09_query_rewrite.py # Day 14: 查询改写（Query Expansion）
```

## 目录结构

```
rag-demo/
├── config.example.json   # 配置模板（提交到 git）
├── config.json           # 本地配置（.gitignore 排除，含 API Key）
├── .gitignore
│
├── lib/                  # 🔧 公共库（不随天数变化）
│   ├── embedding.py      # 智谱 Embedding（LangChain 标准接口）
│   ├── loader.py         # 多格式文档解析器
│   └── config.py         # 配置加载工具
│
├── steps/                # 📅 按天的演进步骤（博客直接引用）
│   ├── step01_basic.py           # 最简 RAG
│   ├── step02_chroma.py          # ChromaDB CRUD
│   ├── step03_full_rag.py        # 端到端 RAG
│   ├── step04_chunk_exp.py       # Chunking 策略实验
│   ├── step05_rerank.py          # Rerank 重排序
│   └── step06_multiformat.py     # 多格式最终版
│   └── step07_pgvector.py        # PgVector 生产级 RAG
│   ├── step08_ensemble.py        # 混合检索（向量+BM25）
│   └── step09_query_rewrite.py   # 查询改写
│
├── experiments/          # 🔬 实验性代码（探索用，不影响主线）
├── docs/                 # 📄 测试文档（PDF/Word/Markdown 等）
├── output/               # 🗂️ 运行时产物（向量库数据，.gitignore 排除）
│
└── README.md
```

## 学习路径

| 步骤 | 主题 | 对应博客 |
|------|------|----------|
| Step 01 | 最简 RAG | Day 1-2 |
| Step 02 | ChromaDB 基础 | Day 3-4 |
| Step 03 | 端到端 RAG | Day 5-7 |
| Step 04 | Chunking 实验 | Day 8 |
| Step 05 | Rerank 重排序 | Day 9 |
| Step 06 | 多格式最终版 | Day 10 |
| Step 07 | PgVector 生产级 RAG | Day 11-12 |
| Step 08 | 混合检索（向量+BM25） | Day 13 |
| Step 09 | 查询改写（Query Expansion） | Day 14 |

## 技术栈

- **Embedding**: 智谱 BigModel (embedding-3)
- **LLM**: DeepSeek (deepseek-v4-pro)
- **向量库**: ChromaDB / PgVector (生产级)
- **框架**: LangChain 1.x
- **多格式**: PDF / Word / Excel / PPT / Markdown / Text
