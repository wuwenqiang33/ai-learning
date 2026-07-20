"""
Step 11 — LangChain Agent（Day 15-16 进阶）

功能：用 LangChain 的 create_tool_calling_agent 构建企业数据查询 Agent。
相比手写 Agent 循环，LangChain 封装了工具注册、消息管理、错误处理，
让 Agent 开发更简洁、更可维护。

运行前准备：
    pip install langchain langchain-openai langchain-community

运行: python steps/step11_langchain_agent.py
"""

import sys, os

from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from lib.config import load_config

# ── 加载配置 ──
config = load_config()
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]

print("=" * 60)
print("Step 11 — LangChain Agent（企业数据查询）")
print("=" * 60)

# ── 1. 定义工具（用 @tool 装饰器，比手写简洁得多） ──
# 类比 Java：@Tool 注解代替了手动注册 Map<String, Function>

@tool
def query_database(sql: str) -> str:
    """查询企业数据库，执行 SQL 语句并返回结果。

    Args:
        sql: SQL 查询语句
    """
    sample_data = {
        "users": [
            {"id": 1, "name": "张三", "role": "管理员", "department": "技术部"},
            {"id": 2, "name": "李四", "role": "开发", "department": "技术部"},
            {"id": 3, "name": "王五", "role": "测试", "department": "质量部"},
        ],
        "orders": [
            {"order_id": "ORD-001", "user_id": 1, "amount": 1500.00, "status": "已完成"},
            {"order_id": "ORD-002", "user_id": 2, "amount": 3200.50, "status": "进行中"},
            {"order_id": "ORD-003", "user_id": 1, "amount": 800.00, "status": "已完成"},
        ],
    }
    return json.dumps(sample_data, ensure_ascii=False, indent=2)


@tool
def search_documents(query: str) -> str:
    """在企业知识库中搜索相关文档。

    Args:
        query: 搜索关键词
    """
    sample_results = [
        {"title": "企业员工手册 v3.0", "snippet": "员工入职流程包括：HR 面试、背景调查、签订劳动合同...", "score": 0.95},
        {"title": "报销流程指南", "snippet": "差旅报销需在出行后 30 天内提交，附发票和行程单...", "score": 0.87},
    ]
    return json.dumps(sample_results, ensure_ascii=False, indent=2)


@tool
def call_external_api(url: str, method: str = "GET", body: str = "") -> str:
    """调用外部 API 接口，发送 HTTP 请求并返回响应。

    Args:
        url: API 地址
        method: HTTP 方法（GET/POST/PUT/DELETE）
        body: 请求体（JSON 字符串）
    """
    return '{"status": "success", "data": {"code": 200, "message": "OK"}}'


# 工具列表（LangChain 自动从 @tool 装饰器提取 name、description、parameters）
tools = [query_database, search_documents, call_external_api]

print(f"\n[1] 已注册 {len(tools)} 个工具:")
for t in tools:
    print(f"    - {t.name}: {t.description}")


# ── 2. 构建 Agent ──
print("\n[2] 构建 Agent...")

# LLM
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    api_key=DEEPSEEK_KEY,
    base_url=DEEPSEEK_URL,
)

# 创建 Agent（一行代码搞定！）
# 新版 LangChain 用 create_agent 替代了 create_tool_calling_agent + AgentExecutor
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个企业智能助手，可以查询数据库、搜索文档和调用外部 API。",
    name="EnterpriseAssistant",
)


# ── 3. 测试 ──
print("\n[3] 测试 Agent...")

test_cases = [
    ("帮我查一下数据库里有哪些用户？", "查询数据库"),
    ("帮我搜索一下报销流程相关的文档", "搜索文档"),
    ("查询用户张三的信息，然后搜索他相关的订单", "多工具调用"),
    ("直接回答：今天天气怎么样？", "不调用工具"),
]

for query, desc in test_cases:
    print(f"\n{'─' * 50}")
    print(f"测试: {desc}")
    print(f"问题: {query}")
    print(f"{'─' * 50}")

    try:
        result = agent.invoke({"messages": [("human", query)]})
        print(f"\n  最终回答: {str(result)[:300]}...")
    except Exception as e:
        print(f"\n  错误: {e}")


# ── 4. 与手写 Agent 对比 ──
print("\n[4] 手写 vs LangChain 对比...")
print("""
手写 Agent 循环:
  - 优点：理解原理，完全可控
  - 缺点：需要自己处理消息管理、错误恢复、工具注册
  - 代码量：~150 行

LangChain Agent:
  - 优点：一行代码创建 Agent，自动消息管理，错误恢复
  - 缺点：黑盒，调试困难
  - 代码量：~80 行

Java 类比：
  - 手写 Agent = 自己写 Servlet + 手动路由
  - LangChain Agent = Spring MVC + @RestController
""")

print("\n" + "=" * 60)
print("Step 11 完成 — 这就是企业级 Agent 的开发方式！")
print("=" * 60)
