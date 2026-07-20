"""
Step 10 — Function Calling 手写 Agent 循环（Day 15-16 核心成果）

功能：用 OpenAI SDK 的 tools 参数实现 Function Calling，
      手写 Agent 主循环：模型决定调用哪个函数 → 执行工具 → 获取结果 → 继续推理。

这是理解 Function Calling 原理的最佳方式——不依赖任何 Agent 框架。

运行: python steps/step10_function_calling.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import json
from openai import OpenAI
from config import load_config

# ── 加载配置 ──
config = load_config()
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]

print("=" * 60)
print("Step 10 — Function Calling 手写 Agent 循环")
print("=" * 60)

# ── 1. 定义工具（类似 Java 的 Service 接口） ──
# 每个工具就是一个 Python 函数，用 docstring 描述用途

def query_database(sql: str) -> str:
    """查询企业数据库，执行 SQL 语句并返回结果。

    Args:
        sql: SQL 查询语句

    Returns:
        查询结果，格式为 JSON 字符串
    """
    # 实际项目中这里会连接数据库执行 SQL
    # 这里用模拟数据演示
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


def search_documents(query: str) -> str:
    """在企业知识库中搜索相关文档。

    Args:
        query: 搜索关键词

    Returns:
        搜索结果，格式为 JSON 字符串
    """
    # 实际项目中这里会调用 RAG 系统的检索接口
    sample_results = [
        {"title": "企业员工手册 v3.0", "snippet": "员工入职流程包括：HR 面试、背景调查、签订劳动合同...", "score": 0.95},
        {"title": "报销流程指南", "snippet": "差旅报销需在出行后 30 天内提交，附发票和行程单...", "score": 0.87},
    ]
    return json.dumps(sample_results, ensure_ascii=False, indent=2)


def call_external_api(url: str, method: str = "GET", body: str = "") -> str:
    """调用外部 API 接口，发送 HTTP 请求并返回响应。

    Args:
        url: 目标 API 地址
        method: HTTP 方法（GET/POST/PUT/DELETE）
        body: 请求体（JSON 字符串）

    Returns:
        API 响应内容
    """
    # 实际项目中这里会调用 requests 发送 HTTP 请求
    # 这里用模拟数据演示
    return '{"status": "success", "data": {"code": 200, "message": "OK"}}'


# 工具注册表（类似 Java 的 Map<String, Function>）
TOOLS = {
    "query_database": query_database,
    "search_documents": search_documents,
    "call_external_api": call_external_api,
}

# 工具定义（JSON Schema，告诉 LLM 每个工具怎么用）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "查询企业数据库，执行 SQL 语句并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL 查询语句",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在企业知识库中搜索相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_external_api",
            "description": "调用外部 API 接口",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "API 地址",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "description": "HTTP 方法",
                    },
                    "body": {
                        "type": "string",
                        "description": "请求体（JSON 字符串）",
                    },
                },
                "required": ["url", "method"],
            },
        },
    },
]


# ── 2. Agent 主循环 ──
def agent_loop(user_query: str, max_turns: int = 10) -> str:
    """
    Agent 主循环：模型决定调用哪个函数 → 执行工具 → 获取结果 → 继续推理

    类比 Java：就像 Spring 的 DispatcherServlet，用户请求进来后，
    根据 URL 路由到不同的 Controller 处理，再把结果返回给用户。

    Args:
        user_query: 用户的问题
        max_turns: 最大轮次（防止无限循环）

    Returns:
        Agent 的最终回答
    """
    client = OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_URL,
    )

    # 初始消息
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个企业智能助手，可以查询数据库、搜索文档和调用外部 API。"
                "当用户的问题需要你调用工具时，请使用 tools 参数指定的函数。"
                "如果不需要调用工具，直接回答用户的问题。"
            ),
        },
        {"role": "user", "content": user_query},
    ]

    for turn in range(max_turns):
        print(f"\n  [第 {turn + 1} 轮]")

        # 调用 LLM
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )

        msg = response.choices[0].message

        # 如果没有 tool_calls，说明 LLM 决定直接回答
        if not msg.tool_calls:
            print(f"    LLM 直接回答: {msg.content[:100]}...")
            return msg.content

        # 有 tool_calls，说明 LLM 决定调用工具
        print(f"    LLM 决定调用 {len(msg.tool_calls)} 个工具")
        messages.append(msg)  # 把 LLM 的消息加入历史

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id

            print(f"    调用: {func_name}({func_args})")

            # 执行工具
            if func_name in TOOLS:
                try:
                    result = TOOLS[func_name](**func_args)
                    print(f"    结果: {result[:100]}...")
                except Exception as e:
                    result = f"工具执行失败: {e}"
                    print(f"    失败: {e}")
            else:
                result = f"未知工具: {func_name}"
                print(f"    未知工具: {func_name}")

            # 把工具结果加回消息历史
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
            )

    return "Agent 达到最大轮次，未能完成。"


# ── 3. 测试 ──
print("\n[1] 测试查询数据库...")
result = agent_loop("帮我查一下数据库里有哪些用户？")
print(f"\n  最终回答: {result[:300]}...")

print("\n[2] 测试搜索文档...")
result = agent_loop("帮我搜索一下报销流程相关的文档")
print(f"\n  最终回答: {result[:300]}...")

print("\n[3] 测试多工具调用...")
result = agent_loop("查询用户张三的信息，然后搜索他相关的订单")
print(f"\n  最终回答: {result[:300]}...")

print("\n" + "=" * 60)
print("Step 10 完成 — 下一步: step11_langchain_agent.py（LangChain Agent）")
print("=" * 60)
