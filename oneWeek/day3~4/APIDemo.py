# 调用API DEMO
import os
import json
from openai import OpenAI



def helloword(client):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "user", "content": "你好，介绍一下你自己"}
        ]
    )
    # print("responseType:", type(response)) # responseType: <class 'openai.types.chat.chat_completion.ChatCompletion'>
    # print("response:", response)
    print(response.choices[0].message.content)

def interactive_math_tutor(client):
    messages = [
        {"role": "system", "content": "你是一个数学老师"},
    ]

    # 第一轮
    messages.append({"role": "user", "content": "1+1等于几？"})
    resp1 = client.chat.completions.create(model="deepseek-v4-pro", messages=messages)
    print("resp1:", resp1.choices[0].message.content)
    ans_msg = {
        "role": resp1.choices[0].message.role, # assistant
        "content": resp1.choices[0].message.content # 模型的回复
    }
    messages.append(ans_msg)  # 把模型的回复加入历史

    # 第二轮
    messages.append({"role": "user", "content": "再加上3是多少？"})
    resp2 = client.chat.completions.create(model="deepseek-v4-pro", messages=messages)
    print("resp2:", resp2.choices[0].message.content)

def temperatureTest(client):
    # 低温度：严谨、确定性强
    response_low = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "写一首关于春天的诗"}],
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.1   # 几乎每次输出都一样
    )
    print("response_low:", response_low.choices[0].message.content)

    # 高温度：创意、多样性强
    response_high = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "写一首关于春天的诗"}],
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.9   # 每次输出都可能不一样
    )
    print("response_high:", response_high.choices[0].message.content)

def top_pTest(client):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "明天天气怎么样？"}],
        top_p=0.9   # 只从累计概率90%的词里面选，top_p 设得特别小（远小于最高单条概率），效果等价：每次只强制选概率最大的唯一一个词；
    )
    print("response:", response.choices[0].message.content)

def max_tokensTest(client):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "请详细介绍RAG的原理"}],
        max_tokens=2048  # 最多输出2048个token
    )
    # tokens 使用量
    print("tokens:", response.usage.total_tokens)
    # 输出结束的原因 被截断: length 正常结束:stop
    print("finish_reason:", response.choices[0].finish_reason)

    print("response:", response.choices[0].message.content)

def overallTest(client):
    response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "你是一个资深的Java转Python的技术导师，擅长用Java类比讲解Python概念。"},
        {"role": "user", "content": "请用Java的概念帮我理解Python的装饰器"}
    ],
    temperature=0.3,      # 偏严谨，因为要准确解释概念
    top_p=0.9,            # 允许一定的表达多样性
    max_tokens=1024,       # 足够长的回答
    stream=False           # 非流式，一次性返回
    )

    print(response.choices[0].message.content)
    print(f"消耗tokens: {response.usage.total_tokens}")

def few_showTest(client):
    response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "你是一个文本情感分类器，请将用户输入分为正面、负面或中性。"},
        # 示例1
        {"role": "user", "content": "这个产品真好用！"},
        {"role": "assistant", "content": "正面"},
        # 示例2
        {"role": "user", "content": "物流太慢了，等了两周。"},
        {"role": "assistant", "content": "负面"},
        # 示例3
        {"role": "user", "content": "商品收到了。"},
        {"role": "assistant", "content": "中性"},
        # 真正要分类的内容
        {"role": "user", "content": "客服态度特别好，解决问题很快！"},
    ],
    temperature=0.1,  # 分类任务要确定性高
    max_tokens=10,
    extra_body={"thinking": {"type": "disabled"}} # deepseek思考模式开关(1)	{"thinking": {"type": "enabled/disabled"}}
    )

    # 输出: 正面
    print(response.choices[0].message.content)

def jsonInputTest(client):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": """你是一个信息提取助手。请从用户输入中提取姓名、年龄、城市，并以JSON格式返回。
    只返回JSON，不要其他内容。格式如下：
    {"name": "姓名", "age": 年龄数字, "city": "城市"}"""},
            
            # 示例1
            {"role": "user", "content": "我叫张三，今年28岁，住在北京。"},
            {"role": "assistant", "content": "{\"name\": \"张三\", \"age\": 28, \"city\": \"北京\"}"},
            
            # 示例2
            {"role": "user", "content": "李四，35岁，上海。"},
            {"role": "assistant", "content": "{\"name\": \"李四\", \"age\": 35, \"city\": \"上海\"}"},
            
            # 真正要处理的
            {"role": "user", "content": "王芳，25岁，广州人。"},
        ],
        temperature=0.0,  # 结构化输出要零随机
        max_tokens=100,
        extra_body={"thinking": {"type": "disabled"}}
    )
    result = json.loads(response.choices[0].message.content)
    print(type(result),result,sep = "|")
    # {'name': '王芳', 'age': 25, 'city': '广州'}

if __name__ == "__main__":
    
    # 从配置文件读取 API 参数
    with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    client = OpenAI(
        api_key=config["deepseek"]["api_key"],
        base_url=config["deepseek"]["base_url"]
    )

    # 发送请求
    # response = helloword(client)
    
    # 多轮对话：消息是怎么累积的
    # interactive_math_tutor(client)
    

    # temperature 参数 测试 
    # temperatureTest(client)

    # Top-p — 核采样阈值
    # top_pTest(client)

    # Max Tokens — 最大输出长度
    # max_tokensTest(client)

    # 综合调优示例
    # overallTest(client)

    # Few-shot
    # few_showTest(client)

    # 结构化示例
    jsonInputTest(client)





