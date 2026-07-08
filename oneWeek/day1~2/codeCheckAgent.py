# 调用deepseekAPI 检查代码
import os
import json
from openai import OpenAI



if __name__ == "__main__":
    
    # client = OpenAI(
    #     api_key="XXX",
    #     base_url="https://api.deepseek.com"
    #     )
    # 从配置文件读取 API 参数
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    with open(r"D:\AI\aiLearning\oneWeek\day1~2\pythonchan.py", "r", encoding="utf-8") as f:
        content = f.read()
    print(type(content))
    userPrompt="审查以下代码： "+ content
    print(type(userPrompt))
    print(userPrompt)
    # DeepSeek API的chat.completions.create方法用于生成聊天式的响应。
    # 非流式响应
    ''' 
    response = client.chat.completions.create(
    model="deepseek-v4-pro",
    
    messages=[
        {"role": "system", "content": "你是一个高级Python代码审查专家，专注于发现潜在bug、性能问题和安全漏洞。"},
        {"role": "user", "content": userPrompt},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
    )

    print(response.choices[0].message.content)
    '''
    # 流式响应
    ''' 
    response = client.chat.completions.create(
    model="deepseek-v4-pro",
    
    messages=[
        {"role": "system", "content": "你是一个高级Python代码审查专家，专注于发现潜在bug、性能问题和安全漏洞。"},
        {"role": "user", "content": userPrompt},
    ],
    stream=True,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
    )
    reasoning_content = ""
    content = ""

    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content or ""
        else:
            if delta.content is not None:
                content += delta.content or ""

    

    print(reasoning_content)
    print("====content===",content)
    ''' 
    # agnes API 
    # 参数	类型	必填	说明
# model	string	是	模型名称，使用 agnes-2.0-flash。
# messages	array	是	对话消息数组，包含 system、user 和 assistant 消息。
# messages[].content	string / array	是	可为纯文本，也可为包含 text 和 image_url 的内容块数组。
# temperature	number	否	控制输出随机性。值越低，结果越确定。
# top_p	number	否	控制核采样。值越低，输出越聚焦。
# max_tokens	number	否	响应中生成的最大 token 数量。
# stream	boolean	否	是否启用流式输出。
# tools	array	否	工具调用工作流的工具定义。
# tool_choice	string / object	否	控制模型是否使用工具以及如何使用工具。
# chat_template_kwargs	object	否	OpenAI 兼容请求中启用 Thinking 等扩展能力。
# thinking	object	否	Anthropic 兼容请求中启用 Thinking 模式。

    response = client.chat.completions.create(
    model="agnes-2.0-flash",
    
    messages=[
        {"role": "system", "content": "你是一个高级Python代码审查专家，专注于发现潜在bug、性能问题和安全漏洞。"},
        {"role": "user", "content": userPrompt},
    ]
    )
    print(response.choices[0].message.content)


