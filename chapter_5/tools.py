"""
使用langchain实现工具调用，实现更复杂的任务
案例代码里没有把工具调用的结果插入到提示词中，一块送给llm
这里只是对比了下两种方式的输出
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
#from langchain.agents import create_agent

import os, getpass
import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv
import logging
load_dotenv()

llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

#定义工具
@langchain_tool
def search_information(query:str) -> str:
    """
    根据主题提供事实信息，用于回答问题
    """
    print(f"\n --tools:工具调用：search_information，查询主题：{query}")

    #使用预设的结果模拟输出，用来对比实验搜索工具之后的输出
    
    stimulated_result = {
        "伦敦天气": "伦敦当前天气多云，气温 15°C。",
        "法国首都": "法国的首都是巴黎。",
        "世界人口": "地球人口约 80 亿。",
        "世界最高的山":"珠穆朗玛峰是海最高的山峰。",
        "当前时间":"现在是2025年10月21日19.34分",
        "default":f"模拟搜索'{query}'：未找到具体信息，但该主题很有趣。"
    }
    
    """
    这行代码的工作流程：
    query.lower(): 将输入查询转换为小写，确保大小写不敏感
    .get()方法: 在字典中查找匹配的键
    default值: 如果找不到匹配项，返回 "default" 键对应的值
    """

    #result = stimulated_result.get(query.lower(),stimulated_result["default"])
    result = stimulated_result.get(query.lower())
    print(f"\n --tools:工具调用结果：{result}")
    return result

tools = [search_information]

agent_prompt = ChatPromptTemplate.from_messages([
     ("system","你是一个乐于助人的助手。"),
     ("human","{input}"),
     ("placeholder","{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)


#agent = create_agent(llm,tools=tools,system_prompt="你是一个乐于助人的助手。")

async def run_agent_with_tools(query: str):
    """用 Agent 执行查询并打印最终回复。"""
    print(f"\n‑‑‑ :emoji: Agent 运行查询：'{query}' ‑‑‑")
    try:
        response = await agent_executor.ainvoke({"input": query})
        print("\n‑‑‑ ✅ Agent 最终回复‑‑‑")
        print(response["output"])
    except Exception as e:
        print(f"\n:emoji: Agent 执行出错：{e}")


async def main():
    """并发运行多个Agent查询"""
    tasks = [
        run_agent_with_tools("伦敦天气如何？"),
        run_agent_with_tools("法国首都是哪里？"),
        run_agent_with_tools("地球人口是多少？"),
        run_agent_with_tools("世界最高的山是哪一座?"),
        run_agent_with_tools("现在几点了？")
    ]

    await asyncio.gather(*tasks)

nest_asyncio.apply()
asyncio.run(main())

