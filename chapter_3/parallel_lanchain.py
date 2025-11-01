import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()
llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

#定义独立的链条
#三个链条分别执行不同的任务，可以并行地运行

summarize_chain:Runnable = (
    ChatPromptTemplate.from_messages([
        ("system","请简明扼要地总结以下主题"),
        ("user","{topic}")
    ])
    | llm
    | StrOutputParser()
)

question_chain:Runnable = (
    ChatPromptTemplate.from_messages([
        ("system","请针对以下主题生成三个有趣的问题："),
        ("user","{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain:Runnable = (
    ChatPromptTemplate.from_messages([
        ("system","请从以下主题中提取 5-10 个关键词，用逗号分隔："),
        ("user","{topic}")
    ])
    | llm
    | StrOutputParser()
)

# 构建并行 + 汇总链
# 1.定义并行任务快，结果和原始的tpoic一起传递到下一步
# 核心：通过RunnableParallel定义并行任务，这样执行的时候会并行执行每个任务，然后结果和原始的topic一起传递到下一步
map_chain = RunnableParallel(
    {
        "summary":summarize_chain,
        "questions":question_chain,
        "key_terms":terms_chain,
        "topic":RunnablePassthrough(),#传递给原始topic
    }
)

#2.定义最终汇总的prompt，整合并行结果

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system","""根据以下信息：摘要：{summary},相关问题：{questions},关键词：{key_terms}
    请综合生成完整答案。"""),
    ("user","原始主题：{topic}")
])

# 3.构建完整的链条，将并行结果直接传递给汇总prompt，在通过llm和输出解析器处理

full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

#运行链
async def run_parallel_example(topic:str) -> None:
    """ 
    异步调用并行处理链，输出综合的结果
    topic：输入的参数
    """
    print(f"\n --并行 Langchain 示例，主题:{topic}--")

    try:
        response = await full_parallel_chain.ainvoke(topic)
        print("\n -- 最终响应 --")
        print(response)
    except Exception as e:
        print(f"\n链执行出错:{e}")

test_topic = "太空探索的历史"
asyncio.run(run_parallel_example(test_topic))
