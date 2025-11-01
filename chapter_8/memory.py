"""
短期记忆的实现
"""
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="glm-4.6",
    temperature=0.6,
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("你是一个友好的助手，总是以积极的态度回答用户的问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

response = conversation.predict(question="你好，我是小明，很高兴认识你。")
print(response)

response = conversation.predict(question="你叫什么名字？")
print(response)

