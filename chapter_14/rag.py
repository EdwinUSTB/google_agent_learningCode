"""
使用langchain和graph实现一个简单的RAG
"""

import os
import requests
from typing import List, Dict, Any,TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

#from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain_openai import OpenAIEmbeddings
import dotenv

dotenv.load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xh.v1api.cc/v1/"
)

loader = TextLoader("硕士培养计划.txt")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

client = weaviate.Client(embedded_options=EmbeddedOptions())

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xh.v1api.cc/v1/"
)

vectorstore = Weaviate.from_documents(
    client=client,
    documents=chunks,
    embedding=embeddings,
    by_text=False
)

retriever = vectorstore.as_retriever()

class RAGGraphState(TypedDict):
    question:str
    documents:List[Document]
    generation:str

def retrieve_documents_node(state:RAGGraphState)->RAGGraphState:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents":documents,"question":question,"generation":""}

def generate_response_node(state:RAGGraphState)->RAGGraphState:
    question = state["question"]
    documents = state["documents"]

    template = """你是一个问题回答任务助手，请根据以下问题和相关文档进行回答。
    如果你不知道这个问题，则直接输出你不知道这个问题的答案，请不要胡编乱造答案。
    问题：{question}
    上下文：{context}
    """

    prompt = ChatPromptTemplate.from_template(template)
    context = "\n\n".join([doc.page_content for doc in documents])
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"question":question,"context":context})
    return {"generation":generation,"question":question,"documents":documents}

workflow = StateGraph(RAGGraphState)

workflow.add_node("retrieve_documents",retrieve_documents_node)
workflow.add_node("generate_response",generate_response_node)
workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents","generate_response")
workflow.add_edge("generate_response",END)

app = workflow.compile()

if __name__ == "__main__":
    print("\n Running RAG workflow...")
    query = "北京科技大学硕士培养计划是什么？"
    inputs = {"question":query}
    for s in app.stream(inputs):
        print(s)