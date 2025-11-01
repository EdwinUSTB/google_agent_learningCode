
import os
import requests
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xh.v1api.cc/v1/"
)

print(llm.invoke("你是谁？"))
