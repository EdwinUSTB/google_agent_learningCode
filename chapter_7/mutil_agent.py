import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew,Process
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

"""
使用国内的模型需要调整一下接口
"""

llm = ChatOpenAI(
    temperature=0.1,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

def main():
    """
    初始化并运行内容创作团队
    """

    #定义agent
    researcher = Agent(
        role="研究专家",
        goal="查找并总结AI最新趋势",
        backstory="你是一个经验丰富的研究专家，擅长查找和总结AI领域的最新趋势。",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    writer = Agent(
        role="技术内容写作者",
        goal="根据研究结果撰写清晰易懂的博客",
        backstory="你是一个技术内容写作者，擅长将复杂的技术概念转化为通俗易懂的文字。",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    #定义任务
    research_task = Task(
        description="调研2024-2025年Agent技术领域的发展",
        expected_output = "一份详细的研究报告，包括主要趋势、关键技术、应用案例等",
        agent=researcher,
    )

    write_task = Task(
        description="根据研究报告撰写一篇500字左右的博客",
        expected_output="一篇清晰易懂的博客文章，适合技术爱好者阅读",
        agent=writer,
        context = [research_task],
    )
    
    #创建团队
    blog_creation_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=True,
    )

    #运行团队
    print("use zhipuai llm create bolg")

    try:
        result = blog_creation_crew.kickoff()
        print("\n 🎉 博客创作完成！")
        print("\n 📝 博客内容：")
        print(result)
    except Exception as e:
        print(f"\n ❌ 创作失败：{e}")
        

if __name__ == "__main__":
    main()
