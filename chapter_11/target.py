"""
实战代码示例‑ 迭代 2
‑以 LangChain 和 OpenAI API 展示目标设定与监控模式：

目标：构建一个 AI 智能体，能根据指定目标为用例编写代码：
‑ 接收编程问题（用例）作为输入。
‑ 接收目标列表（如“简单”、“已测试”、“处理边界情况”）作为输入。
‑ 使用 LLM（如 GPT‑4o）生成并优化 Python 代码，直到目标达成（最多 5 次迭代，可自定义）。
‑ 判断目标是否达成时，LLM 仅返回 True 或 False，便于停止迭代。
‑ 最终将代码保存为 .py 文件，文件名简洁，带头部注释。

"""

import os
from dotenv import load_dotenv
import random
import re
from pathlib import Path
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="glm-4.6",
    temperature=0.6,
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

def generate_prompt(use_case:str,goals:list[str],previous_code:str="",feedback="")->str:
    print("构建提示词")
    base_prompt = f"""
    你是AI代码生成智能体。你的工作是基于用户的输入，写出相对应Python代码。
    用户的输入：{use_case}
    你的目标是：
    {chr(10).join(f"-{g.strip()}"for g in goals)}
    """

    if previous_code:
        print("将之前的代码添加到prompt用来优化")
        base_prompt += f"\n 之前生成的代码：\n{previous_code}"

    if feedback:
        print("将反馈添加到prompt用来优化")
        base_prompt += f"\n 反馈：\n{feedback}"
        base_prompt += "\n 请仅提交修订后的Python代码。请勿包含代码外的注释或说明。"

    return base_prompt

def get_code_feedback(code:str,goals:list[str])->str:
    print("评估代码是否符合目标")
    feedback_prompt = f"""
    您是一名Python代码审查员。下方展示一段代码片段。根据以下目标：
    {chr(10).join(f"-{g.strip()}"for g in goals)}
    请对这段代码进行评审，并指出是否达到了预期目标。如需改进，请说明在以下方面需要优化：
    清晰度、简洁性、正确性、边界情况处理或测试覆盖率。
    代码:
    {code}
    """
    return llm.invoke(feedback_prompt)

def goals_met(feedback_text:str,goals:list[str]) ->bool:
    """ 
    根据反馈的文本，使用大模型去评估是否完成了目标。
    返回 True 或 False
    """
    review_prompt = f"""
    你是一名AI Python代码审查员,下面是用户的目标：
    {chr(10).join(f"-{g.strip()}"for g in goals)}
    下面是代码审查员对代码的反馈：
    {feedback_text}
    请根据反馈，判断是否完成了目标。
    返回 true 或 false,不要包含任何其他内容。
    """
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == "true"

def clean_code_block(code:str)->str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().endswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def add_comment_header(code:str,use_case:str)->str:
    comment = f"该段代码实现了用户下面的用例\n {use_case.strip()}\n"
    return comment + "\n" + code

def to_snake_case(text:str)->str:
    """
    这个函数用于生成文件名，将用户输入的用例描述转换为符合 Python 命名规范的蛇形命名格式，
    确保生成的文件名：
    只包含字母、数字和下划线
    - 全小写
    - 符合 Python 变量命名规范
    - 适合作为文件名使用
    """
    text = re.sub(r"[^a‑zA‑Z0‑9]","", text)
    return re.sub(r"\s+", "_", text.strip().lower())

def save_code_to_file(code:str,use_case:str)->str:
    print("保存代码文件中...")

    summary_prompt = (
        f"将下列用例概括为单个小写单词或短语，不超过10个字符，适用于Python文件名。\n\n{use_case}"
    )

    raw_summary = llm.invoke(summary_prompt).content.strip()
    short_name = re.sub(r"[^a‑zA‑Z0‑9_]","", raw_summary.replace(" ", "_").lower())[:10]

    random_suffix = str(random.randint(1000, 9999))

    filename = f"{short_name}_{random_suffix}.py"
    filepath = Path.cwd() / filename
    with open(filepath,"w") as f:
        f.write(code)
    print(f"代码已保存到 {filepath}")
    return str(filepath)


def run_code_agent(use_case:str,goals_input:str,max_iterations:int=3)->str:
    goals = [g.strip() for g in goals_input.split(",")]
    print(f"\n 目标：{use_case}")
    print("目标列表：")
    for g in goals:
        print(f" - {g}")

    previous_code = ""
    feedback = ""

    for i in range(max_iterations):
        print(f"\n === 第{i+1}轮迭代 ===\n")
        """ 
        在代码中，feedback 可能来自两个来源：
        - 直接字符串：人工输入的反馈
        - LLM响应对象：从 get_code_feedback() 函数返回的AI响应对象
        通过 isinstance(feedback,str) 检查，确保无论哪种情况都能正确提取文本内容，避免类型错误。
        """
        prompt = generate_prompt(use_case,goals,previous_code,feedback if isinstance(feedback,str) else feedback.content)
        
        print("\n 生成代码中...")
        code_response = llm.invoke(prompt)

        raw_code = code_response.content.strip()
        code = clean_code_block(raw_code)
        print("\n 生成的代码:\n" + code)

        print("\n 评估代码是否符合目标...")
        feedback = get_code_feedback(code,goals)  
        feedback_text = feedback.content.strip()
        print("\n 反馈:\n" + feedback_text)

        if goals_met(feedback_text,goals):
            print("\n 目标达成，代码生成完成")
            break

        print("\n 目标没有达到，继续迭代...")
        previous_code = code
    
    final_code = add_comment_header(code,use_case)
    return save_code_to_file(final_code,use_case)

if __name__ == "__main__":
    print("欢迎使用AI代码生成智能体")

    #example 1
    use_case_input = "编写代码计算给定正整数的二进制间隙。"
    goals_input = "代码需简洁易懂、功能正确、能处理全面的边界情况，仅接受正整数输入，并输出结果及若干示例"
    run_code_agent(use_case_input,goals_input)