""" 
使用langchain实现迭代反思，不断优化代码的效果
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage

import os
load_dotenv()
llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

def run_reflection_loop():
    """
    演示多部AI反思循环，逐步优化python函数
    """
    task_prompt = """
    你的任务是创建一个名为 `calculate_factorial` 的 Python 函数。
    该函数需满足以下要求：
    1. 只接受一个整数参数 n。
    2. 计算其阶乘（n!）。
    3. 包含清晰的 docstring，说明函数功能。
    4. 处理边界情况：0 的阶乘为 1。
    5. 处理无效输入：若输入为负数则抛出 ValueError。
    """

    # 反思循环，优化生成效果
    max_iterations = 3
    current_code = " "
    #构建历史对话，为每个步骤提供上下文
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print(f"\n -- 第 {i+1} 轮反思 --")

        # 1.生成代码，以及优化阶段
        if i == 0:
            print("\n -- 生成初始代码 --")
            response = llm.invoke(message_history)
            current_code = response.content
            print(f"\n 初始代码:\n{current_code}")
        else:
            print("\n -- 生成改进代码 --")
            #使用以往的历史消息、上次的代码，还有对应的修改意见
            message_history.append(HumanMessage(content="请根据批判意见优化代码"))
            response = llm.invoke(message_history)
            current_code = response.content

        print("\n‑‑‑ 第{i+1}轮生成的代码：‑‑‑\n" + current_code)
        message_history.append(response)

        # 反思阶段
        print("\n -- 对生成的代码进行反思 --")
        reflector_prompt = [
            SystemMessage(content="""
                你是一名资深软件工程师，精通 Python。
                你的职责是对提供的 Python 代码进行细致代码审查。
                请根据原始任务要求，严格评估代码。
                检查是否有 bug、风格问题、遗漏边界情况及其他可改进之处。
                若代码完美且满足所有要求，仅回复 'CODE_IS_PERFECT'。
                否则，请以项目符号列表形式给出批判意见。
            """),
            HumanMessage(content = f"原始任务：\n{task_prompt}\n\n待审查的代码：\n{current_code}"),
        ]
        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # 停止条件
        if "CODE_IS_PERFECT" in critique:
            print("\n -- 代码完美，无需改进 --")
            break
        print("\n -- 批判意见： --\n" + critique)
        message_history.append(HumanMessage(content=f"上次代码批判意见：\n{critique}"))

    print("\n -- 最终代码： --\n" + current_code)

if __name__ == "__main__":
    run_reflection_loop()




