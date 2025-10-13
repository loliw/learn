from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
chat_model = "mistral-small-latest"
base_url="https://api.mistral.ai/v1"
api_key=os.getenv('mistral_key')

llm = LitellmModel(model=chat_model, api_key=api_key, base_url=base_url)

通过修改上述的三个变量即可使用自己的api

之后可以使用下方代码创建多个智能体：
from agents import Agent
agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model=llm,
)

其中，可以加入handoffs来告诉智能体可以调用其他哪几个智能体
加入handoff_description来声明每一个会被调用的智能体的用处，例子如下：
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model=llm,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    model=llm,
)

同时可以添加拦截智能体用来让智能体自己决定拦截不符合规定的输入
from agents import GuardrailFunctionOutput, InputGuardrail, Agent, Runner
from pydantic import BaseModel

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
    model=llm,
)

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

上述代码创造了一个拦截审查智能体，然后用函数包装成输出GuardrailFunctionOutput
之后只需要在参数中加入 input_guardrails，即可进行拦截
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
    model=llm,
)
若是被拦截的话会抛出error，InputGuardrailTripwireTriggered
