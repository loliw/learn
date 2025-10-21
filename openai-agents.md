# Agent 框架使用教程

## 一、初始化模型

```python
from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量中读取 API Key
chat_model = "mistral-small-latest"
base_url = "https://api.mistral.ai/v1"
api_key = os.getenv('mistral_key')

llm = LitellmModel(model=chat_model, api_key=api_key, base_url=base_url)
```

只需修改上方的三个变量（`chat_model`、`base_url`、`api_key`）即可替换为你自己的 API。

---

## 二、创建智能体（Agent）

```python
from agents import Agent

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples.",
    model=llm,
)
```

---

## 三、智能体间协作（handoff）

### 定义 handoff 描述

```python
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples.",
    model=llm,
)
```

### 定义主调度智能体

```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    model=llm,
)
```

---

## 四、输入拦截智能体（Guardrail）

### 定义输出结构

```python
from pydantic import BaseModel

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str
```

### 定义审查 Agent

```python
from agents import GuardrailFunctionOutput, InputGuardrail, Agent, Runner

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
    model=llm,
)
```

### 包装为 Guardrail 函数

```python
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )
```

### 在主智能体中启用 Guardrail

```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
    model=llm,
)
```

若被拦截，会抛出 `InputGuardrailTripwireTriggered` 错误。

---

## 五、定义工具（Tool）

```python
@function_tool 
def get_weather(city: str) -> str: 
    '''
    查询一个城市的天气

    Args:
        city: 要查询天气的城市
    '''
    print(f"天气查询工具被调用了，查询的是 {city}")
    return f"The weather in {city} is sunny."
```

注释部分非常重要，会被用作智能体理解函数功能的提示信息。

---

## 六、定义结构化输出
注意必须要支持openaisdk的api接口，例如ds,qw的官方api没有完全适配，但是用vllm或者ollama本地部署即可支持
不然的话只能通过提示词的方式使用

```python
from pydantic import BaseModel

class Candidate(BaseModel):
    name: str
    experience: str

agent = Agent(
    name="简历助手",
    instructions="根据要求的格式抽取相应的信息。",
    model=llm,
    output_type=Candidate,
)
```

或使用提示词实现严格的 JSON 输出：

```python
class IntakeOutput(BaseModel):
    is_homework: bool
    is_patent: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions=(
        "You are a strict JSON generator. "
        "Decide whether the user's question is about homework and/or patent (IP) consulting. "
        "Output only ONE single-line JSON object with EXACTLY these keys: "
        '{"is_homework": true|false, "is_patent": true|false, "reasoning": "<string>"} '
        "No extra text, no markdown, no code fences."
    ),
    output_type=IntakeOutput,
    model=llm,
)
```

---

## 七、上下文传递（RunContextWrapper）

### 定义上下文

```python
from dataclasses import dataclass

@dataclass
class UserInfo:
    name: str
    uid: int
```

### 工具读取上下文

```python
@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
    '''
    获取当前用户的年龄信息。
    '''
    return f"User {wrapper.context.name} is 47 years old"
```

---

## 八、动态提示词（Dynamic Instructions）

### 定义动态上下文

```python
@dataclass
class PatInfo:
    ip_type: str

    def advice(self) -> str:
        if self.ip_type == "发明":
            return "重点为用户讲解发明专利申请的实质审查要求"
        elif self.ip_type == "实用新型":
            return "重点为用户讲解实用新型专利申请的形式审查要求"
```

### 定义动态提示词函数

```python
def dynamic_instructions(
    context: RunContextWrapper[PatInfo], agent: Agent[PatInfo]
) -> str:
    return f"用汉赋的句式{context.context.advice()}."
```

### 应用到智能体

```python
async def main():
    ip_info = PatInfo(ip_type="实用新型")

    agent = Agent[PatInfo](
        name="Assistant",
        instructions=dynamic_instructions,
        model=llm,
    )
```

---

## 九、Hook（生命周期事件）

### Agent 级 Hook

事件包括：
- on_start  
- on_end  
- on_handoff  
- on_tool_start  
- on_tool_end  

```python
class MyAgentHooks(AgentHooks):
    def __init__(self):
        self.event_counter = 0

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"{self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output) -> None:
        self.event_counter += 1
        print(f"{self.event_counter}: Agent {agent.name} ended with output {output}")
```

使用方式：

```python
agent = Agent(
    name="旅行助手",
    model=llm,
    hooks=MyAgentHooks(),
    instructions="You are a helpful assistant."
)
```

---

### Runner 级 Hook

事件包括：
- on_agent_start  
- on_agent_end  
- on_handoff  
- on_tool_start  
- on_tool_end  

```python
class MyRunHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"{self.event_counter}: Agent {agent.name} started")

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output) -> None:
        self.event_counter += 1
        print(f"{self.event_counter}: Agent {agent.name} ended with output {output}")
```

使用方式：

```python
async def main():
    result = await Runner.run(agent, hooks=MyRunHooks(), input="孟子全名叫什么?")
```

---

## 十、流式输出（Streaming）

```python
async def main():
    result = Runner.run_streamed(agent, "给我讲个程序员相亲的笑话")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
```

常见的 `event.type`：
- raw_response_event  
- agent_updated_stream_event  
- run_item_stream_event  

---

## 十一、多轮对话（Multi-turn）

### 消息输入示例

```python
messages = [
    {"role": "user", "content": "孔子的全名叫什么?"},
    {"role": "user", "content": "孟子的全名叫什么?"},
]

result = await Runner.run(agent, input=messages)
```

### 使用 `to_input_list()` 实现上下文延续

```python
# 第一轮
result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
print(result.final_output)  # San Francisco

# 第二轮
new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
result = await Runner.run(agent, new_input)
print(result.final_output)
```

---

## 十二、终端交互 Demo

```python
async def main():
    await run_demo_loop(agent)
```

可在终端进行交互式对话。

---

## 十二、工具调用 Demo
有三类工具：
1.托管工具（只能openai模型）

2.函数调用（python函数）

3.Agent作为工具：
当不想让子agent接管控制权，比如严格的流水线，或者任务输入输出清晰：出题、改写、归纳、抽取字段、生成材料清单、审校一段文本等。
需要并发，并且信息完备，不需要构思任务

若是想将一个agent转为tool可以使用as_tool，如下所示：
```python
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
    model=llm,
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
    model=llm,
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    model=llm,
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)
```

可以像上面提到过的那样用@function_tool创建函数调用工具，然后通过@function_tool(failure_error_function=custom_error_handler)来规定当这个工具处理报错了如何去处理，类似于这样：
```python
def custom_error_handler(e: Exception) -> str:
    return f"⚠️ 工具执行失败: {type(e).__name__} - {str(e)}。请检查输入参数格式。"

@function_tool(failure_error_function=custom_error_handler)
async def divide(wrapper, a: int, b: int) -> str:
    return str(a / b)
```

## 十二、mcp
mcp有三种server
1.stdio
2.sse
3.StreamableHttp

如下可以创建一个sse的mcp server，使用@mcp.tool()
```python
import random
from mcp.server.fastmcp import FastMCP
# Create server
mcp = FastMCP("Secret Word")
@mcp.tool()
def get_secret_word() -> str:
    print("使用工具 get_secret_word()")
    return random.choice(["apple", "banana", "cherry"])
if __name__ == "__main__":
    mcp.run(transport="sse")
```

之后通过运行这段代码即可将服务部署在端口上
之后可以用MCPServerStdio、MCPServerSse和MCPServerStreamableHttp来连接上述三种server，下面是链接sse的代码
```python
async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to answer the questions.",
        model=llm,
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="required"),
    )

    # Run the `get_secret_word` tool
    message = "What's the secret word?"
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


async def main():
    async with MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://127.0.0.1:8000/sse",
        },
    ) as server:
        await run(server)


if __name__ == "__main__":
    asyncio.run(main())
```
第三方的mcp可以从下面两个网站找：
https://mcpmarket.cn/ 
https://www.modelscope.cn/mcp 
然后申请到的第三方mcp可以用下列代码进行使用
```python
async def main():
    async with MCPServerSse(
        params={
            "url": "<填自己申请的MCP服务的SSE地址>",
        }
    ) as my_mcp_server: # 建议用这种上下文写法，否则需要手动连接和关闭MCP服务。
        agent = Agent(
            name="Assistant",
            instructions="你是一个火车票查询助手，能够查询火车票信息。",
            mcp_servers=[my_mcp_server],
            model_settings=ModelSettings(tool_choice="required"),
            model=llm,
        )

        message = "明天从广州到杭州可以买哪些火车票？"
        print(f"Running: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)
```

如果想用本地的开源第三方的mcp的话就这样
```python
async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/学习资料"],
    }
) as my_mcp_server:
```

## 十三、移交
把一个任务转交给另一个智能体就是移交。底层是以工具的形式实现的，如果向名为Refund Agent智能体的转交，对应工具将被命名为transfer_to_refund_agent
除了handoffs参数，也可以用handoff()函数自定义移交
最基础的就是
```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    model=llm,
)
```

用handoff()的话就是
```python
def on_handoff(ctx: RunContextWrapper[None]):
    print("Handoff called")
handoff_obj = handoff(
    agent=math_tutor_agent,
    on_handoff=on_handoff,
    tool_name_override="custom_handoff_tool",
    tool_description_override="Custom description",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, handoff_obj],
    model=llm,
)
```
这里参数
agent是要移交的agent
on_handoff是出发移交时执行的函数
tool_name_override是覆盖默认的移交工具名称，因为系统默认的是transfer_to_<agent_name>不直观也不好理解
tool_description_override是覆盖默认工具描述 ，因为系统默认的是Transfer control to agent <agent_name>，模型不好理解
input_type可以指定接受的输入类型
input_filter用来过滤后续智能体接收的输入，因为在移交的时候，目标智能体接受的是上一个智能体完整的全部输出，但是有时候并不需要很多无关信息，所以可以像这样自己写，或者调用一些内置好的方法
```python
def my_input_filter(data: HandoffInputData) -> HandoffInputData:
    # 只保留最后三条用户消息
    new_history = [m for m in data.history[-3:] if m.role == "user"]
    data.history = new_history
    return data
```
又比如
```python
handoff_obj = handoff(
    agent=math_tutor_agent,
    input_filter=handoff_filters.remove_all_tools,
)
```

为了更好的移交，建议使用前缀模板加入到提示词中，如下
```python
triage_prompt = prompt_with_handoff_instructions(
    base_prompt="你是一个负责问题分诊的智能体，根据问题内容决定是否转交专家。",
    handoffs=[handoff_math, handoff_history],
)

# 创建 Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions=triage_prompt,
    handoffs=[handoff_math, handoff_history],
    model=llm,
)
```

又或者
```python
billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    <Fill in the rest of your prompt here>.""",
)
```

## 十四、追踪
追踪功能默认启用。如需禁用可通过以下两种方式：
1、通过设置环境变量 OPENAI_AGENTS_DISABLE_TRACING=1 全局关闭追踪功能 2、针对单个程序文件禁用追踪：将agents.set_tracing_disabled 中的 disabled 参数设为True 3、针对单次运行禁用追踪：将 agents.run.RunConfig.tracing_disabled 设为 True

Traces（追踪记录） 表示一个完整"工作流"的端到端操作，由多个 Span 组成。追踪记录包含以下属性：
workflow_name：表示逻辑工作流或应用名称，例如"代码生成"或"客户服务"
trace_id：追踪记录的唯一标识符。若未提供将自动生成，必须符合 trace_<32_alphanumeric> 格式
group_id：可选分组 ID，用于关联同一会话中的多个追踪记录（例如聊天线程 ID）
disabled：若为 True，则该追踪记录不会被保存
metadata：追踪记录的元数据（可选）
Spans（跨度） 表示具有起止时间的操作单元。跨度包含：
started_at 和 ended_at 时间戳
所属追踪记录的 trace_id
指向父级跨度的 parent_id（如存在）
记录跨度详情的 span_data。例如 AgentSpanData 包含智能体信息，GenerationSpanData 包含大模型生成信息等

当run的时候追踪就会被记录到trace() 中
agent_span()记录每次智能体运行
generation_span()记录大模型生成内容
function_span()记录函数工具调用
guardrail_span()记录防护机制触发
handoff_span()记录移交智能体
transcription_span()记录语音转文字
speech_span()记录文字转语音

可以像这样把多条run放到同一个trace
```python
async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"): 
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
```

可以用set_trace_processors()替换掉默认的追踪处理器，比如说换成langsmith
```python
from langsmith.wrappers import OpenAIAgentsTracingProcessor
set_trace_processors([OpenAIAgentsTracingProcessor()])
```
来源：https://github.com/datawhalechina/wow-agent/tree/main/tutorial/%E7%AC%AC03%E7%AB%A0-openai-agents
