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

来源：https://github.com/datawhalechina/wow-agent/tree/main/tutorial/%E7%AC%AC03%E7%AB%A0-openai-agents
