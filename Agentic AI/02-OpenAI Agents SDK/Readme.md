# OpenAI Agents SDK

## Table of Contents
1. [Introduction to Asynchronous Python (AsyncIO)](#introduction-to-asynchronous-python-asyncio)
2. [OpenAI Agents SDK Overview](#openai-agents-sdk-overview)
3. [Core Concepts and Terminology](#core-concepts-and-terminology)
4. [Implementation Walkthrough](#implementation-walkthrough)
5. [Code Analysis and Optimization](#code-analysis-and-optimization)
6. [Vibe Coding Best Practices](#vibe-coding-best-practices)
7. [Monitoring and Tracing](#monitoring-and-tracing)

## Introduction to Asynchronous Python (AsyncIO)

### Why AsyncIO Matters for Agent Frameworks

Before diving into OpenAI Agents SDK, understanding asynchronous Python is crucial. All modern agent frameworks utilize AsyncIO because:

- **Lightweight Concurrency**: AsyncIO provides a lightweight alternative to multithreading/multiprocessing
- **I/O Bound Operations**: LLM API calls involve significant network waiting time
- **Resource Efficiency**: Thousands of concurrent operations without heavy resource consumption
- **Framework Ubiquity**: All agent frameworks (OpenAI Agents SDK, CrewAI, LangGraph, AutoGen) use AsyncIO

### AsyncIO Fundamentals

#### The Short Version
```python
# Regular function becomes async coroutine
async def do_some_processing():
    # Some work here
    return "done"

# Must use await to call async functions
result = await do_some_processing()
```

#### The Complete Picture

**Key Concepts:**
- **Coroutine**: An `async def` function that can be paused and resumed
- **Event Loop**: Manages execution of coroutines, switching between them during I/O waits
- **await**: Schedules a coroutine for execution and waits for completion

**How It Works:**
1. When you define `async def function_name()`, you create a coroutine (not a regular function)
2. Calling a coroutine returns a coroutine object but doesn't execute it
3. `await` schedules the coroutine in the event loop for execution
4. The event loop can switch between coroutines when one is waiting for I/O

**Concurrent Execution:**
```python
import asyncio

# Execute multiple coroutines concurrently
results = await asyncio.gather(
    coroutine1(),
    coroutine2(),
    coroutine3()
)
```

This allows all three coroutines to run "simultaneously" - when one waits for I/O, others continue executing.

## OpenAI Agents SDK Overview

### Framework Philosophy

OpenAI Agents SDK is designed with specific principles:

- **Lightweight**: Minimal overhead and simple architecture
- **Non-opinionated**: Flexible approach without rigid design patterns
- **Abstraction of Boilerplate**: Handles JSON tool calling complexity automatically
- **Multi-model Support**: Not limited to OpenAI models despite the name

### Why This Framework Excels

1. **Simplicity**: Reduces complex tool calling to simple function definitions
2. **Flexibility**: Allows custom implementation patterns
3. **Monitoring**: Built-in tracing and observability
4. **Efficiency**: Leverages AsyncIO for optimal performance

## Core Concepts and Terminology

### Three Essential Concepts

1. **Agent**: A package around LLM calls with specific role and purpose
   - Contains system prompt (instructions)
   - Defines model to use
   - Encapsulates specific functionality

2. **Handoffs**: Interactions between agents
   - Mechanism for agent-to-agent communication
   - Enables complex multi-agent workflows

3. **Guardrails**: Checks and controls for agent behavior
   - Input validation
   - Output filtering
   - Behavioral constraints

### Three-Step Execution Pattern

Every agent execution follows this pattern:
1. **Create Agent Instance**: Define role, instructions, and model
2. **Wrap with Trace**: Enable monitoring and logging
3. **Execute with Runner**: Use `Runner.run()` to execute the agent

## Implementation Walkthrough

### Step 1: Environment Setup

```python
from dotenv import load_dotenv
from agents import Agent, Runner, trace

# Load environment variables (API keys, etc.)
load_dotenv(override=True)
```

**Technical Details:**
- `load_dotenv(override=True)` ensures environment variables are refreshed
- Essential for API key management
- `override=True` prevents caching issues during development

### Step 2: Agent Creation

```python
agent = Agent(
    name="Jokester",
    instructions="You are a joke teller",
    model="gpt-4o-mini"
)
```

**Component Analysis:**
- **name**: Identifier for the agent (used in tracing and debugging)
- **instructions**: System prompt that defines agent behavior
- **model**: Specifies which LLM to use (supports multiple providers)

**Model Flexibility:**
Despite the "OpenAI" name, the SDK supports various models:
- OpenAI models (default when just model name provided)
- Other providers through proper configuration
- Custom model endpoints

### Step 3: Execution with Tracing

```python
with trace("Telling a joke"):
    result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)
```

**Execution Flow:**
1. **Trace Context**: Creates monitoring session named "Telling a joke"
2. **Runner.run()**: Executes the agent with the provided prompt
3. **Async Execution**: Must use `await` since `Runner.run()` is a coroutine
4. **Result Handling**: Access output through `result.final_output`

## Code Analysis and Optimization

### Original Implementation Analysis

The lab code demonstrates the minimal viable implementation:

```python
# Basic agent creation
agent = Agent(name="Jokester", instructions="You are a joke teller", model="gpt-4o-mini")

# Simple execution
with trace("Telling a joke"):
    result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)
```

### Optimized Implementation

Here's an improved version with better error handling and structure:

```python
import asyncio
from typing import Optional
from dotenv import load_dotenv
from agents import Agent, Runner, trace

class JokeAgent:
    """Optimized joke-telling agent with error handling and reusability."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.agent = Agent(
            name="Jokester",
            instructions="You are a professional comedian. Create clever, appropriate jokes that are both funny and insightful.",
            model=model
        )
    
    async def tell_joke(self, topic: str, trace_name: Optional[str] = None) -> str:
        """
        Generate a joke about the specified topic.
        
        Args:
            topic: The subject for the joke
            trace_name: Optional name for tracing (defaults to topic-based name)
        
        Returns:
            The generated joke as a string
        """
        if not trace_name:
            trace_name = f"Joke about {topic}"
        
        try:
            with trace(trace_name):
                result = await Runner.run(
                    self.agent, 
                    f"Tell a clever joke about {topic}"
                )
                return result.final_output
        except Exception as e:
            return f"Sorry, I couldn't generate a joke right now. Error: {str(e)}"

# Usage example
async def main():
    load_dotenv(override=True)
    
    joke_agent = JokeAgent()
    joke = await joke_agent.tell_joke("Autonomous AI Agents")
    print(joke)

# For Jupyter notebook execution
if __name__ == "__main__":
    # In Jupyter, just use: await joke_agent.tell_joke("topic")
    asyncio.run(main())
```


## Vibe Coding Best Practices

### The Five Pillars of Effective Vibe Coding

#### 1. Good Vibes - Craft Reusable Prompts
```
Create a prompt template like:
"Generate concise, clean Python code for [specific task]. 
Use current APIs as of [today's date]. 
Avoid verbose implementations and excessive error handling.
Focus on the core functionality."
```

#### 2. Vibe but Verify - Multi-LLM Validation
- Ask the same question to multiple LLMs (ChatGPT, Claude, etc.)
- Compare responses for accuracy and approach
- Choose the most appropriate solution

#### 3. Step Up the Vibe - Incremental Development
```python
# Instead of generating 200 lines at once, break into steps:
# Step 1: Create basic agent structure (10 lines)
# Step 2: Add error handling (10 lines)  
# Step 3: Implement tracing (10 lines)
# Step 4: Add optimization features (10 lines)
```

**Planning Prompt Example:**
```
"I need to build an AI agent system. Instead of code, give me 4-5 simple steps 
where each step is independently testable and under 10 lines of code."
```

#### 4. Vibe and Validate - Cross-Verification
After getting a solution:
```
"I asked for [original question] and got this answer: [solution].
Please review this for:
- Correctness and bugs
- More concise alternatives  
- Better structure or clarity
- Current best practices"
```

#### 5. Vibe with Variety - Multiple Approaches
```
"Give me three different ways to implement [specific functionality].
Explain the rationale for each approach and when to use each one."
```

### Vibe Coding Anti-Patterns to Avoid

❌ **Don't**: Generate large code blocks without understanding

❌ **Don't**: Accept first solution without verification  

❌ **Don't**: Skip incremental testing

❌ **Don't**: Ignore error handling until the end


✅ **Do**: Build incrementally with testing

✅ **Do**: Understand every line of generated code

✅ **Do**: Validate solutions across multiple LLMs

✅ **Do**: Ask for explanations of complex logic


## Monitoring and Tracing

### OpenAI Platform Integration

The `trace()` context manager integrates with OpenAI's monitoring platform:

```python
with trace("Custom Operation Name"):
    # All agent interactions within this block
    # are grouped under the specified trace name
    result = await Runner.run(agent, prompt)
```

### Accessing Traces

1. Navigate to [https://platform.openai.com/traces](https://platform.openai.com/traces)
2. Find your trace by the name provided to `trace()`
3. Examine the complete interaction flow:
   - System instructions (agent's instructions)
   - User prompt
   - Model response
   - Token usage and timing

### Advanced Tracing Patterns

```python
# Hierarchical tracing for complex workflows
async def complex_agent_workflow():
    with trace("Main Workflow"):
        # Sub-operations automatically nested
        with trace("Data Analysis"):
            analysis = await Runner.run(analyst_agent, data_prompt)
        
        with trace("Report Generation"):
            report = await Runner.run(writer_agent, f"Create report: {analysis.final_output}")
        
        return report
```

### Trace Benefits for Development
- **Debugging**: See exact prompts and responses
- **Optimization**: Identify bottlenecks and token usage
- **Monitoring**: Track agent behavior in production
- **Collaboration**: Share interaction logs with team members

### Multi-Agent Patterns

While this code snippets shows single-agent usage, the framework supports:
- Agent handoffs for complex workflows
- Concurrent agent execution with `asyncio.gather()`
- Agent specialization and role-based architectures

OpenAI Agents SDK provides a clean, lightweight foundation for building AI agent systems. Its strength lies in:

1. **Simplicity**: Minimal boilerplate for common patterns
2. **Flexibility**: Non-opinionated architecture allows custom implementations  
3. **AsyncIO Integration**: Efficient handling of I/O-bound operations
4. **Monitoring**: Built-in tracing for observability
5. **Multi-model Support**: Not locked into OpenAI ecosystem

The combination of AsyncIO understanding and the SDK's clean abstractions creates a powerful foundation for building sophisticated agent systems. The key is to start simple (as shown in the lab) and incrementally add complexity while maintaining clean, testable code.
> the goal isn't just to make agents work, but to build maintainable, observable, and scalable agent systems that can evolve with your requirements.

---

# OpenAI Agents SDK: Building Agentic Architectures

## Table of Contents
1. [Three-Layer Agentic Architecture](#three-layer-agentic-architecture)
2. [Agent Workflows vs True Agents](#agent-workflows-vs-true-agents)
3. [Tools: Functions and Agent Wrapping](#tools-functions-and-agent-wrapping)
4. [Handoffs: Agent Delegation Patterns](#handoffs-agent-delegation-patterns)
5. [Complete Implementation Analysis](#complete-implementation-analysis)
6. [Optimized Code Examples](#optimized-code-examples)
7. [Design Patterns and Best Practices](#design-patterns-and-best-practices)

## Three-Layer Agentic Architecture

This outlines a progressive approach to building sophisticated agent systems through three distinct layers:

### Layer 1: Simple Agent Workflows
- Sequential agent calls orchestrated by Python code
- Manual coordination between agents
- Direct control flow management

### Layer 2: Tool-Enhanced Agents
- Agents that can use external functions as tools
- Automatic JSON boilerplate generation
- Function calling abstraction

### Layer 3: Agent Collaboration
- Agents calling other agents as tools
- Handoff mechanisms for delegation
- Complex multi-agent orchestration

## Agent Workflows vs True Agents

### The Critical Distinction

According to Anthropic's definition, there's a fundamental difference between "agent workflows" and true "agents":

**Agent Workflows**: Predetermined sequences where humans define the exact steps and order
**True Agents**: Systems where the agent makes autonomous decisions about what to do and when

### The Pivotal Moment

The transformation from workflow to agent occurs when you give an agent **tools** and let it decide:
- Which tools to use
- When to use them  
- In what order to execute them

This single change - providing tools instead of scripted sequences - represents the shift to true agentic behavior.

## Tools: Functions and Agent Wrapping

### Function Tools: Eliminating JSON Boilerplate

The traditional approach required extensive JSON schemas:

```python
# Old way - manual JSON schema definition
tool_schema = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send email to prospects",
        "parameters": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "Email body"}
            },
            "required": ["body"]
        }
    }
}
```

OpenAI Agents SDK eliminates this complexity:

```python
@function_tool
def send_email(body: str):
    """Send out an email with the given body to all sales prospects"""
    # Implementation here
    return {"status": "success"}
```

**Key Benefits:**
- Automatic JSON schema generation from function signature
- Type hints become parameter specifications
- Docstrings become tool descriptions
- Zero boilerplate code

### Agent-as-Tool Pattern

A revolutionary concept: entire agents can become tools for other agents.

```python
# Create a specialized agent
sales_agent = Agent(
    name="Professional Sales Agent",
    instructions="You write professional, serious cold emails",
    model="gpt-4o-mini"
)

# Convert agent to tool
sales_tool = sales_agent.as_tool(
    tool_name="sales_agent1",
    tool_description="Write a cold sales email"
)
```

**Technical Implementation:**
- The framework creates a wrapper function around the agent
- When the tool is called, it executes the wrapped agent
- Results are returned as if from a regular function
- Full agent capabilities preserved within tool interface

## Handoffs: Agent Delegation Patterns

### Conceptual Framework

**Tools vs Handoffs:**
- **Tools**: Request-response pattern, control returns to caller
- **Handoffs**: Delegation pattern, control passes permanently to recipient

### Technical Differences

```python
# Tool usage - control returns
result = await agent.use_tool("email_writer", "Write email")
# Agent continues processing after tool completes

# Handoff usage - control transfers
await agent.handoff_to("email_manager", email_content)
# Original agent's work is complete
```

### Implementation Pattern

```python
# Agent that can receive handoffs
emailer_agent = Agent(
    name="Email Manager",
    instructions="Format and send emails",
    tools=[subject_tool, html_tool, send_tool],
    handoff_description="Convert an email to HTML and send it"
)

# Agent that can make handoffs
sales_manager = Agent(
    name="Sales Manager", 
    instructions="Generate emails then handoff for sending",
    tools=[sales_tools],
    handoffs=[emailer_agent]
)
```

## Complete Implementation Analysis

### Sales Development Representative System

The example implements a complete SDR (Sales Development Representative) system demonstrating all three architectural layers:

#### Layer 1: Parallel Email Generation

```python
# Optimized parallel execution
async def generate_email_variants(message: str) -> List[str]:
    """Generate multiple email variants concurrently"""
    with trace("Parallel email generation"):
        results = await asyncio.gather(
            Runner.run(sales_agent1, message),
            Runner.run(sales_agent2, message), 
            Runner.run(sales_agent3, message)
        )
    return [result.final_output for result in results]
```

**Key Concepts:**
- `asyncio.gather()` enables true concurrent execution
- Each agent has distinct personality (professional, engaging, concise)
- Trace context groups related operations

#### Layer 2: Tool Integration

```python
# Streamlined tool creation
@function_tool

def  send_email(body:  str):

""" Send out an email with the given body to all sales prospects """

sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

from_email =  Email("kejxxx@gmail.com")  # Change to your verified sender

to_email =  To("mkejxxx@gmail.com")  # Change to your recipient

content =  Content("text/plain", body)

mail =  Mail(from_email, to_email,  "Sales email", content).get()

sg.client.mail.send.post(request_body=mail)

return  {"status":  "success"}
```
---

```python
#alternative to sendGrid_email

@function_tool

def  send_email_bk(body:  str):

""" Send out an email with the given body to all sales prospects via Resend """

# Set up email sender, recipient, and content

from_email =  "onboarding@resend.dev"  # Replace with your verified sender

to_email =  "kejxxx@gmail.com"#os.getenv("GMAIL_TO") " # Replace with recipient's email

# Resend API headers and payload

api_key=os.environ.get('RESEND_API_KEY')

headers =  {

"Authorization":  f"Bearer {api_key}",

"Content-Type":  "application/json"

}

payload =  {

"from":  f"Mohamed kejxxx <{from_email}>",

"to":  [to_email],

"subject":  "Sales email",

"html":  f"<p>{body}</p>"  # Body wrapped in <p> tags for HTML format

}

# Send email using Resend API

response = requests.post("https://api.resend.com/emails",  json=payload,  headers=headers)

# Check if the request was successful

if response.status_code ==  202:

return  {"status":  "success"}

else:

return  {"status":  "failure",  "message": response.text}
```

#### Layer 3: Autonomous Decision Making

```python
# Sales manager with autonomous tool selection
sales_manager = Agent(
    name="Sales Manager",
    instructions="""
    You are a Sales Manager at ComplAI. Follow these steps:
    1. Use all three sales_agent tools to generate drafts
    2. Evaluate and select the single best email
    3. Use send_email tool to send only the best email
    
    Rules:
    - Never write emails yourself - always use tools
    - Send exactly one email
    """,
    tools=email_tools + [send_email],
    model="gpt-4o-mini"
)
```

## Optimized Code Examples

### Minimal Email System Implementation

```python
from agents import Agent, Runner, trace, function_tool
import asyncio
from typing import List, Dict

class EmailSystem:
    """Optimized email generation and sending system"""
    
    def __init__(self):
        self.agents = self._create_agents()
        self.tools = self._create_tools()
        self.manager = self._create_manager()
    
    def _create_agents(self) -> List[Agent]:
        """Create specialized email writing agents"""
        configs = [
            ("Professional", "Write professional, serious cold emails"),
            ("Engaging", "Write witty, engaging cold emails"),
            ("Concise", "Write brief, to-the-point cold emails")
        ]
        
        return [
            Agent(name=f"{name} Sales Agent", instructions=instructions, model="gpt-4o-mini")
            for name, instructions in configs
        ]
    
    def _create_tools(self) -> List:
        """Convert agents to tools and add email function"""
        agent_tools = [
            agent.as_tool(f"agent_{i}", "Generate cold sales email")
            for i, agent in enumerate(self.agents)
        ]
        return agent_tools + [self.send_email]
    
    @function_tool
    def send_email(self, body: str) -> Dict[str, str]:
        """Send email via configured service"""
        # Implementation here
        return {"status": "success"}
    
    def _create_manager(self) -> Agent:
        """Create autonomous sales manager"""
        return Agent(
            name="Sales Manager",
            instructions="Use tools to generate and send best email",
            tools=self.tools,
            model="gpt-4o-mini"
        )
    
    async def process_request(self, message: str) -> str:
        """Process email request autonomously"""
        with trace("Autonomous email processing"):
            result = await Runner.run(self.manager, message)
            return result.final_output

# Usage
system = EmailSystem()
result = await system.process_request("Send cold email to CEO")
```

## Design Patterns and Best Practices

### 1. Agent Specialization Pattern

**Principle**: Create agents with narrow, well-defined responsibilities

```python
# Good: Specialized agents
subject_agent = Agent(name="Subject Writer", instructions="Write email subjects only")
body_agent = Agent(name="Body Writer", instructions="Write email bodies only")

# Avoid: Overly broad agents
general_agent = Agent(name="Email Agent", instructions="Do everything email-related")
```

### 2. Tool Composition Pattern

**Principle**: Build complex capabilities from simple, reusable tools

```python
# Compose complex workflows from simple tools
email_tools = [
    generate_content_tool,
    format_html_tool, 
    add_subject_tool,
    send_email_tool
]

manager = Agent(tools=email_tools, instructions="Use tools to complete email workflow")
```

### 3. Handoff Chain Pattern

**Principle**: Create clear delegation chains for complex processes

```python
# Clear handoff chain: Generator → Formatter → Sender
generator_agent = Agent(handoffs=[formatter_agent])
formatter_agent = Agent(handoffs=[sender_agent]) 
sender_agent = Agent(tools=[send_tools])
```

### 4. Trace Organization Pattern

**Principle**: Use meaningful trace names for debugging and monitoring

```python
# Hierarchical tracing
with trace("Email Campaign"):
    with trace("Content Generation"):
        # Generation logic
    with trace("Quality Review"):
        # Review logic  
    with trace("Delivery"):
        # Sending logic
```


## Advanced Concepts

### Streaming Responses

```python
# Stream agent responses for better UX
result = Runner.run_streamed(agent, message)
async for event in result.stream_events():
    if event.type == "raw_response_event":
        print(event.data.delta, end="", flush=True)
```

### Multi-Model Support

```python
# Use different models for different tasks
fast_agent = Agent(model="gpt-4o-mini")  # For quick tasks
powerful_agent = Agent(model="gpt-4o")   # For complex reasoning
```

### Dynamic Tool Selection

```python
# Agents can choose from available tools based on context
flexible_agent = Agent(
    tools=[tool1, tool2, tool3],
    instructions="Choose the most appropriate tool for each task"
)
```

## Key Takeaways

1. **Progressive Complexity**: Build systems incrementally from simple workflows to autonomous agents
2. **Tool Abstraction**: Leverage framework capabilities to eliminate boilerplate
3. **Agent Specialization**: Create focused agents rather than generalist systems
4. **Delegation Patterns**: Use handoffs for clear responsibility transfer
5. **Monitoring**: Implement comprehensive tracing for debugging and optimization

The OpenAI Agents SDK provides a remarkably clean abstraction for building sophisticated multi-agent systems. The key insight is understanding when to use tools (for capabilities) versus handoffs (for delegation), and how to compose these patterns into robust, autonomous systems.

The progression from simple workflows to true agentic behavior represents a fundamental shift in how we architect AI systems - from predetermined sequences to autonomous decision-making entities that can adapt and respond to complex scenarios.

---

# Complete Code Reference

## Basic Imports and Setup

```python
from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
import asyncio
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
from typing import List, Dict

load_dotenv(override=True)
```

## Traditional vs Modern Tool Definition

```python
# Old way - manual JSON schema definition
tool_schema = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send email to prospects",
        "parameters": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "Email body"}
            },
            "required": ["body"]
        }
    }
}

# Modern way - automatic schema generation
@function_tool
def send_email(body: str):
    """Send out an email with the given body to all sales prospects"""
    return {"status": "success"}
```

## Agent-as-Tool Pattern

```python
# Create a specialized agent
sales_agent = Agent(
    name="Professional Sales Agent",
    instructions="You write professional, serious cold emails",
    model="gpt-4o-mini"
)

# Convert agent to tool
sales_tool = sales_agent.as_tool(
    tool_name="sales_agent1",
    tool_description="Write a cold sales email"
)
```

## Tool vs Handoff Usage Patterns

```python
# Tool usage - control returns
result = await agent.use_tool("email_writer", "Write email")
# Agent continues processing after tool completes

# Handoff usage - control transfers
await agent.handoff_to("email_manager", email_content)
# Original agent's work is complete
```

## Handoff Implementation

```python
# Agent that can receive handoffs
emailer_agent = Agent(
    name="Email Manager",
    instructions="Format and send emails",
    tools=[subject_tool, html_tool, send_tool],
    handoff_description="Convert an email to HTML and send it"
)

# Agent that can make handoffs
sales_manager = Agent(
    name="Sales Manager", 
    instructions="Generate emails then handoff for sending",
    tools=[sales_tools],
    handoffs=[emailer_agent]
)
```

## Parallel Email Generation

```python
async def generate_email_variants(message: str) -> List[str]:
    """Generate multiple email variants concurrently"""
    with trace("Parallel email generation"):
        results = await asyncio.gather(
            Runner.run(sales_agent1, message),
            Runner.run(sales_agent2, message), 
            Runner.run(sales_agent3, message)
        )
    return [result.final_output for result in results]
```

## Tool Integration

```python
@function_tool
def  send_email(body:  str) -> Dict[str, str]:
	""" Send out an email with the given body to all sales prospects """
	sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
	from_email =  Email("kejxx@gmail.com")  # Change to your verified sender
	to_email =  To("mkxx@gmail.com")  # Change to your recipient
	content =  Content("text/plain", body)
	mail =  Mail(from_email, to_email,  "Sales email", content).get()
	sg.client.mail.send.post(request_body=mail)
	return  {"status":  "success"}


# Agent-to-tool conversion
email_tools = [
    agent.as_tool(f"sales_agent{i}", "Write a cold sales email") 
    for i, agent in enumerate([sales_agent1, sales_agent2, sales_agent3], 1)
]
```

```python
#'resend' as Alternative to sendgrid
@function_tool
def  send_email(body:  str) -> Dict[str, str]:
	""" Send out an email with the given body to all sales prospects via Resend """
	# Set up email sender, recipient, and content
	from_email =  "onboarding@resend.dev"  # Replace with your verified sender
	to_email =  "kejxxx@gmail.com"#os.getenv("GMAIL_TO") " # Replace with recipient's email
	# Resend API headers and payload
	api_key=os.environ.get('RESEND_API_KEY')
	headers =  {
	"Authorization":  f"Bearer {api_key}",
	"Content-Type":  "application/json"
	}
	payload =  {
	"from":  f"No Body <{from_email}>",
	"to":  [to_email],"subject":  "Sales email",
	"html":  f"<p>{body}</p>"  # Body wrapped in <p> tags for HTML format
	}
	# Send email using Resend API
	response = requests.post("https://api.resend.com/emails",  json=payload,  headers=headers)
	# Check if the request was successful
	if response.status_code ==  202:
		return  {"status":  "success"}
	else:
		return  {"status":  "failure",  "message": response.text}
```

## Advanced Handoff System

```python
from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict, List
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio
import requests

load_dotenv(override=True)


class EmailService:

    @staticmethod
    @function_tool
    def send_email(body: str):
        sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
        from_email = Email("kejxxx@gmail.com")
        to_email = To("mkejxxx@gmail.com")
        content = Content("text/plain", body)
        mail = Mail(from_email, to_email, "Sales email", content).get()
        sg.client.mail.send.post(request_body=mail)
        return {"status": "success"}

	#Resend as an alternative to sendgrid
    @staticmethod
	@function_tool
    def send_resend_email(body: str):
        from_email = "onboarding@resend.dev"
        to_email = "kejxxx@gmail.com"
        api_key = os.environ.get('RESEND_API_KEY')
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "from": f"Mohamed kejxxx <{from_email}>",
            "to": [to_email],
            "subject": "Sales email",
            "html": f"<p>{body}</p>"
        }
        response = requests.post("https://api.resend.com/emails", json=payload, headers=headers)
        return {"status": "success"} if response.status_code == 202 else {"status": "failure", "message": response.text}

    @staticmethod
    @function_tool
    def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
        sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
        from_email = Email("kejxxx@gmail.com")
        to_email = To("kejxxx@gmail.com")
        content = Content("text/html", html_body)
        mail = Mail(from_email, to_email, subject, content).get()
        sg.client.mail.send.post(request_body=mail)
        return {"status": "success"}

class SalesAgents:
    def __init__(self):
        self.instructions = {
            "professional": "You are a sales agent working for ComplAI, a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. You write professional, serious cold emails.",
            "engaging": "You are a humorous, engaging sales agent working for ComplAI, a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. You write witty, engaging cold emails that are likely to get a response.",
            "busy": "You are a busy sales agent working for ComplAI, a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. You write concise, to the point cold emails."
        }
        
        self.professional_agent = Agent(
            name="Professional Sales Agent",
            instructions=self.instructions["professional"],
            model="gpt-4o-mini"
        )
        
        self.engaging_agent = Agent(
            name="Engaging Sales Agent",
            instructions=self.instructions["engaging"],
            model="gpt-4o-mini"
        )
        
        self.busy_agent = Agent(
            name="Busy Sales Agent",
            instructions=self.instructions["busy"],
            model="gpt-4o-mini"
        )
        
        self.picker_agent = Agent(
            name="sales_picker",
            instructions="You pick the best cold sales email from the given options. Imagine you are a customer and pick the one you are most likely to respond to. Do not give an explanation; reply with the selected email only.",
            model="gpt-4o-mini"
        )
    
    def get_agents(self):
        return [self.professional_agent, self.engaging_agent, self.busy_agent]
    
    def get_tools(self):
        description = "Write a cold sales email"
        return [
            self.professional_agent.as_tool(tool_name="sales_agent1", tool_description=description),
            self.engaging_agent.as_tool(tool_name="sales_agent2", tool_description=description),
            self.busy_agent.as_tool(tool_name="sales_agent3", tool_description=description)
        ]

class EmailFormattingAgents:
    def __init__(self):
        self.subject_writer = Agent(
            name="Email subject writer",
            instructions="You can write a subject for a cold sales email. You are given a message and you need to write a subject for an email that is likely to get a response.",
            model="gpt-4o-mini"
        )
        
        self.html_converter = Agent(
            name="HTML email body converter",
            instructions="You can convert a text email body to an HTML email body. You are given a text email body which might have some markdown and you need to convert it to an HTML email body with simple, clear, compelling layout and design.",
            model="gpt-4o-mini"
        )
        
        self.emailer_agent = Agent(
            name="Email Manager",
            instructions="You are an email formatter and sender. You receive the body of an email to be sent. You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. Finally, you use the send_html_email tool to send the email with the subject and HTML body.",
            tools=[
                self.subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email"),
                self.html_converter.as_tool(tool_name="html_converter", tool_description="Convert a text email body to an HTML email body"),
                EmailService.send_html_email
            ],
            model="gpt-4o-mini",
            handoff_description="Convert an email to HTML and send it"
        )

class SalesManager:
    def __init__(self):
        self.sales_agents = SalesAgents()
        self.email_agents = EmailFormattingAgents()
        
        # Basic sales manager with tools
        self.basic_manager = Agent(
            name="Sales Manager",
            instructions="""
            You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
            Follow these steps carefully:
            1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
            2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
            3. Use the send_email tool to send the best email (and only the best email) to the user.
            Crucial Rules:
            - You must use the sales agent tools to generate the drafts — do not write them yourself.
            - You must send ONE email using the send_email tool — never more than one.
            """,
            tools=self.sales_agents.get_tools() + [EmailService.send_email],
            model="gpt-4o-mini"
        )
        
        # Advanced sales manager with handoffs
        self.advanced_manager = Agent(
            name="Sales Manager",
            instructions="""
            You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
            Follow these steps carefully:
            1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
            2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
            You can use the tools multiple times if you're not satisfied with the results from the first try.
            3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will take care of formatting and sending.
            Crucial Rules:
            - You must use the sales agent tools to generate the drafts — do not write them yourself.
            - You must hand off exactly ONE email to the Email Manager — never more than one.
            """,
            tools=self.sales_agents.get_tools(),
            handoffs=[self.email_agents.emailer_agent],
            model="gpt-4o-mini"
        )

class EmailWorkflow:
    def __init__(self):
        self.sales_agents = SalesAgents()
        self.manager = SalesManager()
    
    async def run_streaming_demo(self):
        result = Runner.run_streamed(self.sales_agents.professional_agent, input="Write a cold sales email")
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
    
    async def run_parallel_emails(self):
        message = "Write a cold sales email"
        with trace("Parallel cold emails"):
            results = await asyncio.gather(
                Runner.run(self.sales_agents.professional_agent, message),
                Runner.run(self.sales_agents.engaging_agent, message),
                Runner.run(self.sales_agents.busy_agent, message)
            )
        outputs = [result.final_output for result in results]
        for output in outputs:
            print(output + "\n\n")
    
    async def run_email_selection(self):
        message = "Write a cold sales email"
        with trace("Selection from sales people"):
            results = await asyncio.gather(
                Runner.run(self.sales_agents.professional_agent, message),
                Runner.run(self.sales_agents.engaging_agent, message),
                Runner.run(self.sales_agents.busy_agent, message)
            )
            outputs = [result.final_output for result in results]
            emails = "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)
            best = await Runner.run(self.sales_agents.picker_agent, emails)
            print(f"Best sales email:\n{best.final_output}")
    
    async def run_basic_manager(self):
        message = "Send a cold sales email addressed to 'Dear CEO'"
        with trace("Sales manager"):
            result = await Runner.run(self.manager.basic_manager, message)
    
    async def run_advanced_manager(self):
        message = "Send out a cold sales email addressed to Dear CEO from Alice"
        with trace("Automated SDR"):
            result = await Runner.run(self.manager.advanced_manager, message)
```

## Design Pattern Examples

```python
# Agent Specialization Pattern
subject_agent = Agent(name="Subject Writer", instructions="Write email subjects only")
body_agent = Agent(name="Body Writer", instructions="Write email bodies only")

# Tool Composition Pattern
email_tools = [
    generate_content_tool,
    format_html_tool, 
    add_subject_tool,
    send_email_tool
]
manager = Agent(tools=email_tools, instructions="Use tools to complete email workflow")

# Handoff Chain Pattern
generator_agent = Agent(handoffs=[formatter_agent])
formatter_agent = Agent(handoffs=[sender_agent]) 
sender_agent = Agent(tools=[send_tools])

# Trace Organization Pattern
with trace("Email Campaign"):
    with trace("Content Generation"):
        pass
    with trace("Quality Review"):
        pass
    with trace("Delivery"):
       pass
```

## Design Patterns and Best Practices

### 1. Agent Specialization Pattern

**Principle**: Create agents with narrow, well-defined responsibilities

```python
# Good: Specialized agents
subject_agent = Agent(name="Subject Writer", instructions="Write email subjects only")
body_agent = Agent(name="Body Writer", instructions="Write email bodies only")

# Avoid: Overly broad agents
general_agent = Agent(name="Email Agent", instructions="Do everything email-related")
```

### 2. Tool Composition Pattern

**Principle**: Build complex capabilities from simple, reusable tools

```python
# Compose complex workflows from simple tools
email_tools = [
    generate_content_tool,
    format_html_tool, 
    add_subject_tool,
    send_email_tool
]

manager = Agent(tools=email_tools, instructions="Use tools to complete email workflow")
```

### 3. Handoff Chain Pattern

**Principle**: Create clear delegation chains for complex processes

```python
# Clear handoff chain: Generator → Formatter → Sender
generator_agent = Agent(handoffs=[formatter_agent])
formatter_agent = Agent(handoffs=[sender_agent]) 
sender_agent = Agent(tools=[send_tools])
```

### 4. Trace Organization Pattern

**Principle**: Use meaningful trace names for debugging and monitoring

```python
# Hierarchical tracing
with trace("Email Campaign"):
    with trace("Content Generation"):
        # Generation logic
    with trace("Quality Review"):
        # Review logic  
    with trace("Delivery"):
        # Sending logic
```
## Advanced Concepts

### Streaming Responses

```python
# Stream agent responses for better UX
result = Runner.run_streamed(agent, message)
async for event in result.stream_events():
    if event.type == "raw_response_event":
        print(event.data.delta, end="", flush=True)
```

### Multi-Model Support

```python
# Use different models for different tasks
fast_agent = Agent(model="gpt-4o-mini")  # For quick tasks
powerful_agent = Agent(model="gpt-4o")   # For complex reasoning
```

### Dynamic Tool Selection

```python
# Agents can choose from available tools based on context
flexible_agent = Agent(
    tools=[tool1, tool2, tool3],
    instructions="Choose the most appropriate tool for each task"
)
```

## Commercial Applications

### Immediate Applications
- **Sales Automation**: Complete SDR workflows
- **Customer Support**: Multi-tier response systems  
- **Content Generation**: Blog posts, marketing materials
- **Data Processing**: Analysis and reporting pipelines

### Advanced Applications
- **Interactive Conversations**: Email-based customer engagement
- **Process Automation**: End-to-end business workflows
- **Quality Assurance**: Multi-agent review systems
- **Personalization**: Dynamic content adaptation

### Implementation Considerations

1. **Scalability**: Design for concurrent agent execution
2. **Monitoring**: Implement comprehensive tracing
3. **Error Recovery**: Build resilient agent interactions
4. **Cost Management**: Optimize model usage patterns
5. **Security**: Validate agent inputs and outputs

## Key Takeaways

1. **Progressive Complexity**: Build systems incrementally from simple workflows to autonomous agents
2. **Tool Abstraction**: Leverage framework capabilities to eliminate boilerplate
3. **Agent Specialization**: Create focused agents rather than generalist systems
4. **Delegation Patterns**: Use handoffs for clear responsibility transfer
5. **Monitoring**: Implement comprehensive tracing for debugging and optimization

>The OpenAI Agents SDK provides a remarkably clean abstraction for building sophisticated multi-agent systems. The key insight is understanding when to use tools (for capabilities) versus handoffs (for delegation), and how to compose these patterns into robust, autonomous systems.

>The progression from simple workflows to true agentic behavior represents a fundamental shift in how we architect AI systems - from predetermined sequences to autonomous decision-making entities that can adapt and respond to complex scenarios.


----

# Advanced OpenAI Agents SDK: Multi-Model Integration, Structured Outputs, and Guardrails

## Overview

This comprehensive guide explores advanced concepts in the OpenAI Agents SDK, focusing on three critical areas for LLM Engineers: multi-model integration through OpenAI-compatible endpoints, structured outputs using Pydantic schemas, and guardrails for input/output validation. The implementation demonstrates a sophisticated Sales Development Representative (SDR) automation system that showcases these concepts in a practical business context.

## Core Architecture Concepts

### Agent-to-Tool Transformation Pattern

The OpenAI Agents SDK implements a powerful abstraction where agents can be seamlessly converted into tools using the `as_tool()` method. This pattern enables hierarchical agent architectures:

```python
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
```

This transformation allows agents to be consumed by other agents as callable functions, creating a composable architecture where complex workflows can be built from simpler agent components.

### Tools vs Handoffs: Control Flow Patterns

The framework distinguishes between two collaboration patterns:

- **Tools**: Function calls that return control to the calling agent
- **Handoffs**: Transfer of control to another agent in the workflow

```python
# Tools - agents as callable functions
tools = [tool1, tool2, tool3]

# Handoffs - control transfer
handoffs = [emailer_agent]
```

This distinction is crucial for designing agent workflows where some operations require return values (tools) while others represent workflow transitions (handoffs).

## Multi-Model Integration Architecture

### OpenAI-Compatible Endpoint Pattern

The SDK leverages OpenAI-compatible endpoints to integrate multiple LLM providers seamlessly:

```python
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
```

### Client Instantiation Pattern

Each model provider requires a dedicated AsyncOpenAI client configured with the appropriate base URL and API key:

```python
deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)
```

### Model Abstraction Layer

The `OpenAIChatCompletionsModel` class provides a unified interface across different providers:

```python
deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client)
llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)
```

This abstraction enables model-agnostic agent creation where the same agent logic can be executed across different LLM providers.

### Agent Specialization Through Model Selection

Different models can be assigned specialized roles based on their characteristics:

```python
sales_agent1 = Agent(name="DeepSeek Sales Agent", instructions=instructions1, model=deepseek_model)
sales_agent2 = Agent(name="Gemini Sales Agent", instructions=instructions2, model=gemini_model)
sales_agent3 = Agent(name="Llama3.3 Sales Agent", instructions=instructions3, model=llama3_3_model)
```

Each agent maintains distinct personalities and capabilities while operating within the same framework.

## Structured Outputs Implementation

### Pydantic Schema Definition

Structured outputs require defining Pydantic BaseModel classes that specify the expected output format:

```python
class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str
```

This schema enforces type safety and provides clear contracts for agent outputs.

### Agent Configuration for Structured Outputs

Agents are configured to produce structured outputs using the `output_type` parameter:

```python
guardrail_agent = Agent( 
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)
```

This configuration ensures the agent's response conforms to the specified schema rather than returning free-form text.

### Benefits of Structured Outputs

1. **Type Safety**: Compile-time validation of expected data structures
2. **API Consistency**: Predictable response formats for downstream processing
3. **Error Reduction**: Elimination of parsing errors from unstructured text
4. **Integration Simplicity**: Direct object access instead of text parsing

## Guardrails Architecture

### Input Guardrail Pattern

Input guardrails validate incoming data before processing begins:

```python
@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(
        output_info={"found_name": result.final_output},
        tripwire_triggered=is_name_in_message
    )
```

### Guardrail Function Output Contract

All guardrails must return a `GuardrailFunctionOutput` object with:
- `output_info`: Dictionary containing diagnostic information
- `tripwire_triggered`: Boolean indicating whether the guardrail was violated

### Agent-Based Guardrails

The framework's unique approach allows guardrails themselves to be powered by LLMs:

```python
guardrail_agent = Agent( 
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)
```

This enables sophisticated validation logic that can understand context and nuance rather than simple pattern matching.

### Guardrail Integration

Guardrails are integrated into agents during instantiation:

```python
careful_sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=[emailer_agent],
    model="gpt-4o-mini",
    input_guardrails=[guardrail_against_name]
)
```

### Exception-Based Control Flow

When guardrails are triggered, the system raises exceptions to halt processing:

```
GuardrailInputTriggered: Guardrail triggered tripwire
```

This provides a clear mechanism for handling policy violations and prevents unauthorized operations.

## Advanced Workflow Patterns

### Multi-Agent Evaluation Pattern

The sales manager implements a sophisticated evaluation workflow:

```python
sales_manager_instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
 
Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
 
2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
You can use the tools multiple times if you're not satisfied with the results from the first try.
 
3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will take care of formatting and sending.
"""
```

This pattern demonstrates:
- **Parallel Generation**: Multiple agents working simultaneously
- **Quality Control**: Iterative refinement capability
- **Decision Making**: Agent-based selection logic
- **Workflow Orchestration**: Controlled handoff to specialized agents

### Asynchronous Execution Model

The framework leverages Python's async/await pattern for efficient I/O operations:

```python
with trace("Automated SDR"):
    result = await Runner.run(sales_manager, message)
```

This enables parallel execution of multiple LLM calls, significantly improving performance for multi-agent workflows.

## Email Processing Pipeline

### Specialized Agent Roles

The email processing pipeline demonstrates role-based agent specialization:

```python
subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model="gpt-4o-mini")
html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model="gpt-4o-mini")
```

### Tool Chain Pattern

The emailer agent orchestrates multiple specialized tools:

```python
email_tools = [subject_tool, html_tool, send_html_email]

emailer_agent = Agent(
    name="Email Manager",
    instructions=instructions,
    tools=email_tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it"
)
```

### Function Tool Decorator

The `@function_tool` decorator seamlessly integrates external services:

```python
@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    # Implementation details...
    return {"status": "success"}
```

## Error Handling and Stability Considerations

### Autonomous Agent Challenges

We highlight inherent instability in autonomous agent systems:

> "There is inherent instability with these autonomous agent frameworks. And that's something that you need to code explicitly for."

### Infinite Loop Prevention

The instruction "You can use the tools multiple times if you're not satisfied with the results" can lead to infinite loops, requiring careful prompt engineering and timeout mechanisms.

### Trace Analysis

The framework provides comprehensive tracing capabilities for debugging:

```python
with trace("Protected Automated SDR"):
    result = await Runner.run(careful_sales_manager, message)
```

Traces reveal execution patterns, including parallel execution and potential performance bottlenecks.

## Production Considerations

### API Key Management

The implementation demonstrates proper environment variable usage for API keys:

```python
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
```

### Model Provider Limitations

The course notes acknowledge current limitations:

> as of today "Anthropic does not offer an OpenAI compatible endpoint"

Workarounds include third-party providers like OpenRouter or the MCP protocol integration.

### Cost Optimization

Different models offer varying cost-performance tradeoffs:
- DeepSeek: Cost-effective option
- Gemini: Google's competitive offering  
- Llama 3.3: Open-source alternative via Groq

## Security and Compliance

### PII Protection

Guardrails serve as critical security controls for preventing data leakage:

```python
instructions="Check if the user is including someone's personal name in what they want you to do."
```

This pattern can be extended to detect:
- Phone numbers
- Email addresses
- Social security numbers
- Credit card information

### Input Validation

The guardrail system provides a framework for comprehensive input validation that goes beyond simple pattern matching to include contextual understanding.

## Future Extensions

### Structured Email Generation

The course suggests implementing structured outputs for email generation:

```python
class EmailOutput(BaseModel):
    subject: str
    recipient: str
    body: str
```

This would provide better type safety and easier integration with email systems.

### Output Guardrails

While the implementation focuses on input guardrails, output guardrails follow the same pattern:

```python
@output_guardrail
async def guardrail_against_inappropriate_content(ctx, agent, output):
    # Validation logic
    return GuardrailFunctionOutput(...)
```

### User Interface Integration

We could add a user interfaces around the agent system, transforming it from a development tool into a production business application.

```python
"""
SDR Automation - OpenAI Agents SDK Implementation
"""

from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from typing import Dict, List
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio


class EmailService:
    """
    CONCEPT: Function Tool Pattern
    - Uses @function_tool decorator to convert regular functions into agent tools
    - Eliminates JSON boilerplate
    - Automatically creates tool schema from function signature
    """
    
    def __init__(self):
        load_dotenv(override=True)
        self.sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    
    @function_tool
    def send_email(self, body: str) -> Dict[str, str]:
        """Send out an email with the given body to all sales prospects"""
        from_email = Email("kejxxx@gmail.com")  # Change to your verified sender
        to_email = To("mkejxxx@gmail.com")  # Change to your recipient
        content = Content("text/plain", body)
        mail = Mail(from_email, to_email, "Sales email", content).get()
        self.sg.client.mail.send.post(request_body=mail)
        return {"status": "success"}
    
    @function_tool
    def send_html_email(self, subject: str, html_body: str) -> Dict[str, str]:
        """Send out an email with the given subject and HTML body to all sales prospects"""
        from_email = Email("kejxxx@gmail.com")  # Change to your verified sender
        to_email = To("kejxxx@gmail.com")  # Change to your recipient
        content = Content("text/html", html_body)
        mail = Mail(from_email, to_email, subject, content).get()
        self.sg.client.mail.send.post(request_body=mail)
        return {"status": "success"}


class SalesAgentFactory:
    """
    CONCEPT: Agent Specialization Pattern
    - Creates specialized agents with different personalities and writing styles
    - Each agent has distinct instructions but uses the same underlying model
    - Demonstrates how instructions shape agent behavior
    """
    
    @staticmethod
    def create_professional_agent() -> Agent:
        """Creates a professional, serious sales agent"""
        instructions = """You are a sales agent working for ComplAI, 
        a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
        You write professional, serious cold emails."""
        
        return Agent(
            name="Professional Sales Agent",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    @staticmethod
    def create_engaging_agent() -> Agent:
        """Creates a humorous, engaging sales agent"""
        instructions = """You are a humorous, engaging sales agent working for ComplAI, 
        a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
        You write witty, engaging cold emails that are likely to get a response."""
        
        return Agent(
            name="Engaging Sales Agent",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    @staticmethod
    def create_concise_agent() -> Agent:
        """Creates a busy, concise sales agent"""
        instructions = """You are a busy sales agent working for ComplAI, 
        a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
        You write concise, to the point cold emails."""
        
        return Agent(
            name="Busy Sales Agent",
            instructions=instructions,
            model="gpt-4o-mini"
        )


class EmailProcessingAgents:
    """
    CONCEPT: Specialized Agent Roles
    - Creates agents with specific, focused responsibilities
    - Demonstrates role-based agent specialization
    - Each agent handles one aspect of email processing
    """
    
    @staticmethod
    def create_subject_writer() -> Agent:
        """Creates an agent specialized in writing email subjects"""
        instructions = """You can write a subject for a cold sales email. 
        You are given a message and you need to write a subject for an email that is likely to get a response."""
        
        return Agent(
            name="Email subject writer",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    @staticmethod
    def create_html_converter() -> Agent:
        """Creates an agent specialized in converting text to HTML"""
        instructions = """You can convert a text email body to an HTML email body. 
        You are given a text email body which might have some markdown 
        and you need to convert it to an HTML email body with simple, clear, compelling layout and design."""
        
        return Agent(
            name="HTML email body converter",
            instructions=instructions,
            model="gpt-4o-mini"
        )


class SDRAutomationSystem:
    """
    CONCEPT: Multi-Agent Orchestration System
    - Demonstrates Tools vs Handoffs pattern
    - Tools: Function calls that return control to calling agent
    - Handoffs: Transfer of control to another agent in workflow
    - Implements parallel agent execution with asyncio.gather()
    """
    
    def __init__(self):
        self.email_service = EmailService()
        self.sales_agents = self._create_sales_agents()
        self.email_manager = self._create_email_manager()
        self.sales_manager = self._create_sales_manager()
    
    def _create_sales_agents(self) -> List[Agent]:
        """
        CONCEPT: Agent-to-Tool Transformation
        - Agents can be converted to tools using as_tool() method
        - Enables hierarchical agent architectures
        - Tools allow agents to be consumed by other agents as callable functions
        """
        factory = SalesAgentFactory()
        return [
            factory.create_professional_agent(),
            factory.create_engaging_agent(),
            factory.create_concise_agent()
        ]
    
    def _create_email_manager(self) -> Agent:
        """
        CONCEPT: Tool Chain Pattern
        - Agent orchestrates multiple specialized tools
        - Sequential tool execution for complex workflows
        - Handoff description enables workflow transitions
        """
        processing_agents = EmailProcessingAgents()
        subject_writer = processing_agents.create_subject_writer()
        html_converter = processing_agents.create_html_converter()
        
        # Convert agents to tools
        subject_tool = subject_writer.as_tool(
            tool_name="subject_writer",
            tool_description="Write a subject for a cold sales email"
        )
        html_tool = html_converter.as_tool(
            tool_name="html_converter",
            tool_description="Convert a text email body to an HTML email body"
        )
        
        tools = [subject_tool, html_tool, self.email_service.send_html_email]
        
        instructions = """You are an email formatter and sender. You receive the body of an email to be sent. 
        You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. 
        Finally, you use the send_html_email tool to send the email with the subject and HTML body."""
        
        return Agent(
            name="Email Manager",
            instructions=instructions,
            tools=tools,
            model="gpt-4o-mini",
            handoff_description="Convert an email to HTML and send it"
        )
    
    def _create_sales_manager(self) -> Agent:
        """
        CONCEPT: Multi-Agent Evaluation Pattern
        - Orchestrates multiple agents for parallel generation
        - Implements quality control through iterative refinement
        - Uses both tools (for generation) and handoffs (for workflow transfer)
        """
        # Convert sales agents to tools
        tools = []
        for i, agent in enumerate(self.sales_agents, 1):
            tool = agent.as_tool(
                tool_name=f"sales_agent{i}",
                tool_description="Write a cold sales email"
            )
            tools.append(tool)
        
        instructions = """
        You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
         
        Follow these steps carefully:
        1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
         
        2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
        You can use the tools multiple times if you're not satisfied with the results from the first try.
         
        3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will take care of formatting and sending.
         
        Crucial Rules:
        - You must use the sales agent tools to generate the drafts — do not write them yourself.
        - You must hand off exactly ONE email to the Email Manager — never more than one.
        """
        
        return Agent(
            name="Sales Manager",
            instructions=instructions,
            tools=tools,
            handoffs=[self.email_manager],
            model="gpt-4o-mini"
        )
    
    async def run_parallel_generation(self, message: str) -> List[str]:
        """
        CONCEPT: Asynchronous Execution Model
        - Uses asyncio.gather() for parallel agent execution
        - Leverages async/await for efficient I/O operations
        - Enables simultaneous LLM calls for improved performance
        """
        with trace("Parallel cold emails"):
            results = await asyncio.gather(
                Runner.run(self.sales_agents[0], message),
                Runner.run(self.sales_agents[1], message),
                Runner.run(self.sales_agents[2], message),
            )
        
        return [result.final_output for result in results]
    
    async def run_with_selection(self, message: str) -> str:
        """
        CONCEPT: Agent-Based Selection Pattern
        - Uses dedicated agent for quality evaluation
        - Implements decision-making logic through agent instructions
        - Demonstrates agent collaboration for complex workflows
        """
        # Create sales picker agent
        sales_picker = Agent(
            name="sales_picker",
            instructions="""You pick the best cold sales email from the given options. 
            Imagine you are a customer and pick the one you are most likely to respond to. 
            Do not give an explanation; reply with the selected email only.""",
            model="gpt-4o-mini"
        )
        
        with trace("Selection from sales people"):
            # Generate emails in parallel
            results = await asyncio.gather(
                Runner.run(self.sales_agents[0], message),
                Runner.run(self.sales_agents[1], message),
                Runner.run(self.sales_agents[2], message),
            )
            outputs = [result.final_output for result in results]
            
            # Format for selection
            emails = "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)
            
            # Select best email
            best = await Runner.run(sales_picker, emails)
            
            return best.final_output
    
    async def run_automated_sdr(self, message: str):
        """
        CONCEPT: End-to-End Workflow Orchestration
        - Combines tools and handoffs for complete automation
        - Demonstrates workflow transitions through handoff pattern
        - Implements tracing for observability and debugging
        """
        with trace("Automated SDR"):
            result = await Runner.run(self.sales_manager, message)
            return result


# Usage Example
async def main():
    """
    CONCEPT: System Integration and Execution
    - Demonstrates complete SDR automation workflow
    - Shows how all patterns work together in practice
    """
    sdr_system = SDRAutomationSystem()
    
    # Example 1: Parallel generation
    print("=== Parallel Generation ===")
    outputs = await sdr_system.run_parallel_generation("Write a cold sales email")
    for i, output in enumerate(outputs, 1):
        print(f"Agent {i}: {output}\n")
    
    # Example 2: With selection
    print("=== With Selection ===")
    best_email = await sdr_system.run_with_selection("Write a cold sales email")
    print(f"Best email: {best_email}\n")
    
    # Example 3: Full automation with handoffs
    print("=== Full Automation ===")
    await sdr_system.run_automated_sdr("Send out a cold sales email addressed to Dear CEO from Alice")


if __name__ == "__main__":
    asyncio.run(main())

```

## Key takeaways

This implementation demonstrates sophisticated patterns for building production-ready agent systems using the OpenAI Agents SDK. The combination of multi-model integration, structured outputs, and guardrails provides a robust foundation for enterprise applications requiring both flexibility and safety. The modular architecture enables easy extension and customization while maintaining clear separation of concerns across different system components.

The key insights for LLM Engineers are:

1. **Abstraction Layers**: Use model abstraction to enable provider flexibility
2. **Structured Contracts**: Implement Pydantic schemas for type safety
3. **Security by Design**: Integrate guardrails as first-class citizens
4. **Workflow Orchestration**: Leverage tools vs handoffs appropriately
5. **Async Patterns**: Utilize asynchronous execution for performance
6. **Observability**: Implement comprehensive tracing for debugging

These patterns form the foundation for building scalable, secure, and maintainable agent-based systems in production environments.



----

# Deep Research Agent

## Overview

This guide provides a thorough explanation of building a Deep Research Agent using OpenAI's Agents SDK, one of the classic use cases in Agentic AI. The system demonstrates how to create autonomous agents that can search the internet, synthesize information, and generate comprehensive reports.
> a sanitized version of the code is located in code folder

## Core Architecture

The Deep Research Agent follows a multi-agent architecture with specialized roles:

1. **Search Agent** - Executes web searches using OpenAI's hosted tools
2. **Planner Agent** - Determines optimal search strategies using structured outputs
3. **Writer Agent** - Synthesizes research into comprehensive reports
4. **Email Agent** - Formats and delivers final reports

## Key Concepts

### 1. Hosted Tools

OpenAI provides three hosted tools that run remotely:
- **WebSearchTool** - Web search capabilities (Made available by openAI - expensive!)
- **FileSearchTool** - Vector store searches
- **ComputerTool** - Screenshot and computer automation

The WebSearchTool costs approximately 2.5 cents per call, making cost management important for production use.

### 2. Structured Outputs

Structured outputs use Pydantic models to enforce response schemas, ensuring consistent data formats between agents. This is crucial for agent-to-agent communication.

### 3. Asynchronous Processing

The system uses asyncio for parallel execution of multiple searches, significantly improving performance when handling multiple queries simultaneously.

## Implementation Details

### Search Agent Implementation

```python
INSTRUCTIONS = "You are a research assistant. Given a search term, you search the web for that term and \
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 \
words. Capture the main points. Write succintly, no need to have complete sentences or good \
grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the \
essence and ignore any fluff. Do not include any additional commentary other than the summary itself."

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
```

**Key Technical Points:**
- Uses OpenAI's hosted WebSearchTool with "low" context size for cost optimization
- `tool_choice="required"` ensures the agent must use the search tool
- Instructions emphasize concise, factual summaries optimized for downstream processing
- The agent acts as a wrapper around the hosted search functionality

### Planner Agent with Structured Outputs

> improve the search query before sending it to `search_agent`
```python
class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)
```

**Technical Analysis:**
- **Structured Outputs**: Uses Pydantic BaseModel to enforce response schema
- **Field Descriptions**: Critical for providing context to the model about field purposes
- **Chain of Thought**: Requesting "reason" before "query" encourages better reasoning
- **JSON Conversion**: Behind the scenes, Pydantic models are converted to JSON schemas for the model

### Writer Agent for Report Generation

```python
class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")

writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ReportData,
)
```

**Key Features:**
- Generates structured reports with summary, full content, and follow-up suggestions
- Uses markdown format for rich text formatting
- Aims for 1000+ words and 5-10 pages of content

### Email Agent with Function Tools

```python
@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("kejxxx@gmail.com")
    to_email = To("mkejxxx@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return "success"

email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model="gpt-4o-mini",
)
```

**Technical Details:**
- Uses `@function_tool` decorator to convert Python functions into agent tools
- Integrates with SendGrid for email delivery
- Converts markdown reports to HTML format
- Agent has full discretion over subject line and formatting

## Workflow Implementation

### Core Workflow Functions

#### 1. Search Planning
```python
async def plan_searches(query: str):
    """ Use the planner_agent to plan which searches to run for the query """
    print("Planning searches...")
    result = await Runner.run(planner_agent, f"Query: {query}")
    print(f"Will perform {len(result.final_output.searches)} searches")
    return result.final_output

	#e.g. searches=[WebSearchItem(reason='To find the most recent and trending AI agent frameworks in 2025.', query='latest AI agent frameworks 2025'), WebSearchItem(reason='To look for reports or articles that discuss the advancements in AI agent frameworks for the year 2025.', query='AI agent frameworks 2025 news'), WebSearchItem(reason='To identify emerging technologies and tools specifically designed for AI agents that were popular or released in 2025.', query='new AI agent technologies 2025')]

```

#### 2. Parallel Search Execution
```python
async def perform_searches(search_plan: WebSearchPlan):
    """ Call search() for each item in the search plan """
    print("Searching...")
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("Finished searching")
    return results

async def search(item: WebSearchItem):
    """ Use the search agent to run a web search for each item in the search plan """
    input = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input)
    return result.final_output
```

**Asynchronous Processing Benefits:**
- Multiple searches execute in parallel using `asyncio.gather()`
- Significantly reduces total execution time
- Each search task is independent and can run concurrently
- Provides context (reason) to search agent for better results

#### 3. Report Generation and Delivery
```python
async def write_report(query: str, search_results: list[str]):
    """ Use the writer agent to write a report based on the search results"""
    print("Thinking about report...")
    input = f"Original query: {query}\nSummarized search results: {search_results}"
    result = await Runner.run(writer_agent, input)
    print("Finished writing report")
    return result.final_output

async def send_email(report: ReportData):
    """ Use the email agent to send an email with the report """
    print("Writing email...")
    result = await Runner.run(email_agent, report.markdown_report)
    print("Email sent")
    return report
```

### Main Execution Flow

```python
query = "Latest AI Agent frameworks in 2025"

with trace("Research trace"):
    print("Starting research...")
    search_plan = await plan_searches(query)
    search_results = await perform_searches(search_plan)
    report = await write_report(query, search_results)
    await send_email(report)  
    print("Hooray!")
```

**Execution Sequence:**
1. **Planning Phase**: Planner agent creates search strategy
2. **Research Phase**: Multiple searches execute in parallel
3. **Synthesis Phase**: Writer agent creates comprehensive report
4. **Delivery Phase**: Email agent formats and sends results

## Advanced Technical Concepts

### Chain of Thought Prompting
The system implements chain of thought prompting by requesting reasoning before queries:
- Models predict tokens sequentially
- Asking for reasoning first improves subsequent query quality
- Creates coherent, logical search strategies
- Leverages next-token prediction behavior for better outcomes

### Cost Optimization Strategies
- Use `search_context_size="low"` for reduced costs
- Limit number of searches (configurable via `HOW_MANY_SEARCHES`)
- Choose cost-effective models (gpt-4o-mini vs gpt-4)
- Monitor API usage through OpenAI dashboard

### Error Handling and Robustness
- Environment variable validation for API keys
- Structured outputs prevent malformed responses
- Async error handling for parallel operations
- Trace functionality for debugging and monitoring

### Scalability Considerations
- Parallel processing reduces latency
- Modular agent design allows independent scaling
- Configurable search depth based on requirements
- Stateless design enables horizontal scaling

## Production Deployment Considerations

### Security
- Email validation and sanitization
- Rate limiting for cost control
- Input validation for search queries

### Monitoring and Observability
- OpenAI trace integration for debugging
- Cost tracking and alerting
- Performance metrics collection
- Error logging and alerting

### Configuration Management
- Configurable search parameters
- Model selection flexibility
- Email template customization
- Search result limits

## Extension Possibilities

### Enhanced Search Capabilities
- Multiple search providers integration
- Domain-specific search filtering
- Real-time vs cached results
- Search result ranking and filtering

### Advanced Report Generation
- Multi-format output (PDF, Word, etc.)
- Custom report templates
- Citation and reference management
- Visual content integration

### Workflow Enhancements
- User approval workflows
- Scheduled research tasks
- Multi-language support
- Collaborative research features

## Best Practices

### Prompt Engineering
- Clear, specific instructions for each agent
- Context-aware prompting with reasoning chains
- Structured output schemas for consistency
- Error handling instructions

### Agent Design Patterns
- Single responsibility principle for agents
- Loose coupling between agent components
- Standardized communication protocols
- Graceful degradation strategies

### Performance Optimization
- Async/await for I/O bound operations
- Parallel processing where possible
- Caching strategies for repeated queries
- Resource pooling for API connections

### Testing and Validation
- Unit tests for individual agents
- Integration tests for full workflows
- Cost simulation and budgeting
- Output quality validation

## Conclusion

The Deep Research Agent demonstrates sophisticated agentic AI patterns including multi-agent orchestration, structured outputs, asynchronous processing, and hosted tool integration. This architecture provides a foundation for building production-ready research automation systems while maintaining cost efficiency and scalability.

The modular design allows for easy extension and customization, making it applicable across various business domains and use cases. The combination of OpenAI's hosted tools with custom agent logic creates a powerful platform for automated research and reporting.
