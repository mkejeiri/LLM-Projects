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

