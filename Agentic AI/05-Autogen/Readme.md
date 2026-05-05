# 05 — AutoGen: Microsoft's Multi-Agent Framework

## Table of Contents

- [Introduction](#introduction)
- [The AutoGen / AG2 Fork Drama](#the-autogen--ag2-fork-drama)
- [AutoGen Architecture Overview](#autogen-architecture-overview)
  - [AutoGen Core](#autogen-core)
  - [AutoGen AgentChat](#autogen-agentchat)
  - [AutoGen Studio](#autogen-studio)
  - [Magentic-One](#magentic-one)
- [Building Blocks of AgentChat](#building-blocks-of-agentchat)
  - [Models](#1-models)
  - [Messages](#2-messages)
  - [Agents](#3-agents)
  - [Teams](#4-teams)
- [Lab 1: Airline Agent with SQLite Tool Use](#lab-1-airline-agent-with-sqlite-tool-use)
  - [Environment Setup](#environment-setup)
  - [Creating a Model Client (OpenRouter)](#creating-a-model-client-openrouter)
  - [Alternative: Local Model with Ollama](#alternative-local-model-with-ollama)
  - [Creating a Message](#creating-a-message)
  - [Creating an Agent](#creating-an-agent)
  - [Invoking the Agent](#invoking-the-agent)
  - [Building a SQLite Tool](#building-a-sqlite-tool)
  - [Arming the Agent with Tools](#arming-the-agent-with-tools)
  - [Executing the Tool-Equipped Agent](#executing-the-tool-equipped-agent)
- [Lab 2: AgentChat Deep Dive](#lab-2-agentchat-deep-dive)
  - [Multi-Modal Conversations](#multi-modal-conversations)
  - [Structured Outputs with Pydantic](#structured-outputs-with-pydantic)
  - [Using LangChain Tools from AutoGen](#using-langchain-tools-from-autogen)
  - [Teams: Multi-Agent Collaboration](#teams-multi-agent-collaboration)
  - [MCP Tools Preview](#mcp-tools-preview)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Introduction

AutoGen is an **open-source multi-agent framework from Microsoft Research**, first released in its current form (v0.4) in January 2026 as a ground-up rewrite adopting an **asynchronous, event-driven architecture**. The rewrite addressed criticisms of the earlier 0.2 version around observability, flexibility, control, and scalability.

This course uses **AutoGen v0.5.1** — the latest stable release at time of writing. While 0.5.x builds on the 0.4 rewrite, it is not a radical departure from 0.4; the core architecture remains the same.

> ⚠️ **Documentation Warning**: When searching for AutoGen documentation, be very careful to distinguish between docs for **v0.4+** (the current Microsoft version) and **v0.2** (the legacy version / AG2 fork). They look and feel quite different.

---

## The AutoGen / AG2 Fork Drama

Late 2024 saw significant drama in the AutoGen ecosystem:

**What happened:**
- The original co-creator of AutoGen and several key contributors **left Microsoft**
- They created a **fork** called **AG2** (AutoGen Gen 2), also branded as **AgentOS 2**
- The co-creator is now at Google, working on this forked version

**The confusing part:**
- AG2 forked from **AutoGen 0.2** (the older version), making it compatible with the legacy API
- AG2 is **incompatible** with Microsoft's AutoGen 0.4+ rewrite
- The AG2 team controls the **official `autogen` PyPI package** — meaning `pip install autogen` gives you AG2, *not* Microsoft's official AutoGen
- The AG2 team also controls the **original AutoGen Discord server**, causing further community confusion

**AG2's rationale:** Move faster without Microsoft's corporate bureaucracy. As of now, AG2 is at version 0.8, releasing rapidly.

**Microsoft's position:** They have no plans to slow down AutoGen development. The Microsoft version has the larger enterprise user base and broader community traction.

**For this course:** We use **Microsoft's official AutoGen** (installed via `autogen-agentchat` and `autogen-ext` packages, not the bare `autogen` package).

| | Microsoft AutoGen | AG2 (fork) |
|---|---|---|
| **Version** | 0.5.1+ | 0.8+ |
| **Based on** | 0.4 rewrite (async, event-driven) | 0.2 legacy |
| **PyPI package** | `autogen-agentchat`, `autogen-ext` | `autogen` |
| **GitHub** | [microsoft/autogen](https://github.com/microsoft/autogen) | [ag2ai/ag2](https://github.com/ag2ai/ag2) |
| **Discord** | New Microsoft-managed server | Original (controlled by AG2 team) |

---

## AutoGen Architecture Overview

AutoGen is not a single thing — it's an umbrella encompassing multiple layers:

![AutoGen Architecture Blocks](pic/autogen-bloc.png)

### AutoGen Core

A **model-agnostic runtime** for building scalable multi-agent systems. Think of it as the fabric/infrastructure layer:

- Manages **messaging between agents** (even distributed across machines)
- Provides an **agent runtime** — the execution environment agents live in
- Handles event-driven communication patterns
- Simpler than LangGraph but shares some conceptual overlap (orchestration of agent interactions)

Core is the foundation everything else is built on.

### AutoGen AgentChat

The **high-level framework** built on top of Core — and the primary focus of this module. AgentChat is the direct comparable to:
- **CrewAI** (crews ↔ teams)
- **OpenAI Agents SDK** (agents with tools)
- **LangGraph** (agent interaction patterns)

It provides a lightweight, simple abstraction for:
- Wrapping LLMs in agent constructs
- Equipping agents with tools
- Enabling agent-to-agent interaction

### AutoGen Studio

A **low-code/no-code** visual interface for constructing agent workflows. Microsoft positions this as a **research environment** — explicitly not production-ready. Less relevant for engineers who prefer code.

### Magentic-One

A **pre-built command-line application** — an out-of-the-box agent system you run from the terminal. Think of it as a canned agent framework similar to building your own "sidekick" agent, but provided ready-made.

**Our focus:** AutoGen Core + AgentChat (primarily AgentChat).

---

## Building Blocks of AgentChat

The core abstractions are deliberately simple and will feel familiar if you've used CrewAI or OpenAI Agents SDK:

### 1. Models

The wrapper around calling a large language model. Equivalent to `LLM` in other frameworks. AutoGen supports multiple providers through its extension system:

- `OpenAIChatCompletionClient` — OpenAI-compatible APIs (OpenAI, OpenRouter, Azure, etc.)
- `OllamaChatCompletionClient` — Local models via Ollama

### 2. Messages

A first-class concept in AutoGen that represents **all communication**:

- **User → Agent**: A user's request to an agent
- **Agent → Agent**: Inter-agent communication in multi-agent setups
- **Internal (tool calls)**: Events within an agent's execution (function calls, results)

This unified message abstraction means you can inspect the full trace of what happened during agent execution — critical for observability.

### 3. Agents

The core execution unit. An `AssistantAgent` wraps:
- A model client (the underlying LLM)
- A system message (instructions/persona)
- Tools (Python functions the agent can invoke)
- Configuration (streaming, reflection behavior)

### 4. Teams

Groups of agents that collaborate to achieve a goal. Analogous to a **Crew** in CrewAI — see [03-CrewAI](../03-CrewAI/) for the team/crew concept explored in depth.

---

## Lab 1: Airline Agent with SQLite Tool Use

A complete walkthrough building an airline booking agent that queries a SQLite database for ticket prices.

### Environment Setup

```python
import os
from dotenv import load_dotenv

# Load environment variables from the project root .env file
load_dotenv(dotenv_path="../.env", override=True)
```

The `.env` file contains:
```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### Creating a Model Client (OpenRouter)

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

# OpenAIChatCompletionClient works with any OpenAI-compatible API.
# We point it at OpenRouter by overriding base_url and api_key.
# model_info is required because "openai/gpt-4o-mini" (OpenRouter's naming)
# isn't in AutoGen's built-in model registry — we must explicitly declare capabilities.
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
    model_info={
        "vision": True,              # Model supports image inputs
        "function_calling": True,    # Model supports tool/function calling
        "json_output": True,         # Model can produce structured JSON
        "structured_output": True,   # Model supports structured output schemas
        "family": "unknown",         # Not a recognized family — tells AutoGen to skip lookup
    }
)
```

**Why `model_info`?** AutoGen maintains an internal registry of known models (GPT-4o, Claude, etc.) with their capabilities. When using OpenRouter's prefixed model names (e.g., `openai/gpt-4o-mini`), AutoGen can't find it in the registry and raises a `ValueError`. Providing `model_info` explicitly bypasses the lookup.

### Alternative: Local Model with Ollama

```python
# Ollama local model — requires Ollama server running locally (ollama serve)
# Useful for development without API costs or for air-gapped environments
# from autogen_ext.models.ollama import OllamaChatCompletionClient
# ollamamodel_client = OllamaChatCompletionClient(model="llama3.2")
```

The Ollama client is a drop-in replacement — you can swap `model_client` references to use local inference with zero code changes elsewhere.

### Creating a Message

```python
from autogen_agentchat.messages import TextMessage

# TextMessage is the fundamental message type in AgentChat.
# 'source' identifies who sent the message — here it's the end user.
# This object is what gets passed to agents via on_messages().
message = TextMessage(content="I'd like to go to London", source="user")
message
```

**Output:**
```
TextMessage(source='user', content="I'd like to go to London", type='TextMessage')
```

Messages are typed objects with metadata (ID, timestamp, source) — not raw strings. This enables full traceability through multi-agent workflows.

### Creating an Agent

```python
from autogen_agentchat.agents import AssistantAgent

# AssistantAgent is the most fundamental agent class in AgentChat.
# It wraps an LLM with a persona (system_message) and optional tools.
agent = AssistantAgent(
    name="airline_agent",                # Unique identifier for this agent
    model_client=model_client,           # The LLM backend to use
    system_message="You are a helpful assistant for an airline. You give short, humorous answers.",
    model_client_stream=True             # Stream responses token-by-token
)
```

This is the minimal agent — no tools, just an LLM with a persona. Comparable to creating an `Agent` in OpenAI Agents SDK or a `@agent` in CrewAI.

### Invoking the Agent

```python
from autogen_core import CancellationToken

# on_messages() is the primary way to invoke an agent.
# It's async (coroutine) — must be awaited.
# CancellationToken signals when the interaction is complete.
response = await agent.on_messages(
    [message],                            # List of messages to process
    cancellation_token=CancellationToken() # Required — framework uses this for lifecycle management
)
response.chat_message.content
```

**Output:**
```
'Great choice! London: where the rain is as regular as your morning coffee
and the fish and chips can double as a food group! Ready to jet off? ✈️'
```

The `CancellationToken` is a framework requirement for async lifecycle management — just instantiate it and pass it in.

### Building a SQLite Tool

Now we create a database-backed tool the agent can use:

```python
import sqlite3

# Delete existing database file if it exists (idempotent setup)
if os.path.exists("tickets.db"):
    os.remove("tickets.db")

# Create a fresh SQLite database with a cities table
conn = sqlite3.connect("tickets.db")
c = conn.cursor()
c.execute("CREATE TABLE cities (city_name TEXT PRIMARY KEY, round_trip_price REAL)")
conn.commit()
conn.close()
```

```python
# Helper to populate the database
def save_city_price(city_name, round_trip_price):
    conn = sqlite3.connect("tickets.db")
    c = conn.cursor()
    # REPLACE INTO = upsert (insert or update if PK exists)
    c.execute("REPLACE INTO cities (city_name, round_trip_price) VALUES (?, ?)",
              (city_name.lower(), round_trip_price))
    conn.commit()
    conn.close()

# Seed data — roundtrip prices in USD
save_city_price("London", 299)
save_city_price("Paris", 399)
save_city_price("Rome", 499)
save_city_price("Madrid", 550)
save_city_price("Barcelona", 580)
save_city_price("Berlin", 525)
```

```python
def get_city_price(city_name: str) -> float | None:
    """ Get the roundtrip ticket price to travel to the city """
    # This docstring is critical — AutoGen uses it as the tool description
    # that gets sent to the LLM for function calling.
    conn = sqlite3.connect("tickets.db")
    c = conn.cursor()
    # Parameterized query prevents SQL injection
    c.execute("SELECT round_trip_price FROM cities WHERE city_name = ?",
              (city_name.lower(),))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None
```

```python
# Quick sanity check — does the tool work?
get_city_price("Rome")  # → 499.0
```

### Arming the Agent with Tools

```python
from autogen_agentchat.agents import AssistantAgent

smart_agent = AssistantAgent(
    name="smart_airline_agent",
    model_client=model_client,
    system_message="You are a helpful assistant for an airline. "
                   "You give short, humorous answers, including the price of a roundtrip ticket.",
    model_client_stream=True,
    tools=[get_city_price],       # Pass Python functions directly — no decorator needed!
    reflect_on_tool_use=True      # Agent continues processing after tool returns (not just raw result)
)
```

**Key differences from other frameworks:**

| Framework | Tool Registration |
|---|---|
| **AutoGen AgentChat** | Pass raw Python function — no decorator, no wrapper |
| **OpenAI Agents SDK** | Requires `@function_tool` decorator |
| **LangGraph** | Requires `@tool` decorator or `StructuredTool` |
| **CrewAI** | Requires `@tool` decorator |

AutoGen inspects the function's **type hints** and **docstring** to automatically generate the tool schema sent to the LLM. This is the lightest-weight tool registration of any framework we've covered.

**`reflect_on_tool_use=True`**: Without this, the agent would return the raw tool output (e.g., `299.0`) directly. With reflection enabled, the agent takes the tool result and generates a natural language response incorporating it. You almost always want this set to `True`.

### Executing the Tool-Equipped Agent

```python
response = await smart_agent.on_messages([message], cancellation_token=CancellationToken())

# Inner messages show the agent's internal reasoning/tool-calling trace
for inner_message in response.inner_messages:
    print(inner_message.content)

# Final response to the user
response.chat_message.content
```

**Output:**
```
[FunctionCall(id='call_isnx2qXkwn1cadjd7rRjrGna', arguments='{"city_name":"London"}', name='get_city_price')]
[FunctionExecutionResult(content='299.0', name='get_city_price', call_id='call_isnx2qXkwn1cadjd7rRjrGna', is_error=False)]
```
```
'Sure, grab your passport and your sense of adventure! A roundtrip ticket
to London is just $299. Don't forget to brush up on your British slang—"cheerio"! 🍵✈️'
```

**What happened internally:**
1. The LLM received the user message + tool schema
2. It decided to call `get_city_price` with `city_name="London"`
3. AutoGen executed the function, got `299.0`
4. Because `reflect_on_tool_use=True`, the result was fed back to the LLM
5. The LLM generated a final human-friendly response incorporating the price

The `inner_messages` trace gives full observability into the agent's tool-calling behavior — essential for debugging and auditing.

---

## Lab 2: AgentChat Deep Dive

This lab extends the fundamentals from Lab 1 into more advanced territory: vision/multi-modal inputs, structured outputs via Pydantic, cross-framework tool interop with LangChain, multi-agent teams, and a preview of MCP (Model Context Protocol).

### Multi-Modal Conversations

AutoGen AgentChat supports sending images alongside text using `MultiModalMessage`. The model receives both and can reason about the visual content.

```python
import os
from io import BytesIO
import requests
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import Image as AGImage  # AutoGen's image wrapper
from PIL import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from IPython.display import display, Markdown

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)
```

```python
# Reusable model client — OpenRouter with explicit model_info
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
    model_info={
        "vision": True,              # Required for multi-modal — model must support image inputs
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    }
)
```

**Loading and wrapping an image:**

```python
# IMPORTANT: Use raw.githubusercontent.com for actual image bytes
# The github.com/blob/ URL returns an HTML page, causing UnidentifiedImageError
url = "https://raw.githubusercontent.com/mkejeiri/LLM-Projects/master/Agentic%20AI/05-Autogen/pic/autogen-bloc.png"

# Pipeline: HTTP GET → bytes → PIL Image → AutoGen AGImage wrapper
pil_image = Image.open(BytesIO(requests.get(url).content))
img = AGImage(pil_image)  # Converts PIL image into AutoGen's internal format
img
```

**Creating a multi-modal message:**

```python
# MultiModalMessage accepts a list mixing text strings and AGImage objects
# The LLM receives both the text instruction and the image for vision analysis
multi_modal_message = MultiModalMessage(
    content=["Describe the content of this image in detail", img],
    source="User"
)
```

**Sending to a vision-capable agent:**

```python
# Standard AssistantAgent — no special config needed for vision beyond model_info["vision"]=True
describer = AssistantAgent(
    name="description_agent",
    model_client=model_client,
    system_message="You are good at describing images",
)

response = await describer.on_messages([multi_modal_message], cancellation_token=CancellationToken())
reply = response.chat_message.content
display(Markdown(reply))
```

The model correctly identifies visual elements, text within images (like "AI" written on a doorway), artistic style, and the conceptual message being conveyed — demonstrating strong vision capabilities through the same simple `on_messages()` interface.

---

### Structured Outputs with Pydantic

AutoGen makes structured outputs trivial — define a Pydantic model and pass it as `output_content_type`. The response is automatically parsed into a typed Python object.

```python
from pydantic import BaseModel, Field
from typing import Literal

# Define the output schema as a Pydantic model
# AutoGen converts this to a JSON schema, instructs the LLM to conform to it,
# and parses the JSON response back into this object automatically
class ImageDescription(BaseModel):
    scene: str = Field(description="Briefly, the overall scene of the image")
    message: str = Field(description="The point that the image is trying to convey")
    style: str = Field(description="The artistic style of the image")
    orientation: Literal["portrait", "landscape", "square"] = Field(description="The orientation of the image")
```

```python
# output_content_type is the only change needed — everything else is identical
describer = AssistantAgent(
    name="description_agent",
    model_client=model_client,
    system_message="You are good at describing images in detail",
    output_content_type=ImageDescription,  # ← This is all it takes
)

response = await describer.on_messages([multi_modal_message], cancellation_token=CancellationToken())
reply = response.chat_message.content  # This is now an ImageDescription instance, NOT a string
reply
```

**Accessing structured fields:**

```python
import textwrap
# reply is a fully typed Pydantic object — access fields directly
print(f"Scene:\n{textwrap.fill(reply.scene)}\n\n")
print(f"Message:\n{textwrap.fill(reply.message)}\n\n")
print(f"Style:\n{textwrap.fill(reply.style)}\n\n")
print(f"Orientation:\n{textwrap.fill(reply.orientation)}\n\n")
```

**What happens behind the scenes:**
1. AutoGen converts the Pydantic model → JSON Schema
2. The schema is sent to the LLM as a structured output constraint
3. The LLM returns conforming JSON
4. AutoGen parses the JSON → Pydantic object

This is ideal for downstream processing: writing to databases, rendering in UIs, or feeding into other typed systems.

---

### Using LangChain Tools from AutoGen

AutoGen provides `LangChainToolAdapter` — a bridge that wraps any LangChain tool into an AutoGen-compatible tool. This gives you access to LangChain's **massive tool ecosystem** without leaving AutoGen.

```python
# AutoGen's bridge class — wraps LangChain tools for use in AutoGen agents
from autogen_ext.tools.langchain import LangChainToolAdapter

# LangChain tools we want to use:
from langchain_community.utilities import GoogleSerperAPIWrapper  # Web search via Serper API
from langchain_community.agent_toolkits import FileManagementToolkit  # Read/write/list/move files
from langchain.agents import Tool  # Generic LangChain tool wrapper
```

**Wrapping LangChain tools for AutoGen:**

```python
prompt = """Your task is to find a one-way non-stop flight from JFK to LHR in June 2026.
First search online for promising deals.
Next, write all the deals to a file called flights.md with full details.
Finally, select the one you think is best and reply with a short summary.
Reply with the selected flight only, and only after you have written the details to the file."""

# Step 1: Create a LangChain tool (identical to what we'd use in LangGraph)
serper = GoogleSerperAPIWrapper()
langchain_serper = Tool(
    name="internet_search",
    func=serper.run,
    description="useful for when you need to search the internet"
)

# Step 2: Wrap it with LangChainToolAdapter → becomes an AutoGen tool
autogen_serper = LangChainToolAdapter(langchain_serper)
autogen_tools = [autogen_serper]

# Step 3: Wrap the FileManagementToolkit tools (read, write, list, move, copy, delete)
# root_dir sandboxes all file operations to the "sandbox" directory for safety
langchain_file_management_tools = FileManagementToolkit(root_dir="sandbox").get_tools()
for tool in langchain_file_management_tools:
    autogen_tools.append(LangChainToolAdapter(tool))  # Each one gets wrapped

# Inspect what tools the agent will have
for tool in autogen_tools:
    print(tool.name, tool.description)
```

**Running the agent with LangChain tools:**

```python
# Create agent with the adapted tools — same interface as native AutoGen tools
agent = AssistantAgent(
    name="searcher",
    model_client=model_client,
    tools=autogen_tools,          # Mix of adapted LangChain tools
    reflect_on_tool_use=True      # Generate natural language response after tool execution
)

message = TextMessage(content=prompt, source="user")
result = await agent.on_messages([message], cancellation_token=CancellationToken())

# Trace the tool calls
for message in result.inner_messages:
    print(message.content)
display(Markdown(result.chat_message.content))
```

**Continuing the conversation (multi-turn):**

```python
# Agent retains context — send follow-up to trigger file-writing step
message = TextMessage(content="OK proceed", source="user")

result = await agent.on_messages([message], cancellation_token=CancellationToken())
for message in result.inner_messages:
    print(message.content)
display(Markdown(result.chat_message.content))
```

The agent searches the web, writes results to `sandbox/flights.md`, and selects the best option — all using LangChain tools transparently through AutoGen's adapter layer.

**Key insight:** Any tool from the LangChain ecosystem (Python REPL, Wolfram Alpha, Wikipedia, SQL databases, etc.) can be used in AutoGen with a single `LangChainToolAdapter()` wrapper. No rewriting needed.

---

### Teams: Multi-Agent Collaboration

Teams are AutoGen's equivalent of CrewAI's crews — multiple agents collaborating on a task. The simplest pattern is `RoundRobinGroupChat`: agents take turns in sequence until a termination condition is met.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination  # Stops when a word appears
from autogen_agentchat.teams import RoundRobinGroupChat          # Round-robin turn-taking

from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool

# Set up the search tool
serper = GoogleSerperAPIWrapper()
langchain_serper = Tool(name="internet_search", func=serper.run, description="useful for when you need to search the internet")
autogen_serper = LangChainToolAdapter(langchain_serper)

prompt = """Find a one-way non-stop flight from JFK to LHR in June 2026."""
```

**Defining the team members:**

```python
# Primary agent: the researcher — has tools, does the actual work
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    tools=[autogen_serper],
    system_message="You are a helpful AI research assistant who looks for promising deals on flights. Incorporate any feedback you receive.",
)

# Evaluator agent: the critic — no tools, just reviews and provides feedback
# Says "APPROVE" when satisfied, which triggers termination
evaluation_agent = AssistantAgent(
    "evaluator",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedback is addressed.",
)
```

**Configuring the team:**

```python
# TextMentionTermination: stops the loop when "APPROVE" appears in any message
# This is simple but somewhat brittle — in production, prefer structured output checks
text_termination = TextMentionTermination("APPROVE")

# RoundRobinGroupChat: primary → evaluator → primary → evaluator → ...
# max_turns prevents infinite loops if the evaluator never approves
team = RoundRobinGroupChat(
    [primary_agent, evaluation_agent],
    termination_condition=text_termination,
    max_turns=20  # Safety valve — prevents runaway token consumption
)
```

**Running the team:**

```python
# team.run() orchestrates the full multi-agent conversation
# Unlike on_messages() (single agent), run() manages the back-and-forth
result = await team.run(task=prompt)
for message in result.messages:
    print(f"{message.source}:\n{message.content}\n\n")
```

**Typical execution flow:**
1. **User** → "Find a one-way non-stop flight from JFK to LHR"
2. **Primary** → searches the web, returns flight options
3. **Evaluator** → "Good information but lacks specifics. Focus on dates, organize better."
4. **Primary** → "Thank you for the feedback. Here's a revised response..." (improved)
5. **Evaluator** → "APPROVE" → termination condition met, loop ends

**⚠️ Important caveats about teams:**

- **Runaway conversations**: Unlike LangGraph (which has a recursion limit of 25), AutoGen has no built-in limit. Agents can loop indefinitely, consuming tokens. Always set `max_turns`.
- **Brittle termination**: Relying on the LLM to output an exact string ("APPROVE") is fragile. In production, use structured outputs or more robust termination logic.
- **Prompt sensitivity**: Small changes in system messages can dramatically affect how many turns the conversation takes (observed: same setup taking 10 seconds one run, >60 seconds the next).

---

### MCP Tools Preview

**MCP (Model Context Protocol)** is Anthropic's open standard for packaging tools so any LLM can discover and use them. Think of it as:

- **LangChain tools**: Great ecosystem, but tied to the LangChain/LangGraph ecosystem
- **MCP tools**: Open standard — anyone can write tools, any framework can consume them

Anthropic describes MCP as the **"USB-C connector for AI"** — a universal protocol for plugging models into tools and resources (like RAG context).

AutoGen provides `mcp_server_tools()` to consume MCP tools just as easily as LangChain tools:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

# StdioServerParams defines how to launch an MCP server as a local subprocess
# mcp-server-fetch: an open-source MCP tool that runs Playwright headlessly to fetch web pages
fetch_mcp_server = StdioServerParams(
    command="uvx",                      # uv's tool runner (like npx for Python)
    args=["mcp-server-fetch"],          # The MCP server package to run
    read_timeout_seconds=30             # Timeout for server responses
)

# Connect to the MCP server and retrieve its tools as AutoGen-compatible objects
# This is analogous to LangChainToolAdapter but for the MCP ecosystem
fetcher = await mcp_server_tools(fetch_mcp_server)

# Use MCP tools exactly like native tools or LangChain-adapted tools
agent = AssistantAgent(
    name="fetcher",
    model_client=model_client,
    tools=fetcher,              # MCP tools drop in seamlessly
    reflect_on_tool_use=True
)

# agent.run() is a convenience method — handles the full task lifecycle
# (alternative to on_messages() for simpler single-task invocations)
result = await agent.run(task="Review https://www.srajdev.com/p/understanding-agentic-ai-architecture and summarize what you learn. Reply in Markdown.")
display(Markdown(result.messages[-1].content))

```

**What's happening:**
1. `uvx mcp-server-fetch` launches a local MCP server process
2. The server exposes a `fetch` tool (headless Playwright browser)
3. AutoGen discovers the tool via the MCP protocol
4. The agent uses it to scrape a website and summarize the content

**Key MCP concepts (preview — full coverage in [06-MCP with OpenAI Agents SDK](../06-MCP%20with%20OpenAI%20Agents%20SDK/)):**
- MCP servers expose tools via a standardized protocol (JSON-RPC over stdio or HTTP)
- Tools are self-describing (name, description, input schema) — no manual schema writing
- Any framework can consume MCP tools: AutoGen, LangGraph, OpenAI Agents SDK, etc.
- There's a growing public ecosystem of MCP servers (file systems, databases, APIs, browsers)

> ⚠️ **Windows note**: MCP servers have known issues on Windows as of May 2026. Use WSL (Windows Subsystem for Linux) as a workaround. Mac and Linux work natively.

---

## Key Takeaways

1. **Lightweight abstraction**: AgentChat is arguably the simplest of the frameworks covered (CrewAI, OpenAI Agents SDK, LangGraph). Minimal boilerplate, no decorators for tools.

2. **OpenAI-compatible**: The `OpenAIChatCompletionClient` works with any OpenAI-compatible API (OpenRouter, Azure, local vLLM, etc.) — just override `base_url`.

3. **Unified message model**: Everything is a message — user inputs, agent responses, tool calls, tool results. `MultiModalMessage` extends this to images seamlessly.

4. **Tool registration is zero-ceremony**: Pass a typed Python function with a docstring. AutoGen handles schema generation automatically.

5. **Cross-framework interop**: `LangChainToolAdapter` gives access to the entire LangChain tool ecosystem. `mcp_server_tools()` gives access to the entire MCP ecosystem. One framework, all tools.

6. **Structured outputs are trivial**: Pass a Pydantic model as `output_content_type` — the response is automatically a typed object. No manual JSON parsing.

7. **Teams need guardrails**: `RoundRobinGroupChat` is simple but can loop indefinitely. Always set `max_turns` and prefer robust termination conditions over string matching.

8. **Async-first**: All agent invocations are coroutines (`await`). The framework is built on an event-driven architecture from the ground up.

9. **Microsoft-backed open source**: Unlike CrewAI/LangChain where commercialization may drive the roadmap, AutoGen is a Microsoft Research contribution — fully open source with no monetization pressure on the framework itself.

---

## References

- **Microsoft AutoGen GitHub**: https://github.com/microsoft/autogen
- **AutoGen Documentation (v0.4+)**: https://microsoft.github.io/autogen/
- **AG2 Fork GitHub**: https://github.com/ag2ai/ag2
- **AG2 Documentation**: https://docs.ag2.ai/
- **AutoGen v0.4 Announcement Blog**: https://www.microsoft.com/en-us/research/blog/autogen-update/
- **Magentic-One**: https://github.com/microsoft/autogen/tree/main/python/packages/magentic-one
- **OpenRouter (API provider used in labs)**: https://openrouter.ai/
- **MCP (Model Context Protocol)**: https://modelcontextprotocol.io/
- **MCP Server Fetch (Playwright-based)**: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
- **LangChain Tools Directory**: https://python.langchain.com/docs/integrations/tools/
- **Google Serper API**: https://serper.dev/
