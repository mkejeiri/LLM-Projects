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
- [Lab 3: AutoGen Core — The Agent Interaction Runtime](#lab-3-autogen-core--the-agent-interaction-runtime)
  - [AutoGen Core vs Semantic Kernel](#autogen-core-vs-semantic-kernel)
  - [Core Philosophy: Decoupling Logic from Messaging](#core-philosophy-decoupling-logic-from-messaging)
  - [Runtimes: Standalone vs Distributed](#runtimes-standalone-vs-distributed)
  - [Defining Messages](#defining-messages)
  - [Defining Agents: RoutedAgent and Message Handlers](#defining-agents-routedagent-and-message-handlers)
  - [Creating a Runtime and Registering Agents](#creating-a-runtime-and-registering-agents)
  - [Sending Messages](#sending-messages)
  - [Delegating to an LLM: Core + AgentChat Integration](#delegating-to-an-llm-core--agentchat-integration)
  - [Multi-Agent Interaction: Rock Paper Scissors](#multi-agent-interaction-rock-paper-scissors)
- [Lab 4: AutoGen Core — Distributed Runtime](#lab-4-autogen-core--distributed-runtime)
  - [Distributed Architecture: Host + Workers](#distributed-architecture-host--workers)
  - [Setting Up the gRPC Host](#setting-up-the-grpc-host)
  - [Defining Agents (Unchanged from Standalone)](#defining-agents-unchanged-from-standalone)
  - [Mode 1: All-in-One Worker](#mode-1-all-in-one-worker)
  - [Mode 2: Multiple Workers (Truly Distributed)](#mode-2-multiple-workers-truly-distributed)
  - [The Key Insight: Transparent Distribution](#the-key-insight-transparent-distribution)
- [Lab 5: The Agent Creator — Self-Replicating Agents](#lab-5-the-agent-creator--self-replicating-agents)
  - [The Big Idea](#the-big-idea)
  - [Architecture Overview](#architecture-overview)
  - [messages.py — Shared Message Type](#messagespy--shared-message-type)
  - [agent.py — The Prototype Template](#agentpy--the-prototype-template)
  - [creator.py — The Agent That Creates Agents](#creatorpy--the-agent-that-creates-agents)
  - [world.py — The Orchestrator](#worldpy--the-orchestrator)
  - [Running It](#running-it)
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

#### Magentic-One vs Kiro CLI

| | Magentic-One | Kiro CLI |
|---|---|---|
| **Purpose** | General-purpose multi-agent system for complex tasks (web browsing, file handling, coding) | AI-assisted development agent for coding, system ops, and AWS workflows |
| **Architecture** | Orchestrator + specialist agents (WebSurfer, FileSurfer, Coder, Terminal) | Single agent with tool access (filesystem, shell, AWS, web search) |
| **When to use** | Research tasks requiring multiple autonomous agents collaborating (e.g., multi-step web research, data gathering across sources) | Day-to-day development: writing code, debugging, managing infrastructure, creating PRs |
| **Customization** | Limited — pre-built agent team, you configure but don't code agents | Extensible via specs, hooks, and context; works within your existing project |
| **Runtime** | Standalone Python process, requires AutoGen ecosystem | Terminal CLI, integrates with your shell and dev environment directly |
| **Interaction** | Task-based — give it a goal, it orchestrates autonomously | Conversational — interactive back-and-forth in your terminal |

**TL;DR:** Use Magentic-One when you need autonomous multi-agent collaboration on complex research/data tasks. Use Kiro CLI when you need an AI pair-programmer in your terminal for hands-on development work.

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

## Lab 3: AutoGen Core — The Agent Interaction Runtime

AutoGen Core is the **infrastructure layer** beneath AgentChat. It is a model-agnostic, platform-agnostic **agent interaction framework** — it doesn't care how you implement your agents, only how they communicate. You can use AgentChat agents, raw OpenAI calls, LangChain, or anything else as the underlying logic.

From a positioning standpoint, AutoGen Core is comparable to **LangGraph** — both orchestrate interactions between agents. The key difference in emphasis:

| | AutoGen Core | LangGraph |
|---|---|---|
| **Primary concern** | Messaging & agent lifecycle across diverse, potentially distributed agents | Robustness, repeatability, and state replay |
| **Agent diversity** | Agents can be written in different languages, use different frameworks | Agents are typically LangGraph nodes |
| **Architecture** | Event-driven message dispatch | Directed graph with state machine semantics |

### AutoGen Core vs Semantic Kernel

A common question: where does **Microsoft Semantic Kernel** fit?

Semantic Kernel is **not** an agentic framework in the same sense. It's more analogous to **LangChain** — heavyweight glue code for:
- Wrapping LLM calls
- Memory management
- Tool calling (called "plugins" in SK)
- Prompt templating
- Structured outputs

There is overlap (SK has some agent functionality), but Microsoft positions them differently:
- **AutoGen**: Exclusively focused on building autonomous multi-agent applications
- **Semantic Kernel**: Stitching together LLM calls for business applications

### Core Philosophy: Decoupling Logic from Messaging

The fundamental principle of AutoGen Core:

> **The framework handles agent creation, lifecycle, and message delivery. You handle the agent logic.**

AutoGen Core manages:
- **Creating agents** — the full lifecycle from instantiation to teardown
- **Routing messages** — dispatching messages to the correct agent and handler based on type signatures
- **Discovery** — agents can find and message other agents by their ID

What it does **not** manage:
- What your agent actually does (call an LLM, query a database, return a hardcoded string — it doesn't care)

### Runtimes: Standalone vs Distributed

The **Runtime** is the execution environment where agents live and interact. Two types:

1. **`SingleThreadedAgentRuntime`** (Standalone) — runs locally on your machine, single-threaded. Used for development and simple deployments.
2. **Distributed Runtime** — allows remote agents across machines to interact. Covered in Lab 4.

### Defining Messages

In AutoGen Core, you define your own message types. This is analogous to defining **state** in LangGraph, but the emphasis is on communication rather than state management:

```python
import os
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core import SingleThreadedAgentRuntime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)
```

```python
# A simple dataclass to transport information between agents.
# Analogous to defining 'state' in LangGraph — but here it's a message.
# You can make this more sophisticated with multiple fields
# (e.g., images, metadata, structured data).

@dataclass
class Message:
    content: str
```

The message is a plain Python dataclass. You can have multiple message types — AutoGen Core uses the **type signature** of handler methods to route different message types to different handlers automatically.

### Defining Agents: RoutedAgent and Message Handlers

Every AutoGen Core agent is a subclass of `RoutedAgent`. Each agent has a unique **Agent ID** with two components:

- `agent.id.type` — the kind of agent (e.g., `"simple_agent"`, `"LLMAgent"`)
- `agent.id.key` — a unique instance identifier (e.g., `"default"`)

The combination `(type, key)` is globally unique within a runtime. In a distributed runtime, agents from anywhere in the world can be addressed by their type + key.

```python
# SimpleAgent: a RoutedAgent subclass — the basic unit in AutoGen Core.
# It has an agent ID (type + key) for unique identification in the runtime.
# @message_handler decorator registers this method to receive Message objects.
# AutoGen Core routes messages to the correct handler based on the message type signature.

class SimpleAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Simple")

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(content=f"This is {self.id.type}-{self.id.key}. You said '{message.content}' and I disagree.")
```

**Key points:**
- The `@message_handler` decorator is what makes a method eligible to receive messages
- The method name is irrelevant — what matters is the **type annotation** of the `message` parameter
- You can have **multiple handlers** with different message types (e.g., `TextMessage`, `ImageMessage`) and AutoGen Core will dispatch to the correct one automatically
- All handlers are **async coroutines** — the framework is async-first
- Handlers receive a `MessageContext` providing cancellation tokens and metadata

### Creating a Runtime and Registering Agents

```python
# Create a standalone runtime — the 'world' where agents live and interact.
# register() tells the runtime about this agent TYPE and provides a factory to instantiate it.
# This doesn't create an agent yet — just makes the type known.
runtime = SingleThreadedAgentRuntime()
await SimpleAgent.register(runtime, "simple_agent", lambda: SimpleAgent())
```

**Important distinction:** `register()` does **not** instantiate an agent. It registers a **type** with a **factory function**. The runtime will instantiate agents on-demand when messages are sent to them. This is lazy instantiation — agents are created only when needed.

### Sending Messages

```python
runtime.start()  # Start the runtime — it now processes messages in the background

# AgentId(type, key) uniquely identifies an agent in the runtime.
# send_message dispatches our Message to the agent — runtime handles routing.
agent_id = AgentId("simple_agent", "default")
response = await runtime.send_message(Message("Well hi there!"), agent_id)
print(">>>", response.content)
```

**Output:**
```
>>> This is simple_agent-default. You said 'Well hi there!' and I disagree.
```

```python
# Always stop and close the runtime before creating a new one
await runtime.stop()
await runtime.close()
```

The runtime must be explicitly stopped and closed before creating another one — this is a lifecycle requirement enforced by AutoGen Core.

### Delegating to an LLM: Core + AgentChat Integration

The real power emerges when you combine AutoGen Core (messaging infrastructure) with AgentChat (LLM abstraction). The Core agent becomes a **management wrapper** that delegates actual LLM work to an AgentChat `AssistantAgent`:

```python
# MyLLMAgent: a RoutedAgent that delegates to an AgentChat AssistantAgent.
# The AutoGen Core agent is just a 'management wrapper' — the real LLM work
# is done by the _delegate (AssistantAgent from AgentChat).
# This shows how Core and AgentChat can work together.

class MyLLMAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("LLMAgent")
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"}
        )
        self._delegate = AssistantAgent("LLMAgent", model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        print(f"{self.id.type} received message: {message.content}")
        # Convert our Core Message into an AgentChat TextMessage for the delegate
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        reply = response.chat_message.content
        print(f"{self.id.type} responded: {reply}")
        return Message(content=reply)
```

**What's happening:**
- The Core agent receives a `Message` (our custom dataclass)
- It converts it to an AgentChat `TextMessage` (the format AgentChat expects)
- It delegates to the `AssistantAgent` which calls the LLM
- It wraps the response back into our `Message` format

This pattern cleanly separates concerns: Core handles routing/lifecycle, AgentChat handles LLM interaction.

**Demonstrating agent-to-agent interaction:**

```python
runtime = SingleThreadedAgentRuntime()
await SimpleAgent.register(runtime, "simple_agent", lambda: SimpleAgent())
await MyLLMAgent.register(runtime, "LLMAgent", lambda: MyLLMAgent())

# Demonstrate agent-to-agent interaction via the runtime:
# 1. Send 'Hi there!' to LLMAgent (GPT-4o-mini responds)
# 2. Forward LLM's response to SimpleAgent (which disagrees)
# 3. Forward disagreement back to LLMAgent (to see how it handles it)
runtime.start()
response = await runtime.send_message(Message("Hi there!"), AgentId("LLMAgent", "default"))
print(">>>", response.content)
response = await runtime.send_message(Message(response.content), AgentId("simple_agent", "default"))
print(">>>", response.content)
response = await runtime.send_message(Message(response.content), AgentId("LLMAgent", "default"))
```

**Output:**
```
LLMAgent received message: Hi there!
LLMAgent responded: Hello! How can I assist you today?
>>> Hello! How can I assist you today?
>>> This is simple_agent-default. You said 'Hello! How can I assist you today?' and I disagree.
LLMAgent received message: This is simple_agent-default. You said 'Hello! How can I assist you today?' and I disagree.
LLMAgent responded: I appreciate your feedback! How would you prefer I greet you or assist you?
```

The LLM gracefully handles the disagreement — demonstrating how heterogeneous agents (one hardcoded, one LLM-backed) can interact through the same runtime.

### Multi-Agent Interaction: Rock Paper Scissors

A more complete example showing three agents collaborating: two players (backed by different LLMs) and a judge/orchestrator.

```python
from autogen_ext.models.ollama import OllamaChatCompletionClient

# Player1Agent: delegates to GPT-4o-mini via OpenRouter
# Player2Agent: delegates to local Ollama llama3.2
# Both are RoutedAgents — AutoGen Core doesn't care what LLM backs them.

class Player1Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"},
            temperature=1.0  # High temperature for randomness in choices
        )
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)

class Player2Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Local model via Ollama — demonstrates heterogeneous LLM backends
        model_client = OllamaChatCompletionClient(model="llama3.2", temperature=1.0)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)
```

**The orchestrator agent** — discovers other agents by ID and coordinates the game:

```python
JUDGE = "You are judging a game of rock, paper, scissors. The players have made these choices:\n"

# RockPaperScissorsAgent: an orchestrator agent that:
# 1. Sends instructions to player1 and player2 via self.send_message()
# 2. Collects their choices
# 3. Asks its own LLM delegate to judge the winner
# This demonstrates agents discovering and messaging other agents by ID.

class RockPaperScissorsAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"},
            temperature=1.0
        )
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        instruction = "You are playing rock, paper, scissors. Respond only with the one word, one of the following: rock, paper, or scissors."
        message = Message(content=instruction)
        # Discover and message other agents by their AgentId
        inner_1 = AgentId("player1", "default")
        inner_2 = AgentId("player2", "default")
        response1 = await self.send_message(message, inner_1)
        response2 = await self.send_message(message, inner_2)
        result = f"Player 1: {response1.content}\nPlayer 2: {response2.content}\n"
        # Use own delegate to judge the outcome
        judgement = f"{JUDGE}{result}Who wins?"
        message = TextMessage(content=judgement, source="user")
        response = await self._delegate.on_messages([message], ctx.cancellation_token)
        return Message(content=result + response.chat_message.content)
```

**Running the game:**

```python
# Register all 3 agent types with the runtime and start it.
# Each register() provides a factory lambda for instantiation.
runtime = SingleThreadedAgentRuntime()
await Player1Agent.register(runtime, "player1", lambda: Player1Agent("player1"))
await Player2Agent.register(runtime, "player2", lambda: Player2Agent("player2"))
await RockPaperScissorsAgent.register(runtime, "rock_paper_scissors", lambda: RockPaperScissorsAgent("rock_paper_scissors"))
runtime.start()

# Trigger the game: send 'go' to the RockPaperScissors orchestrator.
# It will internally message player1 & player2, collect choices, and judge.
agent_id = AgentId("rock_paper_scissors", "default")
message = Message(content="go")
response = await runtime.send_message(message, agent_id)
print(response.content)
```

**Output:**
```
Player 1: rock
Player 2: paper
Player 2 wins because paper covers rock.
```

**What this demonstrates:**
1. **Agent discovery** — the orchestrator finds players by `AgentId("player1", "default")`
2. **Heterogeneous backends** — Player 1 uses GPT-4o-mini (cloud), Player 2 uses llama3.2 (local Ollama)
3. **Intra-agent messaging** — `self.send_message()` dispatches messages to other agents within the same handler
4. **Orchestration pattern** — one agent coordinates multiple others, collects results, and synthesizes a final answer

**Commercial applications of this pattern:**
- Financial trading: agents arguing about whether an equity is a good investment
- Code review: multiple agents reviewing different aspects of code
- Research: agents gathering information from different sources and synthesizing
- Any scenario requiring diverse, autonomous agents interacting in a managed environment

> ⚠️ **Note on pub/sub**: Beyond direct messaging (`send_message`), AutoGen Core also supports a **publish/subscribe** pattern where agents subscribe to topics and receive broadcast messages. This enables fan-out communication patterns without knowing all recipients upfront.

---

## Lab 4: AutoGen Core — Distributed Runtime

Moving from the standalone `SingleThreadedAgentRuntime` to the **distributed runtime**. This is the second type of runtime in AutoGen Core — it handles messaging **across process boundaries**. Agents can be in different processes, on different machines, written in different programming languages, and AutoGen Core handles all the plumbing.

> ⚠️ **Experimental**: Microsoft states that the distributed runtime is still experimental and APIs are liable to change. This is more of an architecture preview and future direction than a production-ready system.

### Distributed Architecture: Host + Workers

The distributed runtime consists of two components:

| Component | Role |
|---|---|
| **Host Service** (`GrpcWorkerAgentRuntimeHost`) | Central coordinator. Listens on a gRPC port, routes messages between workers, manages sessions for direct messages. |
| **Worker Runtime** (`GrpcWorkerAgentRuntime`) | Holds and executes agents. Connects to the host, advertises its registered agents, handles actual code execution. |

**gRPC** (Google Remote Procedure Calls) is a cross-language protocol for calling functions across process boundaries — like REST but direct function-to-function. It's used extensively in distributed systems where interactive messaging needs to cross process boundaries.

The flow:
1. Host starts listening on a port
2. Workers connect to the host and register their agent types
3. When a message is sent to an agent, the host routes it to the correct worker
4. The worker executes the agent's handler and returns the response

### Setting Up the gRPC Host

```python
import os
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)

# Toggle: if True, all agents run in a single worker process.
# If False, each agent gets its own worker — simulating truly distributed agents.
ALL_IN_ONE_WORKER = False
```

```python
# Same Message dataclass as Lab 3 — the transport object between agents.
# In a distributed runtime, this gets serialized over gRPC between workers/processes.
# The message definition is identical whether standalone or distributed — that's the point.
@dataclass
class Message:
    content: str
```

```python
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

# The Host is the central coordinator for the distributed runtime.
# It listens on a gRPC port and routes messages between workers.
# gRPC = cross-language remote procedure calls (like REST but direct function-to-function).
# Workers connect to this host and advertise which agents they have.
# The host handles session management and message delivery across process boundaries.
host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
host.start()
```

**Setting up a web search tool** (reusing the LangChain adapter pattern from Lab 2):

```python
# Wrap a LangChain tool (Google Serper web search) for use in AutoGen agents.
serper = GoogleSerperAPIWrapper()
langchain_serper = Tool(name="internet_search", func=serper.run, description="Useful for when you need to search the internet")
autogen_serper = LangChainToolAdapter(langchain_serper)
```

**Task instructions** — a business decision scenario: should we use AutoGen for a new project?

```python
# Player1 researches pros, Player2 researches cons, Judge synthesizes a decision.
instruction1 = "To help with a decision on whether to use AutoGen in a new AI Agent project, \
please research and briefly respond with reasons in favor of choosing AutoGen; the pros of AutoGen."

instruction2 = "To help with a decision on whether to use AutoGen in a new AI Agent project, \
please research and briefly respond with reasons against choosing AutoGen; the cons of Autogen."

judge = "You must make a decision on whether to use AutoGen for a project. \
Your research team has come up with the following reasons for and against. \
Based purely on the research from your team, please respond with your decision and brief rationale."
```

### Defining Agents (Unchanged from Standalone)

This is the critical point — **the agent code is identical to the standalone version**. The agents don't know they're distributed. `self.send_message()` works the same whether messages route locally or across gRPC process boundaries.

```python
# KEY INSIGHT: This agent code is IDENTICAL to the standalone version (Lab 3).
# The agents don't know they're distributed — self.send_message() works the same
# whether messages route locally or across gRPC process boundaries.
# That transparency is the power of AutoGen Core.

class Player1Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"}
        )
        # Armed with web search tool + reflect_on_tool_use for natural language responses
        self._delegate = AssistantAgent(name, model_client=model_client, tools=[autogen_serper], reflect_on_tool_use=True)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)

class Player2Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"}
        )
        self._delegate = AssistantAgent(name, model_client=model_client, tools=[autogen_serper], reflect_on_tool_use=True)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)

class Judge(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"}
        )
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        # Dispatch research tasks to player1 (pros) and player2 (cons)
        message1 = Message(content=instruction1)
        message2 = Message(content=instruction2)
        inner_1 = AgentId("player1", "default")
        inner_2 = AgentId("player2", "default")
        response1 = await self.send_message(message1, inner_1)
        response2 = await self.send_message(message2, inner_2)
        # Compile research and ask own LLM to make a final decision
        result = f"## Pros of AutoGen:\n{response1.content}\n\n## Cons of AutoGen:\n{response2.content}\n\n"
        judgement = f"{judge}\n{result}Respond with your decision and brief explanation"
        message = TextMessage(content=judgement, source="user")
        response = await self._delegate.on_messages([message], ctx.cancellation_token)
        return Message(content=result + "\n\n## Decision:\n\n" + response.chat_message.content)
```

**Note:** Player1 and Player2 are structurally identical — you could use a single class. They're kept separate to allow swapping in different models (e.g., DeepSeek for one, GPT-4o for the other) for comparative research.

### Mode 1: All-in-One Worker

All three agents in a single worker process. Messages still route through the gRPC host, but everything is in one process:

```python
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime

if ALL_IN_ONE_WORKER:
    # Single worker holds all agents — messages still route through the host
    worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker.start()

    await Player1Agent.register(worker, "player1", lambda: Player1Agent("player1"))
    await Player2Agent.register(worker, "player2", lambda: Player2Agent("player2"))
    await Judge.register(worker, "judge", lambda: Judge("judge"))

    agent_id = AgentId("judge", "default")
```

### Mode 2: Multiple Workers (Truly Distributed)

Each agent in its own worker runtime — simulating agents on separate machines:

```python
else:
    # Each agent in its own worker — the host routes messages between them via gRPC
    worker1 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker1.start()
    await Player1Agent.register(worker1, "player1", lambda: Player1Agent("player1"))

    worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker2.start()
    await Player2Agent.register(worker2, "player2", lambda: Player2Agent("player2"))

    worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker.start()
    await Judge.register(worker, "judge", lambda: Judge("judge"))
    agent_id = AgentId("judge", "default")
```

**Running the distributed system:**

```python
# Trigger the Judge — it will dispatch to player1 & player2 across workers via gRPC,
# collect their research, and synthesize a final decision.
response = await worker.send_message(Message(content="Go!"), agent_id)
display(Markdown(response.content))
```

**Sample output:**
```markdown
## Pros of AutoGen:
- Multi-agent collaboration framework
- Scalability and flexibility
- Memory and coherent context management
- Ease of development
- Versatile applications

## Cons of AutoGen:
- Limited customization in some areas
- Performance variability
- Cost considerations
- Still experimental (distributed features)

## Decision:
Recommend using AutoGen. The pros significantly outweigh the cons...
```

**Cleanup:**

```python
# Stop all workers — in distributed mode, each worker must be stopped individually
await worker.stop()
if not ALL_IN_ONE_WORKER:
    await worker1.stop()
    await worker2.stop()

# Stop the gRPC host — shuts down the message routing infrastructure
await host.stop()
```

### The Key Insight: Transparent Distribution

The most powerful aspect of AutoGen Core's distributed runtime:

1. **Zero code changes** — the agent class definitions are identical between standalone and distributed. You don't add any distributed-specific code.
2. **`self.send_message()` is transparent** — whether it calls a local function or makes a gRPC call across the network, the API is the same.
3. **Configuration, not code** — the only difference is *how you register agents* (with a `SingleThreadedAgentRuntime` vs a `GrpcWorkerAgentRuntime`).
4. **Cross-language potential** — because gRPC is language-agnostic, agents could theoretically be written in JavaScript, Go, Rust, etc. and still participate in the same runtime.

**Microsoft's vision:** A future where potentially millions of agents interact globally. AutoGen Core provides the "playpen" — a world where agents can live and interact regardless of where they are or what language they're written in. You just wrap your agent logic in a `RoutedAgent`, define your `Message` types, and the framework handles everything else.

---

## Lab 5: The Agent Creator — Self-Replicating Agents

The week 5 capstone project: an agent that **writes, imports, registers, and messages new agents** at runtime. This demonstrates AutoGen Core's dynamic agent lifecycle management in a distributed runtime.

> ⚠️ **Safety warning**: This project generates and executes Python code without guardrails. Run at your own risk. The generated agents could theoretically change model requirements or include unexpected logic. Cost is minimal (~$0.02 for 20 agents with GPT-4o-mini).

### The Big Idea

| Aspect | Details |
|---|---|
| **Educational** | Demonstrates dynamic agent creation, `importlib`, async Python, inter-agent messaging |
| **Entertaining** | Self-replicating agents that collaborate on business ideas |
| **Commercial angle** | The spawned agents brainstorm Agentic AI business ideas and refine each other's work |
| **Risks** | Unreliable (LLM-generated code may fail), unsafe (executes generated code natively) |

**Flow:**
1. `world.py` launches a distributed runtime and registers a **Creator** agent
2. Creator receives requests like `"agent1.py"`, `"agent2.py"`, etc.
3. For each: Creator reads `agent.py` as a template → asks LLM to generate a variation → saves it → dynamically imports it → registers it with the runtime → sends it `"Give me an idea"`
4. Each spawned agent generates a business idea and may forward it to a random peer for refinement
5. Results are saved as `idea1.md`, `idea2.md`, etc.

### Architecture Overview

```
world.py (orchestrator, not an agent)
  └── Creator agent (writes + registers new agents)
        ├── agent1.py (spawned, unique personality)
        ├── agent2.py (spawned, different interests)
        ├── agent3.py (spawned, may message agent1 for refinement)
        └── ... up to agentN.py
```

All files: [`code/`](code/)

### messages.py — Shared Message Type

Separated into its own module to minimize tokens when the Creator feeds the template to the LLM.

```python
# messages.py — Shared message type and utility for finding peer agents.
from dataclasses import dataclass
from autogen_core import AgentId
import glob
import os
import random

@dataclass
class Message:
    content: str

def find_recipient() -> AgentId:
    """Find a random peer agent by scanning for agentN.py files in the directory."""
    try:
        agent_files = glob.glob("agent*.py")
        agent_names = [os.path.splitext(file)[0] for file in agent_files]
        agent_names.remove("agent")  # Remove the template itself
        agent_name = random.choice(agent_names)
        print(f"Selecting agent for refinement: {agent_name}")
        return AgentId(agent_name, "default")
    except Exception as e:
        print(f"Exception finding recipient: {e}")
        return AgentId("agent1", "default")
```

### agent.py — The Prototype Template

This is the **clone template** — the Creator reads this file and asks the LLM to generate variations. Each variation gets a unique `system_message` reflecting different interests, sectors, and personality traits.

```python
# agent.py — The TEMPLATE agent that gets cloned by the Creator.
import os
from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a creative entrepreneur. Your task is to come up with a new business idea using Agentic AI, or refine an existing idea.
    Your personal interests are in these sectors: Healthcare, Education.
    You are drawn to ideas that involve disruption.
    You are optimistic, adventurous and have risk appetite.
    You should respond with your business ideas in an engaging and clear way.
    """

    # Probability that this agent forwards its idea to a peer for refinement
    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"},
            temperature=0.7
        )
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        # With some probability, forward the idea to a random peer for refinement
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my business idea. Please refine it and make it better. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)
```

### creator.py — The Agent That Creates Agents

The most interesting piece — uses `importlib` to dynamically import LLM-generated Python modules and register them with the runtime at execution time.

```python
# creator.py — The Agent Creator: writes, imports, and registers NEW agents.
import os
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, TRACE_LOGGER_NAME
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import importlib
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)

class Creator(RoutedAgent):

    system_message = """
    You are an Agent that is able to create new AI Agents.
    You receive a template in the form of Python code that creates an Agent using Autogen Core and Autogen Agentchat.
    You should use this template to create a new Agent with a unique system message.
    The class must be named Agent, inherit from RoutedAgent, and have an __init__ that takes a name parameter.
    Respond only with the python code, no other text, and no markdown code blocks.
    """

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model_info={"vision": True, "function_calling": True, "json_output": True, "structured_output": True, "family": "unknown"},
            temperature=1.0  # High creativity for diverse agent personalities
        )
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    def get_user_prompt(self):
        """Read agent.py template and build the generation prompt."""
        prompt = "Please generate a new Agent based strictly on this template.\n\nHere is the template:\n\n"
        with open("agent.py", "r", encoding="utf-8") as f:
            template = f.read()
        return prompt + template

    @message_handler
    async def handle_my_message_type(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        filename = message.content  # e.g. "agent1.py"
        agent_name = filename.split(".")[0]
        # Ask LLM to generate a new agent variation
        text_message = TextMessage(content=self.get_user_prompt(), source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        # Save generated code to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.chat_message.content)
        print(f"** Creator has created agent {agent_name} - registering with Runtime")
        # Dynamically import the module and register it with the distributed runtime
        module = importlib.import_module(agent_name)
        await module.Agent.register(self.runtime, agent_name, lambda: module.Agent(agent_name))
        # Send the new agent its first message
        result = await self.send_message(messages.Message(content="Give me an idea"), AgentId(agent_name, "default"))
        return messages.Message(content=result.content)
```

**Key techniques:**
- `importlib.import_module(agent_name)` — dynamically imports LLM-generated Python at runtime
- `module.Agent.register(self.runtime, ...)` — registers the new agent type with the distributed runtime
- `self.send_message(...)` — immediately messages the newly created agent

### world.py — The Orchestrator

Not an agent — a plain Python script that launches the runtime and uses `asyncio.gather()` to spawn all agents in parallel:

```python
# world.py — Launches the distributed runtime and spawns N agents concurrently.
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost, GrpcWorkerAgentRuntime
from agent import Agent
from creator import Creator
from autogen_core import AgentId
import messages
import asyncio

HOW_MANY_AGENTS = 20

async def create_and_message(worker, creator_id, i: int):
    """Ask Creator to spawn agentN, save the resulting idea."""
    try:
        result = await worker.send_message(messages.Message(content=f"agent{i}.py"), creator_id)
        with open(f"idea{i}.md", "w") as f:
            f.write(result.content)
    except Exception as e:
        print(f"Failed to run worker {i} due to exception: {e}")

async def main():
    host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
    host.start()
    worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker.start()
    await Creator.register(worker, "Creator", lambda: Creator("Creator"))
    creator_id = AgentId("Creator", "default")
    # Launch all creations in parallel — asyncio event loop handles concurrency
    # (not threads — coroutines yield during network I/O waits)
    coroutines = [create_and_message(worker, creator_id, i) for i in range(1, HOW_MANY_AGENTS + 1)]
    await asyncio.gather(*coroutines)
    await worker.stop()
    await host.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

**Why `asyncio.gather`?** Without it, agents would be created serially (slow). With it, all 20 creation requests fire concurrently — while one waits on OpenAI's network response, others can proceed. This is event-loop concurrency, not multithreading.

### Running It

```bash
cd code/
uv run world.py
# or: python world.py
```

**What happens:**
- 20 `agentN.py` files appear (each with unique personalities — fintech, gaming, healthcare, etc.)
- 20 `ideaN.md` files appear with business ideas, some refined by peer agents
- Console shows agents being created, receiving messages, and forwarding ideas to each other

**Challenge (from the course):** Make the Creator able to write a new version of *itself* — a self-replicating creator that can spawn creators that spawn agents. Meta-recursion.

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

10. **Core decouples logic from messaging**: AutoGen Core is the infrastructure layer — it handles agent lifecycle, message routing, and discovery. Your agents can use any LLM, any framework, or no LLM at all. Core doesn't care.

11. **Type-based message dispatch**: AutoGen Core routes messages to handlers based on the Python type annotation of the `message` parameter — enabling multiple handlers per agent for different message types.

12. **Heterogeneous agent ecosystems**: A single runtime can host agents backed by different LLMs (cloud APIs, local Ollama), different frameworks, or even different programming languages (in distributed mode).

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
