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
- [Lab: Airline Agent with SQLite Tool Use](#lab-airline-agent-with-sqlite-tool-use)
  - [Environment Setup](#environment-setup)
  - [Creating a Model Client (OpenRouter)](#creating-a-model-client-openrouter)
  - [Alternative: Local Model with Ollama](#alternative-local-model-with-ollama)
  - [Creating a Message](#creating-a-message)
  - [Creating an Agent](#creating-an-agent)
  - [Invoking the Agent](#invoking-the-agent)
  - [Building a SQLite Tool](#building-a-sqlite-tool)
  - [Arming the Agent with Tools](#arming-the-agent-with-tools)
  - [Executing the Tool-Equipped Agent](#executing-the-tool-equipped-agent)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Introduction

AutoGen is an **open-source multi-agent framework from Microsoft Research**, first released in its current form (v0.4) in January 2025 as a ground-up rewrite adopting an **asynchronous, event-driven architecture**. The rewrite addressed criticisms of the earlier 0.2 version around observability, flexibility, control, and scalability.

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

## Lab: Airline Agent with SQLite Tool Use

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

## Key Takeaways

1. **Lightweight abstraction**: AgentChat is arguably the simplest of the frameworks covered (CrewAI, OpenAI Agents SDK, LangGraph). Minimal boilerplate, no decorators for tools.

2. **OpenAI-compatible**: The `OpenAIChatCompletionClient` works with any OpenAI-compatible API (OpenRouter, Azure, local vLLM, etc.) — just override `base_url`.

3. **Unified message model**: Everything is a message — user inputs, agent responses, tool calls, tool results. This enables full execution tracing via `inner_messages`.

4. **Tool registration is zero-ceremony**: Pass a typed Python function with a docstring. AutoGen handles schema generation automatically.

5. **Async-first**: All agent invocations are coroutines (`await`). The framework is built on an event-driven architecture from the ground up.

6. **Microsoft-backed open source**: Unlike CrewAI/LangChain where commercialization may drive the roadmap, AutoGen is a Microsoft Research contribution — fully open source with no monetization pressure on the framework itself.

---

## References

- **Microsoft AutoGen GitHub**: https://github.com/microsoft/autogen
- **AutoGen Documentation (v0.4+)**: https://microsoft.github.io/autogen/
- **AG2 Fork GitHub**: https://github.com/ag2ai/ag2
- **AG2 Documentation**: https://docs.ag2.ai/
- **AutoGen v0.4 Announcement Blog**: https://www.microsoft.com/en-us/research/blog/autogen-update/
- **Magentic-One**: https://github.com/microsoft/autogen/tree/main/python/packages/magentic-one
- **OpenRouter (API provider used in this lab)**: https://openrouter.ai/
