# CrewAI Framework

## Table of Contents

1. [Overview & Context](#overview--context)
2. [CrewAI Product Ecosystem](#crewai-product-ecosystem)
3. [The Open Source Framework](#the-open-source-framework)
4. [Crews vs Flows](#crews-vs-flows)
5. [Core Architecture & Concepts](#core-architecture--concepts)
6. [Project Structure](#project-structure)
7. [YAML Configuration (Low-Code Approach)](#yaml-configuration-low-code-approach)
8. [The Crew Class (Python Orchestration)](#the-crew-class-python-orchestration)
9. [Tools in CrewAI](#tools-in-crewai)
10. [Structured Outputs with Pydantic](#structured-outputs-with-pydantic)
11. [Memory System](#memory-system)
12. [Process Modes: Sequential vs Hierarchical](#process-modes-sequential-vs-hierarchical)
13. [Multi-Model Support (LiteLLM)](#multi-model-support)
14. [Code Execution in Agents](#code-execution-in-agents)
15. [Task Context & Dependencies](#task-context--dependencies)
16. [Complete Implementation Examples](#complete-implementation-examples)
17. [Running a Crew](#running-a-crew)
18. [Key Takeaways](#key-takeaways)

---

## Overview & Context

CrewAI is introduced in the agentic AI course following the OpenAI Agents SDK. The transition highlights a recurring learning pattern: getting comfortable with one framework, then moving to the next while noting differences, similarities, strengths, and weaknesses.

**Core Philosophy**: Different projects lend themselves to different frameworks. Engineers should evaluate each and make their own determination about which best fits their use case.

**Position in Framework Hierarchy**:
- CrewAI sits in the **Mid Tier** — easy-to-use with a low-code YAML configuration approach
- Simpler than LangGraph (computational graphs) or AutoGen (ecosystem-heavy)
- More structured than OpenAI Agents SDK (lightweight, direct)
- More abstracted than raw API calls (no framework)

---

## CrewAI Product Ecosystem

CrewAI is actually **three different products**:

| Product | Description | Target |
|---------|-------------|--------|
| **CrewAI Enterprise** | Platform for deploying, running, monitoring, and managing agents through UI screens | Production teams needing hosted agent management |
| **CrewAI Studio** | Low-code/no-code platform for piecing together agent interactions | End users building agents visually |
| **CrewAI Framework** | Open source framework for "orchestrating high-performing AI agents with ease and scale" | Developers writing agent code |

### Monetization Context

Unlike OpenAI and Anthropic (who monetize through their models), CrewAI needs alternative revenue streams. The open source framework drives adoption, while CrewAI Enterprise provides the paid hosting/deployment platform. This means their website has significant upselling toward the enterprise platform.

---

## The Open Source Framework

The focus is exclusively on the **open source framework** — writing code to build agents directly, not using low-code tooling or paying for a hosting platform.

### Two Flavors of the Framework

#### 1. CrewAI Crews

- **Concept**: Autonomous solutions with teams of agents working together
- "Crew" = CrewAI's term for a team/group of agents
- Agents have different roles and collaborate autonomously

**Best for**:
- Autonomous problem solving
- Creative collaboration
- Exploratory tasks

#### 2. CrewAI Flows

- **Concept**: Prescribed, fixed workflows with defined steps, decision points, and outcomes
- Likely a newer addition to the framework (more prominent in recent documentation)
- Emerged from production concerns about running fully autonomous crews

**Best for**:
- Deterministic outcomes
- Auditability requirements
- Precise control over execution

---

## Crews vs Flows

CrewAI Crews refers to a team of agents and tasks working together, structured either in a **sequential** or **hierarchical** manner.

CrewAI Flows enable a more dynamic processing approach, focusing on data flow and automation between tasks.

- **Sequential mode**: Tasks execute one after another
- **Hierarchical mode**: A manager LLM assigns tasks to agents dynamically

| Aspect | Crews | Flows |
|--------|-------|-------|
| **Autonomy** | High — agents choose their own path | Low — predefined workflow steps |
| **Predictability** | Variable | High |
| **Auditability** | Challenging | Built-in |
| **Use Case** | Exploratory, creative | Production, compliance |
| **Complexity** | Higher uncertainty | More straightforward |

---

## Core Architecture & Concepts

CrewAI's architecture revolves around four core primitives:

| Primitive | Description |
|-----------|-------------|
| **Agent** | An autonomous unit with a role, goal, backstory, and assigned LLM. Can have memory and tools. |
| **Task** | A specific assignment carried out by an agent. Has a description, expected output, and assigned agent. |
| **Tool** | A capability given to an agent (search, code execution, custom functions) |
| **Crew** | The aggregate of agents and tasks — the orchestrator that assembles them into a runnable pipeline |

**Key relationship**: A task is assigned to an agent. There can be multiple tasks for a single agent. A crew is simply the team of agents and tasks combined.

### CrewAI is More Opinionated than OpenAI Agents SDK

A critical distinction: CrewAI is **more prescriptive** about how you define agents.

In OpenAI Agents SDK, an agent just had `instructions` — a single system prompt field. You could structure it however you wanted. Very unopinionated.

In CrewAI, an agent has three mandatory fields: **role**, **goal**, and **backstory**. This forces you to think in structured terms about the agent's persona.

**The trade-off**:
- **Benefit**: Forces good prompting practices — thinking about context, persona, and objectives separately
- **Cost**: Less control over the system prompt. How role/goal/backstory get constituted into the actual system prompt is hidden from you. If you need to debug what's happening at the prompt level, you don't have direct access. You'd have to dig into CrewAI internals or use custom prompt templates (an advanced feature).

**Why backstory works (the scientific view)**: LLMs predict the most likely next token given input context. If the backstory says "you're a fair judge who weighs arguments without personal bias," then at inference time, the output tokens will be biased toward patterns seen during training that followed similar context. The backstory increases the probability of generating tokens consistent with that persona.

### How It Differs from OpenAI Agents SDK

| Aspect | OpenAI Agents SDK | CrewAI |
|--------|-------------------|--------|
| **Configuration** | Python code | YAML + Python decorators |
| **Agent Definition** | Single `instructions` field (unopinionated) | Structured `role` + `goal` + `backstory` (opinionated) |
| **Task Concept** | No explicit analog — implicit in agent instructions | First-class primitive with description, expected output, assigned agent |
| **Multi-Model** | OpenAI only | Any model via LiteLLM (OpenAI, Anthropic, Gemini, Groq, Ollama, OpenRouter) |
| **Execution** | Direct function calls | `crew.kickoff(inputs)` |
| **Code Execution** | Custom tool | Built-in with Docker sandboxing |
| **Project Structure** | Free-form Python | Mandatory directory scaffolding via `crewai create crew` |

### LiteLLM Under the Hood

CrewAI uses **LiteLLM** as its model abstraction layer. LiteLLM is extremely lightweight — almost nothing there — you just pass a model name and it connects to any provider.

Model string format: `provider/model-name`

```yaml
# Examples of LLM specifications in agent configs
agent_1:
  llm: gpt-4o-mini              # OpenAI (default provider, can omit "openai/")
  
agent_2:
  llm: openai/gpt-4o            # OpenAI (explicit)

agent_3:
  llm: anthropic/claude-sonnet-4-5  # Anthropic Claude

agent_4:
  llm: gemini/gemini-2.5-flash     # Google Gemini

agent_5:
  llm: groq/llama2-70b-4096     # Groq

agent_6:
  llm: ollama/llama2             # Local Ollama (requires base_url config)

agent_7:
  llm: openrouter/model-name    # OpenRouter (abstraction service)
```

**This is a significant advantage over OpenAI Agents SDK** — truly flexible, lightweight model switching. You can mix and match providers within a single crew.

---

## Project Structure

### Creating a New Crew Project

CrewAI doesn't work in notebooks. You need proper Python modules with a specific directory structure. CrewAI builds an entire project for each crew.

```bash
# Install the CrewAI framework (one-time)
uv tool install crewai

# Create a new crew project
crewai create crew debate

# Or for a flow-based project:
crewai create flow my_project
```

When you run `crewai create crew debate`, it will:
1. Ask which model to start with (e.g., OpenAI → GPT-4o-mini)
2. Ask for an API key (press Enter if you already have a `.env` file)
3. Generate the full directory scaffolding with example code

**Important**: This creates a **uv project** under the hood. CrewAI uses uv for dependency management, so you'll see `pyproject.toml` and `uv.lock` files.

### Generated Directory Layout

```
debate/                         # Project root (also a uv project)
├── pyproject.toml              # Dependencies and entry points
├── uv.lock                     # Locked dependencies
├── knowledge/                  # Knowledge files (RAG context for agents)
│   └── user_preference.txt    # Example scaffolding - user background info
├── output/                     # Task output files (created at runtime)
├── src/
│   └── debate/                # Package named after your project
│       ├── __init__.py
│       ├── main.py            # Entry point - defines inputs and kicks off crew
│       ├── crew.py            # Crew class - assembles agents, tasks, crew
│       ├── config/
│       │   ├── agents.yaml    # Agent definitions (role, goal, backstory, llm)
│       │   └── tasks.yaml     # Task definitions (description, expected_output, agent)
│       └── tools/
│           ├── __init__.py
│           └── custom_tool.py # Custom tool implementations (scaffolding)
```

**Explorer note**: In VS Code/Cursor, if a directory only has one subdirectory, they collapse into one line (e.g., `src/debate/config`). This can be confusing when navigating.

### The Key Files

| File | Purpose |
|------|---------|
| `config/agents.yaml` | Define agents: role, goal, backstory, llm |
| `config/tasks.yaml` | Define tasks: description, expected_output, agent, output_file |
| `crew.py` | Bring it all together with decorators — the most important module |
| `main.py` | Set template variable values and kick off the crew |

### pyproject.toml

```toml
[project]
name = "coder"
version = "0.1.0"
description = "coder using crewAI"
requires-python = ">=3.10,<3.13"
dependencies = [
    "crewai[tools]>=0.108.0,<1.0.0",
]

[project.scripts]
coder = "coder.main:run"
run_crew = "coder.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.crewai]
type = "crew"
```

**Key points**:
- `crewai[tools]` installs the framework with built-in tools (SerperDevTool, etc.)
- Entry points defined under `[project.scripts]` allow running via `uv run coder`
- `[tool.crewai]` marks this as a crew-type project

---

## YAML Configuration (Low-Code Approach)

CrewAI's distinctive feature is **YAML-driven agent and task configuration**. This separates the "what" (roles, goals, task descriptions) from the "how" (Python orchestration logic).

**Why YAML?** The benefit is that your prompts aren't buried within your code all over the place. You've got them nicely separated out and can work on them independently from your Python logic. The con is that it's scaffolding specific to CrewAI — something to get used to.

**Alternative**: You can also create agents and tasks purely in Python code (e.g., `agent = Agent(role="...", goal="...", backstory="...")`). But the YAML approach is the idiomatic CrewAI way.

### agents.yaml

```yaml
debater:
  role: >
    A compelling debater
  goal: >
    Present a clear argument either in favor of or against the motion. The motion is: {motion}
  backstory: >
    You're an experienced debater with a knack for giving concise but convincing arguments.
    The motion is: {motion}
  llm: openai/gpt-4o-mini

judge:
  role: >
    Decide the winner of the debate based on the arguments presented
  goal: >
    Given arguments for and against this motion: {motion}, decide which side is more convincing,
    based purely on the arguments presented.
  backstory: >
    You are a fair judge with a reputation for weighing up arguments without factoring in
    your own views, and making a decision based purely on the merits of the argument.
    The motion is: {motion}
  llm: anthropic/claude-sonnet-4-5
```

**Agent fields**:
- `role`: The agent's job title/function — affects how the LLM frames its responses
- `goal`: What the agent is trying to achieve — the primary directive
- `backstory`: Persona context that shapes the agent's behavior and tone (essentially the system prompt, but you don't control how it's assembled)
- `llm`: Which model to use — `provider/model-name` format via LiteLLM

**Template variables**: `{motion}` is interpolated at runtime from the `inputs` dict passed to `crew.kickoff(inputs={"motion": "..."})`. This is defined in `main.py`.

### tasks.yaml

```yaml
propose:
  description: >
    You are proposing the motion: {motion}.
    Come up with a clear argument in favor of the motion.
    Be very convincing.
  expected_output: >
    Your clear argument in favor of the motion, in a concise manner.
  agent: debater
  output_file: output/propose.md

oppose:
  description: >
    You are in opposition to the motion: {motion}.
    Come up with a clear argument against the motion.
    Be very convincing.
  expected_output: >
    Your clear argument against the motion, in a concise manner.
  agent: debater
  output_file: output/oppose.md

decide:
  description: >
    Review the arguments presented by the debaters and decide which side is more convincing.
  expected_output: >
    Your decision on which side is more convincing, and why.
  agent: judge
  output_file: output/decide.md
```

**Task fields**:
- `description`: What the task requires — the prompt for the assigned agent
- `expected_output`: Defines the format/content of the result — acts as output guardrails
- `agent`: Which agent handles this task (references agent key from agents.yaml)
- `output_file`: Automatically writes the task result to this file path
- `context`: (optional) List of prior tasks whose outputs feed into this task

**Key insight from the debate example**: The same `debater` agent handles both `propose` and `oppose` tasks. The task description determines behavior, not the agent. One agent can play multiple roles through different tasks.

**Naming constraint**: You **cannot** name a task the same as an agent, or you'll get a conflict. For example, you can't have a task called `judge` if you already have an agent called `judge`. Use names like `decide`, `propose_task`, etc.

### Multi-Agent YAML Example (Engineering Team)

```yaml
# agents.yaml
engineering_lead:
  role: >
    Engineering Lead for the engineering team, directing the work of the engineer
  goal: >
    Take the high level requirements and prepare a detailed design for the backend developer;
    everything should be in 1 python module; describe the function and method signatures.
  backstory: >
    You're a seasoned engineering lead with a knack for writing clear and concise designs.
  llm: gpt-4o

backend_engineer:
  role: >
    Python Engineer who can write code to achieve the design described by the engineering lead
  goal: >
    Write a python module that implements the design described by the engineering lead.
    The module should be named {module_name} and the class should be named {class_name}
  backstory: >
    You're a seasoned python engineer with a knack for writing clean, efficient code.
    You follow the design instructions carefully.
  llm: anthropic/claude-sonnet-4-5

frontend_engineer:
  role: >
    A Gradio expert who can write a simple frontend to demonstrate a backend
  goal: >
    Write a gradio UI that demonstrates the given backend module {module_name}.
  backstory: >
    You're a seasoned python engineer highly skilled at writing simple Gradio UIs.
  llm: anthropic/claude-sonnet-4-5
```

```yaml
# tasks.yaml
design_task:
  description: >
    Take the high level requirements and prepare a detailed design for the engineer.
    Here are the requirements: {requirements}
    IMPORTANT: Only output the design in markdown format.
  expected_output: >
    A detailed design for the engineer, identifying the classes and functions in the module.
  agent: engineering_lead
  output_file: output/{module_name}_design.md

code_task:
  description: >
    Write a python module that implements the design described by the engineering lead.
  expected_output: >
    A python module that implements the design and achieves the requirements.
    IMPORTANT: Output ONLY the raw Python code without any markdown formatting.
  agent: backend_engineer
  context:
    - design_task
  output_file: output/{module_name}

frontend_task:
  description: >
    Write a gradio UI in a module app.py that demonstrates the given backend class.
  expected_output: >
    A gradio UI module that can be run as-is, importing the backend class from {module_name}.
    IMPORTANT: Output ONLY the raw Python code.
  agent: frontend_engineer
  context:
    - code_task
  output_file: output/app.py
```

**Key insight**: The `context` field creates task dependencies — `code_task` receives the output of `design_task` as context, enabling a pipeline where each agent builds on the previous agent's work.

---

## The Crew Class (Python Orchestration)

The `crew.py` file is where everything comes together. It's the most important module — it defines your agents, tasks, and crew using decorators.

### Debate Example — crew.py

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class Debate():
    """Debate crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'],
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'],
            verbose=True
        )

    @task
    def propose(self) -> Task:
        return Task(config=self.tasks_config['propose'])

    @task
    def oppose(self) -> Task:
        return Task(config=self.tasks_config['oppose'])

    @task
    def decide(self) -> Task:
        return Task(config=self.tasks_config['decide'])

    @crew
    def crew(self) -> Crew:
        """Creates the Debate crew"""
        return Crew(
            agents=self.agents,   # Auto-populated by @agent decorators
            tasks=self.tasks,     # Auto-populated by @task decorators
            process=Process.sequential,
            verbose=True,
        )
```

### How the Decorator Pattern Works

| Decorator | Purpose |
|-----------|---------|
| `@CrewBase` | Class decorator that enables YAML config loading and auto-collection of agents/tasks |
| `@agent` | Registers the returned Agent into `self.agents` list automatically |
| `@task` | Registers the returned Task into `self.tasks` list automatically |
| `@crew` | Marks the method that assembles the final Crew object |

**Where does `self.agents` come from?** The `@agent` decorator ensures that any function decorated with it automatically adds its returned Agent to the `self.agents` instance variable. Same for `@task` → `self.tasks`. That's why you can just write `agents=self.agents` in the crew method without explicitly building a list.

**Method names must match YAML keys**: `def debater()` maps to `agents_config['debater']`, `def propose()` maps to `tasks_config['propose']`. If these don't match, you'll get KeyErrors.

**Config loading**: `agents_config = 'config/agents.yaml'` and `tasks_config = 'config/tasks.yaml'` are set as class variables. The `@CrewBase` decorator handles loading these YAML files and making them accessible as dictionaries via `self.agents_config` and `self.tasks_config`.

### Debate Example — main.py

```python
#!/usr/bin/env python
import sys
import warnings
import os

from debate.crew import Debate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
os.makedirs('output', exist_ok=True)

def run():
    """Run the crew."""
    inputs = {
        'motion': 'There needs to be strict laws to regulate LLMs'
    }
    result = Debate().crew().kickoff(inputs=inputs)
    print(result.raw)
```

**The `inputs` dict**: This is where template variables from YAML get their values. Every `{motion}` in agents.yaml and tasks.yaml gets replaced with the string from `inputs['motion']`.

**`result.raw`**: The raw text output from the final task/agent in the sequential pipeline.

**Execution flow**:
1. `Debate()` — instantiates the class, loads YAML configs
2. `.crew()` — calls the `@crew` decorated method, assembles the Crew object
3. `.kickoff(inputs=inputs)` — starts execution, interpolates template variables, runs tasks sequentially

---

## Tools in CrewAI

Tools give agents **autonomy to perform actions** at their discretion. The agent decides when to use a tool based on the task description and tool description.

### The Problem Tools Solve

Without tools, agents rely solely on their LLM's training data. This means:
- Research will be based on the model's knowledge cutoff date (e.g., "as of October 2023")
- No access to current news, prices, or real-time data
- Financial reports will contain stale information

**Solution**: Give the researcher agent a web search tool so it can look up current information.

### SerperDevTool — Google Search for Agents

CrewAI provides `SerperDevTool` as a built-in tool for web search via the Serper API.

**Setup**:
1. Sign up at [serper.dev](https://serper.dev) — you get **2500 free credits** (more than enough)
2. Get your API key from the dashboard
3. Add to your `.env` file:

```bash
SERPER_API_KEY=your_key_here
```

**Important**: The env variable must be `SERPER_API_KEY` (not `SERP_API_KEY` — there's a different service called SerpAPI). SERP = Search Engine Results Page.

**Cost comparison**: Unlike OpenAI's web search at ~2.5 cents per lookup, Serper's free tier costs nothing.

**Usage in crew.py**:

```python
from crewai_tools import SerperDevTool

@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config['researcher'],
        verbose=True,
        tools=[SerperDevTool()]  # Agent can now Google search at its discretion
    )
```

That's all it takes — one import, one line in the tools list. The agent autonomously decides when to search based on its task description. In verbose mode, you'll see output like:

```
Search the internet with Serper: "Tesla latest news 2026"
Search the internet with Serper: "Tesla Q1 2026 earnings"
```

### Custom Tools

Custom tools follow the same pattern as OpenAI function calling — Pydantic schema + execution logic:

```python
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests


class PushNotification(BaseModel):
    """Input schema for PushNotificationTool."""
    message: str = Field(..., description="The message to be sent to the user.")


class PushNotificationTool(BaseTool):
    name: str = "Send a Push Notification"
    description: str = (
        "This tool is used to send a push notification to the user."
    )
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str) -> str:
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"

        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)
        return '{"notification": "ok"}'
```

**Tool anatomy**:
- `name`: Human-readable identifier the LLM sees
- `description`: Tells the LLM when/why to use this tool — critical for tool selection
- `args_schema`: Pydantic model defining the JSON schema for arguments
- `_run()`: The actual execution logic — what happens when the LLM calls this tool

**Under the hood**: This is the same "JSON + if statement" pattern from foundations — the LLM outputs structured JSON matching `args_schema`, CrewAI routes it to `_run()`.

---

## Structured Outputs with Pydantic

CrewAI supports typed task outputs using Pydantic models. This is a powerful way to **guide agent behavior** — by defining fields with specific names and descriptions, you force the agent to produce exactly the information you need in a predictable format.

### Why Use Structured Outputs

From the transcript: "By giving the fields with that name and with that description, we are going to force the agent to produce that information in its response. It's a really clever way of making sure that we guide the agent's behavior."

**Benefits**:
- Reliable data flow between tasks (downstream tasks know exactly what format to expect)
- Output files are valid JSON (not free-form text that might need parsing)
- Acts as guardrails — the agent can't wander off-topic if it must fill specific fields

### Defining Schemas

```python
from pydantic import BaseModel, Field
from typing import List


class TrendingCompany(BaseModel):
    """A company that is in the news and attracting attention"""
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending")


# A wrapper list class - one task can produce multiple items
class TrendingCompanyList(BaseModel):
    """List of multiple trending companies"""
    companies: List[TrendingCompany] = Field(description="List of companies trending in the news")


class TrendingCompanyResearch(BaseModel):
    """Detailed research on a company"""
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability")


class TrendingCompanyResearchList(BaseModel):
    """A list of detailed research on all the companies"""
    research_list: List[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")
```

### Applying to Tasks

```python
@task
def find_trending_companies(self) -> Task:
    return Task(
        config=self.tasks_config['find_trending_companies'],
        output_pydantic=TrendingCompanyList,  # Forces structured JSON output
    )

@task
def research_trending_companies(self) -> Task:
    return Task(
        config=self.tasks_config['research_trending_companies'],
        output_pydantic=TrendingCompanyResearchList,  # Also structured
    )
```

**Output file format**: When using `output_pydantic`, the `output_file` should use `.json` extension since the output will be valid JSON conforming to the schema.

**Pro tip on terminology**: Use consistent, simple terms across your schemas, agents, and tasks. The instructor found that using inconsistent language (e.g., calling them "newsworthy companies" in one place and "trending companies" in another) caused less stable outputs. Simple, repeated terminology = more coherent responses.

---

## Memory System

CrewAI has a built-in memory system for persistent agent knowledge:

```python
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        verbose=True,
        memory=True,  # Enable memory system
        # Persistent storage across sessions
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path="./memory/long_term_memory_storage.db"
            )
        ),
        # Current context using RAG
        short_term_memory=ShortTermMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small"}
                },
                type="short_term",
                path="./memory/"
            )
        ),
        # Track key information about entities
        entity_memory=EntityMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small"}
                },
                type="short_term",
                path="./memory/"
            )
        ),
    )
```

**Memory types**:
| Type | Purpose | Storage |
|------|---------|---------|
| **Long-term** | Persistent facts across crew runs | SQLite |
| **Short-term** | Current session context | RAG (embeddings) |
| **Entity** | Track entities (people, companies, concepts) | RAG (embeddings) |

**Use case**: The stock picker uses memory so it "doesn't pick the same company twice" across multiple runs.

---

## Process Modes: Sequential vs Hierarchical

### Sequential Process

Tasks execute in order. Each task's output is available to subsequent tasks via `context`.

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        verbose=True,
    )
```

**Flow**: Task 1 → Task 2 → Task 3 (linear pipeline)

### Hierarchical Process

A **manager agent** dynamically assigns tasks to worker agents:

```python
@crew
def crew(self) -> Crew:
    manager = Agent(
        config=self.agents_config['manager'],
        allow_delegation=True
    )

    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        verbose=True,
        manager_agent=manager,
    )
```

```yaml
# Manager agent config
manager:
  role: >
    Manager
  goal: >
    You are a skilled project manager who can delegate tasks to achieve the goal.
  backstory: >
    You are an experienced and highly effective project manager.
  llm: openai/gpt-4o
```

**Key difference**: In hierarchical mode, the manager LLM decides task assignment and ordering. Worker agents may be called multiple times or in different orders based on the manager's judgment.

**Important details from the transcript**:

- The manager agent is created **separately** — it's NOT decorated with `@agent` and NOT in `self.agents`. It's passed directly to the Crew via `manager_agent=`.
- `allow_delegation=True` on the manager is the equivalent of **handoff** in OpenAI Agents SDK
- You can alternatively use `manager_llm="gpt-4o"` instead of a full agent, but defining a separate agent with role/goal/backstory performs better
- Use a **stronger model** (GPT-4o, not mini) for the manager — it needs to stay coherent with the overall mission
- Hierarchical mode is **less predictable** than sequential — the manager may assign tasks in unexpected orders, call agents multiple times, or take longer. This is both the power and the challenge of autonomous agentic AI.

---

## Multi-Model Support

CrewAI supports mixing models within a single crew via the `llm` field in agent configs:

```yaml
# Use GPT-4o for high-level planning
engineering_lead:
  llm: gpt-4o

# Use Claude for implementation (better at code generation)
backend_engineer:
  llm: anthropic/claude-sonnet-4-5

# Use GPT-4o-mini for cost-effective simple tasks
researcher:
  llm: openai/gpt-4o-mini
```

**Model string format**: `provider/model-name` or just `model-name` for OpenAI defaults.

**Strategic model selection**:
- Expensive models (gpt-4o, claude-sonnet) for complex reasoning/planning
- Cheap models (gpt-4o-mini) for straightforward tasks like research summarization
- Different models for different strengths (Claude for code, GPT for planning)

---

## Code Execution in Agents

This is where CrewAI frameworks really stand out — making something incredibly complex trivially easy. A "coder agent" (sometimes called a "code agent") is not just an agent that generates code, but one that can **write code, execute it, inspect the results, and iterate** — using code execution as a means to solve a greater problem.

### How Simple It Is

```python
@agent
def coder(self) -> Agent:
    return Agent(
        config=self.agents_config['coder'],
        verbose=True,
        allow_code_execution=True,       # That's it - agent can now run Python
        code_execution_mode="safe",      # Runs in Docker container (sandboxed)
        max_execution_time=30,           # Timeout in seconds (prevents infinite loops)
        max_retry_limit=3                # Retries if code fails (syntax/runtime errors)
    )
```

From the transcript: "It seems almost so good. I had to close down Docker to make sure that it failed when it was closed because I almost couldn't believe it was happening. It's that simple."

### Docker Sandboxing

- `code_execution_mode="safe"` runs code inside a **Docker container** — a sandboxed, ring-fenced environment with no access to your host machine
- Without this flag, code runs directly on your platform (risky for untrusted code)
- **Requirement**: [Docker Desktop](https://docs.docker.com/desktop/) must be installed and running (one-click install for Mac/Windows/Linux)

### What Happens During Execution

1. Agent receives the task (e.g., "write code to calculate this series")
2. Agent **plans** how the code will work
3. Agent **writes** the Python code
4. CrewAI **executes** it in a Docker container
5. Output is fed back to the agent
6. Agent **verifies** the output and includes it in its response
7. If execution fails, agent retries (up to `max_retry_limit`)

### Proving Real Execution

The transcript uses a clever test: calculate the first 10,000 terms of `1 - 1/3 + 1/5 - 1/7 + ...` multiplied by 4. This approximates pi, but with only 10,000 terms it gives ~3.14149 (not exact 3.14159). An LLM could predict "pi" from training data, but it can't predict the *inexact* approximation without actually running the computation. The imprecise result proves real execution happened.

---

## Task Context & Dependencies

Context is how you tell CrewAI what information should be passed from one task to another. Without context, each task runs in isolation — the analyst wouldn't see what the researcher found.

### How Context Works

The `context` field in tasks.yaml takes a list of task names. When a task runs, it receives the **full output** of all listed context tasks as part of its input.

```yaml
# Financial Researcher example:
research_task:
  description: >
    Conduct thorough research on company {company}...
  agent: researcher
  # No output_file here - output is consumed by analysis_task via context

analysis_task:
  description: >
    Analyze the research findings and create a comprehensive report on {company}...
  agent: analyst
  context:
    - research_task          # Receives the researcher's full output as context
  output_file: output/report.md
```

**Key insight from the transcript**: "The second agent that did the summary was taking advantage of the output from the first agent, because that was included in its context. And that's how it was able to give what it gave."

### Multi-Task Context (Engineering Team)

Context can create a DAG (directed acyclic graph) of dependencies:

```yaml
design_task:
  agent: engineering_lead
  output_file: output/design.md

code_task:
  agent: backend_engineer
  context:
    - design_task          # Receives design_task output as context
  output_file: output/code.py

frontend_task:
  agent: frontend_engineer
  context:
    - code_task            # Receives code_task output as context
  output_file: output/app.py

test_task:
  agent: test_engineer
  context:
    - code_task            # Also receives code_task output
  output_file: output/test_code.py
```

**Pipeline visualization**:
```
design_task → code_task → frontend_task
                       → test_task
```

### Context vs Output File

| Feature | Purpose |
|---------|---------|
| `context` | Passes task output to downstream tasks as input (in-memory) |
| `output_file` | Writes task output to disk (for human consumption or persistence) |

A task can have both, one, or neither. The research_task in the financial researcher has no `output_file` because its output is only consumed by the analysis_task — it doesn't need to be saved separately.

The `context` mechanism is how CrewAI implements the **Orchestrator-Worker** and **Prompt Chaining** patterns from agentic AI foundations.

---

## Complete Implementation Examples

### Example 1: The Debate Crew (Full Walkthrough)

This is the introductory CrewAI project from the course — a debate between LLMs judged by a different model.

**Concept**: One `debater` agent (GPT-4o-mini) argues both for and against a motion via two separate tasks. A `judge` agent (Claude) evaluates which side was more convincing. This demonstrates:
- One agent handling multiple tasks
- Multi-model mixing (OpenAI debates, Anthropic judges)
- Sequential process mode
- Template variables for dynamic content

**agents.yaml**:
```yaml
debater:
  role: >
    A compelling debater
  goal: >
    Present a clear argument either in favor of or against the motion. The motion is: {motion}
  backstory: >
    You're an experienced debater with a knack for giving concise but convincing arguments.
    The motion is: {motion}
  llm: openai/gpt-4o-mini

judge:
  role: >
    Decide the winner of the debate based on the arguments presented
  goal: >
    Given arguments for and against this motion: {motion}, decide which side is more convincing,
    based purely on the arguments presented.
  backstory: >
    You are a fair judge with a reputation for weighing up arguments without factoring in
    your own views, and making a decision based purely on the merits of the argument.
    The motion is: {motion}
  llm: anthropic/claude-sonnet-4-5
```

**tasks.yaml**:
```yaml
propose:
  description: >
    You are proposing the motion: {motion}.
    Come up with a clear argument in favor of the motion.
    Be very convincing.
  expected_output: >
    Your clear argument in favor of the motion, in a concise manner.
  agent: debater
  output_file: output/propose.md

oppose:
  description: >
    You are in opposition to the motion: {motion}.
    Come up with a clear argument against the motion.
    Be very convincing.
  expected_output: >
    Your clear argument against the motion, in a concise manner.
  agent: debater
  output_file: output/oppose.md

decide:
  description: >
    Review the arguments presented by the debaters and decide which side is more convincing.
  expected_output: >
    Your decision on which side is more convincing, and why.
  agent: judge
  output_file: output/decide.md
```

**crew.py**:
```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class Debate():
    """Debate crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def debater(self) -> Agent:
        return Agent(config=self.agents_config['debater'], verbose=True)

    @agent
    def judge(self) -> Agent:
        return Agent(config=self.agents_config['judge'], verbose=True)

    @task
    def propose(self) -> Task:
        return Task(config=self.tasks_config['propose'])

    @task
    def oppose(self) -> Task:
        return Task(config=self.tasks_config['oppose'])

    @task
    def decide(self) -> Task:
        return Task(config=self.tasks_config['decide'])

    @crew
    def crew(self) -> Crew:
        """Creates the Debate crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

**main.py**:
```python
#!/usr/bin/env python
import warnings
import os
from debate.crew import Debate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
os.makedirs('output', exist_ok=True)

def run():
    inputs = {
        'motion': 'There needs to be strict laws to regulate LLMs'
    }
    result = Debate().crew().kickoff(inputs=inputs)
    print(result.raw)
```

**Running it**:
```bash
cd debate
crewai run
```

**What happens**:
1. The debater agent (GPT-4o-mini) executes the `propose` task → writes `output/propose.md`
2. The same debater agent executes the `oppose` task → writes `output/oppose.md`
3. The judge agent (Claude) executes the `decide` task, seeing both arguments → writes `output/decide.md`
4. `result.raw` contains the judge's final decision

**Extension ideas** (from the course):
- Split the debater into two separate agents with different models (e.g., OpenAI vs DeepSeek)
- Switch who proposes and who opposes to see if it changes the outcome
- Create a leaderboard of which models are most persuasive
- Try more controversial motions

### Example 2: Financial Researcher (Tools + Context)

This is the second CrewAI project from the course — demonstrates **tools** (SerperDevTool for web search) and **context** (passing research output to the analyst).

**The problem**: Without a search tool, the researcher agent relies on its LLM's training data. Running against Tesla initially produced a report "as of October 2023" — clearly stale. Adding SerperDevTool fixed this immediately, producing current 2025 data.

**agents.yaml**:
```yaml
researcher:
  role: >
    Senior Financial Researcher for {company}
  goal: >
    Research the company, news and potential for {company}
  backstory: >
    You're a seasoned financial researcher with a talent for finding
    the most relevant information about {company}.
    Known for your ability to find the most relevant
    information and present it in a clear and concise manner.
  llm: openai/gpt-4o-mini

analyst:
  role: >
    Market Analyst and Report writer focused on {company}
  goal: >
    Analyze company {company} and create a comprehensive, well-structured report
    that presents insights in a clear and engaging way
  backstory: >
    You're a meticulous, skilled analyst with a background in financial analysis
    and company research. You have a talent for identifying patterns and extracting
    meaningful insights from research data, then communicating
    those insights through well crafted reports.
  llm: openai/gpt-4o-mini
```

**tasks.yaml**:
```yaml
research_task:
  description: >
    Conduct thorough research on company {company}. Focus on:
    1. Current company status and health
    2. Historical company performance
    3. Major challenges and opportunities
    4. Recent news and events
    5. Future outlook and potential developments

    Make sure to organize your findings in a structured format with clear sections.
  expected_output: >
    A comprehensive research document with well-organized sections covering
    all the requested aspects of {company}. Include specific facts, figures,
    and examples where relevant.
  agent: researcher

analysis_task:
  description: >
    Analyze the research findings and create a comprehensive report on {company}.
    Your report should:
    1. Begin with an executive summary
    2. Include all key information from the research
    3. Provide insightful analysis of trends and patterns
    4. Offer a market outlook for company, noting that this should not be used for trading decisions
    5. Be formatted in a professional, easy-to-read style with clear headings
  expected_output: >
    A polished, professional report on {company} that presents the research
    findings with added analysis and insights. The report should be well-structured
    with an executive summary, main sections, and conclusion.
  agent: analyst
  context:
    - research_task
  output_file: output/report.md
```

**crew.py**:
```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool  # Requires SERPER_API_KEY in .env


@CrewBase
class ResearchCrew():
    """Research crew for comprehensive topic analysis and reporting"""

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()]  # Gives agent web search capability
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            output_file='output/report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

**main.py**:
```python
import os
from financial_researcher.crew import ResearchCrew

os.makedirs('output', exist_ok=True)

def run():
    inputs = {'company': 'Tesla'}
    result = ResearchCrew().crew().kickoff(inputs=inputs)
    print(result.raw)
```

**What happens when you run it**:
1. Researcher agent (GPT-4o-mini) receives the research_task
2. Agent autonomously uses SerperDevTool to Google "Tesla latest news", "Tesla earnings", etc.
3. Agent compiles findings into a structured research document
4. Analyst agent receives the research output via `context`
5. Analyst writes a polished report → saved to `output/report.md`

**Pro tip from the transcript**: You can swap models freely. If for instance you try DeepSeek for research and Groq (Llama 3 70B) for analysis. The model flexibility via LiteLLM means you can experiment with cost/speed/quality tradeoffs per agent.

### Example 3: Coder Agent (Code Execution in Docker)

This project demonstrates the "coder agent" pattern — an agent that can **write, execute, and verify** Python code autonomously. This is where CrewAI frameworks shine: making something incredibly complex trivially easy.

**What makes this a "coder agent"**: It's not just generating code — it generates Python, runs it in a Docker container, inspects the output, and can retry if it fails. Code execution is a *means to an end*, not just the deliverable.

**agents.yaml**:
```yaml
coder:
  role: >
    Python Developer
  goal: >
    You write python code to achieve this assignment: {assignment}
    First you plan how the code will work, then you write the code, then you run it and check the output.
  backstory: >
    You're a seasoned python developer with a knack for writing clean, efficient code.
  llm: openai/gpt-4o-mini
```

**tasks.yaml**:
```yaml
coding_task:
  description: >
    Write python code to achieve this: {assignment}
  expected_output: >
    A text file that includes the code itself, along with the output of the code.
  agent: coder
  output_file: output/code_and_output.txt
```

**Key design choice**: The expected_output asks for BOTH the code AND its output. This forces the agent to actually run the code (not just generate it) and proves execution happened.

**crew.py**:
```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Docker Desktop required: https://docs.docker.com/desktop/

@CrewBase
class Coder():
    """Coder crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'],
            verbose=True,
            allow_code_execution=True,       # Enables writing + running Python
            code_execution_mode="safe",      # Sandboxed in Docker container
            max_execution_time=30,           # 30 second timeout
            max_retry_limit=3                # Retries on failure
        )

    @task
    def coding_task(self) -> Task:
        return Task(config=self.tasks_config['coding_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

**main.py**:
```python
import warnings
import os
from coder.crew import Coder

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
os.makedirs('output', exist_ok=True)

# Deliberately complex to prove real execution (not just LLM prediction)
# 10,000 terms of Leibniz series * 4 ≈ pi (but imprecise: 3.14149, not 3.14159)
assignment = 'Write a python program to calculate the first 10,000 terms \
    of this series, multiplying the total by 4: 1 - 1/3 + 1/5 - 1/7 + ...'

def run():
    inputs = {'assignment': assignment}
    result = Coder().crew().kickoff(inputs=inputs)
    print(result.raw)
```

**What happens when you run it**:
1. Agent receives the assignment
2. Agent plans the approach (recognize the alternating series pattern)
3. Agent writes Python code (e.g., using `(-1)**i / (2*i + 1)` trick)
4. CrewAI starts a Docker container and executes the code
5. If execution fails (first attempt may error), agent retries
6. Agent receives the output (3.14149...) and includes both code + output in response
7. Result saved to `output/code_and_output.txt`

**Example output**:
```python
# Agent-generated code:
def calculate_series(n_terms):
    total = 0
    for i in range(n_terms):
        term = (-1) ** i / (2 * i + 1)
        total += term
    total *= 4
    return total

result = calculate_series(10000)
print(result)

# Output: 3.1414926535900345
```

The imprecise result (3.14149... not 3.14159...) proves the code was actually executed — an LLM couldn't predict this inexact approximation from training data alone.

### Example 4: Engineering Team (Multi-Agent Collaboration, Full Software Pipeline)

This is the capstone CrewAI project — a full engineering team that designs, implements, tests, and builds a UI for a software system. It demonstrates the true power of "crew" — multiple specialized agents collaborating on a complex task.

**Why this assignment?** The requirements produce a trading account management system that gets reused in a later week of the course. This is a "two for one" — building the crew AND generating useful code for future projects.

**Key architectural decisions from the transcript**:
- Engineering lead does NOT get code execution (it only designs)
- Frontend engineer does NOT get code execution (running Gradio in Docker would be a different ball game)
- Backend engineer and test engineer DO get code execution (they need to run and verify code)
- Tasks have `IMPORTANT: Output ONLY raw Python code` — without this, LLMs output markdown code blocks with backticks, making invalid Python files
- Template variables `{module_name}` work even in `output_file` paths

**agents.yaml**:
```yaml
engineering_lead:
  role: >
    Engineering Lead for the engineering team, directing the work of the engineer
  goal: >
    Take the high level requirements and prepare a detailed design for the backend developer;
    everything should be in 1 python module; describe the function and method signatures.
    Here are the requirements: {requirements}
    The module should be named {module_name} and the class should be named {class_name}
  backstory: >
    You're a seasoned engineering lead with a knack for writing clear and concise designs.
  llm: gpt-4o

backend_engineer:
  role: >
    Python Engineer who can write code to achieve the design described by the engineering lead
  goal: >
    Write a python module that implements the design described by the engineering lead.
    Here are the requirements: {requirements}
    The module should be named {module_name} and the class should be named {class_name}
  backstory: >
    You're a seasoned python engineer with a knack for writing clean, efficient code.
    You follow the design instructions carefully.
  llm: anthropic/claude-sonnet-4-5

frontend_engineer:
  role: >
    A Gradio expert who can write a simple frontend to demonstrate a backend
  goal: >
    Write a gradio UI that demonstrates the given backend module {module_name}.
    Here are the requirements: {requirements}
  backstory: >
    You're a seasoned python engineer highly skilled at writing simple Gradio UIs.
  llm: anthropic/claude-sonnet-4-5

test_engineer:
  role: >
    An engineer who can write unit tests for the given backend module {module_name}
  goal: >
    Write unit tests for the given backend module {module_name}.
  backstory: >
    You're a seasoned QA engineer and software developer who writes great unit tests.
  llm: anthropic/claude-sonnet-4-5
```

**tasks.yaml** (key parts):
```yaml
design_task:
  description: >
    Take the high level requirements and prepare a detailed design for the engineer.
    IMPORTANT: Only output the design in markdown format.
  expected_output: >
    A detailed design identifying the classes and functions in the module.
  agent: engineering_lead
  output_file: output/{module_name}_design.md

code_task:
  description: >
    Write a python module that implements the design described by the engineering lead.
  expected_output: >
    IMPORTANT: Output ONLY the raw Python code without any markdown formatting,
    code block delimiters, or backticks.
  agent: backend_engineer
  context:
    - design_task
  output_file: output/{module_name}

frontend_task:
  description: >
    Write a gradio UI in app.py that demonstrates the given backend class.
    Keep the UI simple - just a prototype or demo.
  expected_output: >
    IMPORTANT: Output ONLY the raw Python code without any markdown formatting.
  agent: frontend_engineer
  context:
    - code_task
  output_file: output/app.py

test_task:
  description: >
    Write unit tests for the given backend module {module_name}.
  expected_output: >
    IMPORTANT: Output ONLY the raw Python code without any markdown formatting.
  agent: test_engineer
  context:
    - code_task
  output_file: output/test_{module_name}
```

**crew.py**:
```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class EngineeringTeam():
    """EngineeringTeam crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Lead only designs - no code execution needed
    @agent
    def engineering_lead(self) -> Agent:
        return Agent(config=self.agents_config['engineering_lead'], verbose=True)

    # Backend engineer writes AND runs code in Docker
    @agent
    def backend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['backend_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3
        )

    # Frontend engineer writes Gradio UI but does NOT execute it
    # (running Gradio in Docker would be a different challenge)
    @agent
    def frontend_engineer(self) -> Agent:
        return Agent(config=self.agents_config['frontend_engineer'], verbose=True)

    # Test engineer writes AND runs unit tests in Docker
    @agent
    def test_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['test_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3
        )

    @task
    def design_task(self) -> Task:
        return Task(config=self.tasks_config['design_task'])

    @task
    def code_task(self) -> Task:
        return Task(config=self.tasks_config['code_task'])

    @task
    def frontend_task(self) -> Task:
        return Task(config=self.tasks_config['frontend_task'])

    @task
    def test_task(self) -> Task:
        return Task(config=self.tasks_config['test_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

**main.py**:
```python
import warnings
import os
from engineering_team.crew import EngineeringTeam

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
os.makedirs('output', exist_ok=True)

requirements = """
A simple account management system for a trading simulation platform.
The system should allow users to create an account, deposit funds, and withdraw funds.
The system should allow users to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio and profit/loss.
The system should prevent the user from withdrawing funds that would leave them with a negative balance,
 or buying more shares than they can afford, or selling shares that they don't have.
The system has access to a function get_share_price(symbol) which returns the current price of a share,
 and includes a test implementation that returns fixed prices for AAPL, TSLA, GOOGL.
"""
module_name = "accounts.py"
class_name = "Account"

def run():
    inputs = {
        'requirements': requirements,
        'module_name': module_name,
        'class_name': class_name
    }
    result = EngineeringTeam().crew().kickoff(inputs=inputs)
    print(result.raw)
```

**What happens when you run it** (~5 minutes total):
1. **Engineering Lead** (GPT-4o) produces a design document in markdown with class/method signatures → `output/accounts.py_design.md`
2. **Backend Engineer** (Claude) implements the full Python module based on the design → `output/accounts.py`
3. **Frontend Engineer** (Claude) builds a Gradio UI with tabs (Account Management, Trading, Reports) → `output/app.py`
4. **Test Engineer** (Claude) writes unit tests with assertions → `output/test_accounts.py`

**Output files**:
```
output/
├── accounts.py_design.md   # Design document (markdown)
├── accounts.py             # Full implementation
├── app.py                  # Gradio UI (runnable with `uv run app.py`)
└── test_accounts.py        # Unit tests
```

**Results from the transcript**:
- The generated `accounts.py` included proper docstrings, validation logic, and even defensive copying of holdings (`.copy()`) — "a pro thing I wouldn't have thought of myself"
- The Gradio UI had tabs for Account Management, Trading, and Reports — fully functional
- Running `uv run app.py` launched a working web interface for the trading simulator
- The test file contained proper unit test scaffolding with assertions

**Key insights**:
- **Think of tasks as user prompts and agents as system prompts** — that's how it all comes together under the hood
- **Context = what information from other tasks gets included in the prompt**
- **YAML formatting matters** — stray tabs cause obscure errors. This is the trade-off of frameworks: you get a lot out of the box, but when things go wrong, debugging is harder because stuff is hidden from you
- **Model selection per role**: GPT-4o for planning/design, Claude for code generation. You can also use GPT-4o-mini (works fine) or even DeepSeek for the test engineer
- **"Vibe coding" warning**: LLMs tend to generate too much code. The crew function only needs `agents=self.agents, tasks=self.tasks` — don't let autocomplete add unnecessary complexity

### Example 5: Stock Picker (Hierarchical Process + Structured Outputs + Custom Tool)

This is the most advanced project of them all — demonstrates all three new concepts together: **structured outputs**, **hierarchical process**, and a **custom tool**.

**Three new concepts in one project**:
1. **Structured outputs** — tasks produce JSON conforming to Pydantic schemas
2. **Custom tool** — PushNotificationTool sends alerts to the user
3. **Hierarchical process** — a manager agent delegates tasks instead of sequential execution

**agents.yaml**:
```yaml
trending_company_finder:
  role: >
    Financial News Analyst that finds trending companies in {sector}
  goal: >
    You read the latest news, then find 2-3 companies that are trending in the news for further research.
    Always pick new companies. Don't pick the same company twice.
  backstory: >
    You are a market expert with a knack for picking out the most interesting companies based on latest news.
  llm: openai/gpt-4o-mini

financial_researcher:
  role: >
    Senior Financial Researcher
  goal: >
    Given details of trending companies in the news, you provide comprehensive analysis of each in a report.
  backstory: >
    You are a financial expert with a proven track record of deeply analyzing hot companies.
  llm: openai/gpt-4o-mini

stock_picker:
  role: >
    Stock Picker from Research
  goal: >
    Given a list of researched companies with investment potential, you select the best one for investment,
    notifying the user and then providing a detailed report. Don't pick the same company twice.
  backstory: >
    You're a meticulous, skilled financial analyst with a proven track record of equity selection.
  llm: openai/gpt-4o-mini

# Manager is separate - uses GPT-4o (stronger model) for better delegation
manager:
  role: >
    Manager
  goal: >
    You are a skilled project manager who can delegate tasks to pick the best company for investment.
  backstory: >
    You are an experienced and highly effective project manager.
  llm: openai/gpt-4o
```

**tasks.yaml**:
```yaml
find_trending_companies:
  description: >
    Find the top trending companies in the news in {sector} by searching the latest news.
    Find new companies that you've not found before.
  expected_output: >
    A list of trending companies in {sector}
  agent: trending_company_finder
  output_file: output/trending_companies.json

research_trending_companies:
  description: >
    Given a list of trending companies, provide detailed analysis of each company by searching online
  expected_output: >
    A report containing detailed analysis of each company
  agent: financial_researcher
  context:
    - find_trending_companies
  output_file: output/research_report.json

pick_best_company:
  description: >
    Analyze the research findings and pick the best company for investment.
    Send a push notification to the user with the decision and 1 sentence rationale.
    Then respond with a detailed report on why you chose this company.
  expected_output: >
    The chosen company and why; the companies not selected and why.
  agent: stock_picker
  context:
    - research_trending_companies
  output_file: output/decision.md
```

**crew.py** (key parts):
```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List
from .tools.push_tool import PushNotificationTool


# Structured output schemas
class TrendingCompany(BaseModel):
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending")

class TrendingCompanyList(BaseModel):
    companies: List[TrendingCompany] = Field(description="List of trending companies")

class TrendingCompanyResearch(BaseModel):
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential")

class TrendingCompanyResearchList(BaseModel):
    research_list: List[TrendingCompanyResearch] = Field(description="Research on all companies")


@CrewBase
class StockPicker():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(config=self.agents_config['trending_company_finder'],
                     tools=[SerperDevTool()])

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(config=self.agents_config['financial_researcher'],
                     tools=[SerperDevTool()])

    @agent
    def stock_picker(self) -> Agent:
        return Agent(config=self.agents_config['stock_picker'],
                     tools=[PushNotificationTool()])  # Custom tool

    @task
    def find_trending_companies(self) -> Task:
        return Task(config=self.tasks_config['find_trending_companies'],
                    output_pydantic=TrendingCompanyList)  # Structured output

    @task
    def research_trending_companies(self) -> Task:
        return Task(config=self.tasks_config['research_trending_companies'],
                    output_pydantic=TrendingCompanyResearchList)

    @task
    def pick_best_company(self) -> Task:
        return Task(config=self.tasks_config['pick_best_company'])

    @crew
    def crew(self) -> Crew:
        # Manager created separately - NOT in @agent list
        manager = Agent(config=self.agents_config['manager'],
                        allow_delegation=True)
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,  # Manager delegates
            verbose=True,
            manager_agent=manager,
        )
```

**main.py**:
```python
import os
from stock_picker.crew import StockPicker

os.makedirs('output', exist_ok=True)

def run():
    inputs = {'sector': 'Technology'}
    result = StockPicker().crew().kickoff(inputs=inputs)
    print(result.raw)
```

**What happens when you run it**:
1. Manager (GPT-4o) receives all tasks and decides how to delegate
2. Trending Company Finder searches news → outputs JSON conforming to `TrendingCompanyList`
3. Financial Researcher analyzes each company → outputs JSON conforming to `TrendingCompanyResearchList`
4. Stock Picker picks the best one, sends a push notification via custom tool, writes decision report

**Output files**:
```
output/
├── trending_companies.json    # Structured: [{name, ticker, reason}, ...]
├── research_report.json       # Structured: [{name, market_position, future_outlook, ...}, ...]
└── decision.md                # Free-form markdown report
```

**Observations from the transcript**:
- The hierarchical process is "less predictable" — the manager has autonomy in how it assigns tasks
- Despite this, it "actually performed really well" — the manager went through the process correctly
- On one run it recommended Anthropic (interesting since OpenAI was doing the processing)
- On a second run it recommended Circle (a runner-up from the first run)
- The push notification was received successfully on the user's phone

### Extension: Callbacks & Dynamic Task Creation

CrewAI provides callback hooks at multiple levels. The transcript mentions the scaffolding code that gets generated includes "functions that get called at the beginning and end" — these are the `@before_kickoff` and `@after_kickoff` decorators.

**Practical example**: The stock picker uses memory so it "doesn't pick the same company twice." But what if you wanted to dynamically research a variable number of companies based on what the first task finds? Callbacks enable this.

#### The 5 Callback Types

| Callback | When it fires | Use case |
|----------|--------------|----------|
| `@before_kickoff` | Before crew starts | Enrich inputs, validate config, set dynamic parameters |
| `@after_kickoff` | After crew finishes | Post-process output, logging, save to DB |
| Task `callback` | After a specific task | Side effects: send notification, trigger external system |
| `task_callback` (crew-level) | After every task | Progress monitoring, cost tracking |
| `step_callback` (crew-level) | After every agent step | Detailed debugging, token usage logging |

#### `@before_kickoff` — Modify Inputs Before Execution

```python
from crewai.project import CrewBase, before_kickoff, after_kickoff

@CrewBase
class StockPicker():

    @before_kickoff
    def prepare_inputs(self, inputs):
        """Runs BEFORE the crew starts — can modify/enrich the inputs dict."""
        # Dynamically set the sector based on day of week, market conditions, etc.
        inputs['sector'] = get_trending_sector()
        inputs['current_date'] = str(datetime.now())
        return inputs  # Must return the modified dict
```

#### `@after_kickoff` — Process Output After Completion

```python
    @after_kickoff
    def process_output(self, output):
        """Runs AFTER the crew finishes — can modify the CrewOutput."""
        # Log the final decision to a database
        save_to_db(output.raw)
        output.raw += f"\n\nProcessed at {datetime.now()}"
        return output
```

#### Task `callback` — Per-Task Side Effects

This is what the stock picker uses conceptually with the push notification tool, but you can also do it via a Python callback instead of giving the agent a tool:

```python
from crewai import TaskOutput

def notify_user(output: TaskOutput):
    """Called after the pick_best_company task completes."""
    # Send push notification with the decision
    push(f"Stock pick: {output.raw[:100]}")

@task
def pick_best_company(self) -> Task:
    return Task(
        config=self.tasks_config['pick_best_company'],
        callback=notify_user,  # Fires after this task, guaranteed
    )
```

**Callback vs Tool for notifications**: A `callback` is deterministic — it always fires after the task. A tool depends on the LLM deciding to use it. If you need guaranteed execution, use a callback.

#### Crew-Level `task_callback` — Monitor All Tasks

```python
def log_progress(output: TaskOutput):
    """Called after EVERY task in the crew completes."""
    print(f"✓ Completed: {output.description[:50]}")
    print(f"  Output length: {len(output.raw)} chars")

@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        verbose=True,
        manager_agent=manager,
        task_callback=log_progress,  # Fires after each task
    )
```

#### Dynamic Task Creation at Runtime

For scenarios where the number of tasks isn't known until runtime (e.g., research N companies found by the first task), you can build tasks programmatically instead of using YAML:

```python
@crew
def crew(self) -> Crew:
    # Dynamically generate research tasks for multiple sectors
    sectors = ['Technology', 'Healthcare', 'Energy']
    tasks = []
    for sector in sectors:
        tasks.append(Task(
            description=f"Find trending companies in {sector}",
            expected_output=f"List of trending companies in {sector}",
            agent=self.trending_company_finder(),
        ))

    # Add a final aggregation task
    tasks.append(Task(
        description="Pick the best company across all sectors",
        expected_output="The chosen company and rationale",
        agent=self.stock_picker(),
        context=tasks[:-1],  # All research tasks as context
    ))

    return Crew(
        agents=self.agents,
        tasks=tasks,  # Dynamically built, not self.tasks
        process=Process.sequential,
        verbose=True,
    )
```

**When to use dynamic tasks vs YAML**:
- **YAML**: Fixed, known task structure — cleaner separation of concerns
- **Dynamic**: Variable number of tasks, conditional logic, runtime-dependent workflows

### Extension: Task Guardrails

Guardrails validate task output **before** it's passed to the next task. If validation fails, the error is sent back to the agent to retry (up to `guardrail_max_retries`).

#### Function-Based Guardrail

```python
from crewai import Task, TaskOutput
from typing import Tuple, Any

def validate_has_ticker(result: TaskOutput) -> Tuple[bool, Any]:
    """Ensure the stock pick includes a valid ticker symbol."""
    if not any(c.isupper() and len(word) <= 5 for word in result.raw.split()
               for c in word if word.isupper()):
        return (False, "Response must include a stock ticker symbol (e.g., AAPL, TSLA)")
    return (True, result.raw)

pick_task = Task(
    description="Pick the best company for investment",
    expected_output="Company name, ticker, and rationale",
    agent=stock_picker,
    guardrail=validate_has_ticker,
    guardrail_max_retries=3,  # Agent gets 3 attempts to fix it
)
```

**How it works**:
1. Agent produces output
2. Guardrail function receives `TaskOutput`, returns `(bool, Any)`
3. If `(True, result)` → output accepted, moves to next task
4. If `(False, "error message")` → error sent back to agent, agent retries

#### LLM-Based Guardrail (String)

For subjective validation, pass a string instead of a function — CrewAI uses the agent's LLM to evaluate:

```python
research_task = Task(
    description="Research trending companies in Technology",
    expected_output="Detailed analysis of 2-3 companies",
    agent=researcher,
    guardrail="The output must contain at least 2 companies with specific financial data",
    guardrail_max_retries=2,
)
```

#### Multiple Guardrails (Chained)

```python
pick_task = Task(
    description="Pick the best company for investment",
    expected_output="Company name, ticker, and rationale",
    agent=stock_picker,
    guardrails=[
        validate_has_ticker,                    # Function: check ticker exists
        "The rationale must be at least 3 sentences long",  # LLM: check quality
    ],
    guardrail_max_retries=3,
)
```

Guardrails execute sequentially — each receives the output from the previous one. This combines deterministic validation (functions) with subjective quality checks (LLM strings).

---

## Running a Crew

### Setup

```bash
# Install the CrewAI CLI tool (one-time)
uv tool install crewai

# Create a new project
crewai create crew debate

# Navigate into the project directory
cd debate

# Install dependencies (uv sync happens automatically on first run)
uv sync
```

### Environment Variables

```bash
# Set in .env file or export
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export SERPER_API_KEY=...        # For web search tools (SerperDevTool)
export PUSHOVER_USER=...         # For push notification tools
export PUSHOVER_TOKEN=...
```

### Execution

```bash
# Run the crew (from within the project directory)
crewai run

# This is equivalent to:
uv run python -m debate.main
```

**First run**: uv will build the environment (install dependencies). Subsequent runs start immediately.

**What happens during execution** (verbose=True output):
1. Each agent is announced as it starts working
2. You see which task is being executed
3. The LLM's reasoning/output is printed in real-time
4. Tool calls are logged
5. Final result is printed

### Output

Tasks with `output_file` automatically write results to disk in the `output/` directory:
```
output/
├── propose.md    # The proposer's argument
├── oppose.md     # The opposer's argument
└── decide.md     # The judge's decision
```

The `result` object from `kickoff()` contains:
- `result.raw` — the final task's text output (the last task in sequential order)
- Structured data if `output_pydantic` was specified on the final task

---

## Key Takeaways

1. **CrewAI is multi-product** — distinguish between Enterprise (platform), Studio (low-code), and Framework (open source)
2. **More opinionated than OpenAI Agents SDK** — forces role/goal/backstory structure instead of free-form instructions. Good for best practices, harder to debug at the prompt level.
3. **YAML-driven configuration** — separates agent personas and task definitions from orchestration logic; prompts aren't buried in code
4. **Tasks are first-class primitives** — no analog in OpenAI Agents SDK. Tasks have descriptions, expected outputs, and are assigned to agents.
5. **Decorator pattern** — `@CrewBase`, `@agent`, `@task`, `@crew` decorators auto-wire YAML configs to Python objects. Method names must match YAML keys.
6. **Template variables** — `{variable}` in YAML gets interpolated from `inputs` dict at `kickoff()`
7. **One agent, multiple tasks** — the same agent can handle different tasks; the task description drives behavior
8. **Naming constraint** — task names cannot conflict with agent names
9. **LiteLLM under the hood** — lightweight, flexible model switching. `provider/model-name` format. Significant advantage over OpenAI Agents SDK.
10. **Sequential vs Hierarchical** — sequential for predictable pipelines, hierarchical for dynamic delegation via a manager LLM
11. **Task context creates pipelines** — the `context` field passes prior task outputs to downstream tasks
12. **Structured outputs** — `output_pydantic` forces typed JSON responses for reliable inter-task data flow
13. **Memory persists across runs** — SQLite + RAG embeddings enable agents to learn from prior sessions
14. **Code execution is built-in** — Docker-sandboxed Python execution with retry logic
15. **Custom tools follow the same pattern** — Pydantic schema + `_run()` method, same as OpenAI function calling under the hood
16. **Project scaffolding required** — `crewai create crew <name>` generates the directory structure. CrewAI doesn't work in notebooks.
17. **`crewai run`** — executes the crew from within the project directory. First run builds the uv environment.
18. **`result.raw`** — gives you the raw text output from the final task in the pipeline
19. **Tasks = user prompts, Agents = system prompts** — that's the mental model for how it all comes together under the hood
20. **Framework trade-off** — you get a lot out of the box (memory, code execution, tool integration), but when things go wrong, debugging is harder because the internals are hidden. YAML formatting errors produce obscure stack traces.
21. **"IMPORTANT: Output ONLY raw Python code"** — without this instruction, LLMs wrap output in markdown code blocks, producing invalid files. Always include this for code-generating tasks with `output_file`.

---

## Code

Complete runnable project code is available in the `code/` directory:

- [`code/debate/`](code/debate/) — The Debate crew (2 agents, 3 tasks, sequential process, multi-model)
- [`code/financial_researcher/`](code/financial_researcher/) — The Financial Researcher crew (2 agents, 2 tasks, SerperDevTool, context passing)
- [`code/coder/`](code/coder/) — The Coder crew (1 agent, 1 task, Docker code execution)
- [`code/engineering_team/`](code/engineering_team/) — The Engineering Team crew (4 agents, 4 tasks, task context dependencies, code execution)
- [`code/stock_picker/`](code/stock_picker/) — The Stock Picker crew (4 agents, 3 tasks, hierarchical process, structured outputs, custom tool)

```
code/
├── debate/
│   ├── pyproject.toml
│   └── src/debate/
│       ├── __init__.py
│       ├── main.py
│       ├── crew.py
│       └── config/
│           ├── agents.yaml
│           └── tasks.yaml
├── financial_researcher/
│   ├── pyproject.toml
│   └── src/financial_researcher/
│       ├── __init__.py
│       ├── main.py
│       ├── crew.py
│       └── config/
│           ├── agents.yaml
│           └── tasks.yaml
├── coder/
│   ├── pyproject.toml
│   └── src/coder/
│       ├── __init__.py
│       ├── main.py
│       ├── crew.py
│       └── config/
│           ├── agents.yaml
│           └── tasks.yaml
├── engineering_team/
│   ├── pyproject.toml
│   └── src/engineering_team/
│       ├── __init__.py
│       ├── main.py
│       ├── crew.py
│       └── config/
│           ├── agents.yaml
│           └── tasks.yaml
└── stock_picker/
    ├── pyproject.toml
    └── src/stock_picker/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── config/
        │   ├── agents.yaml
        │   └── tasks.yaml
        └── tools/
            ├── __init__.py
            └── push_tool.py
```
