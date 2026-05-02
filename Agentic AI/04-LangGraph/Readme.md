# LangGraph — Part 1: Foundations & First Graph

## Table of Contents
- [The LangChain Ecosystem](#the-langchain-ecosystem)
  - [LangChain](#langchain)
  - [LangGraph](#langgraph)
  - [LangGraph Sub-Products](#langgraph-sub-products)
  - [LangSmith](#langsmith)
  - [Deep Dive: LangChain vs LangGraph](#deep-dive-langchain-vs-langgraph)
- [Anthropic's Perspective: Building Effective Agents](#anthropics-perspective-building-effective-agents)
  - [Agents vs Workflows](#agents-vs-workflows)
  - [When (and When Not) to Use Agents](#when-and-when-not-to-use-agents)
  - [On Frameworks](#on-frameworks)
  - [The Augmented LLM (Building Block)](#the-augmented-llm-building-block)
  - [Workflow Patterns](#workflow-patterns)
  - [Autonomous Agents](#autonomous-agents)
  - [Agent-Computer Interface (ACI)](#agent-computer-interface-aci)
  - [Core Principles](#core-principles)
- [LangGraph Terminology](#langgraph-terminology)
  - [Graph](#graph)
  - [State](#state)
  - [Nodes](#nodes)
  - [Edges](#edges)
  - [Reducers](#reducers)
- [The Annotated Type Hint](#the-annotated-type-hint)
- [Immutable State](#immutable-state)
- [The 5 Steps to Build a Graph](#the-5-steps-to-build-a-graph)
- [Example 1: Silly Random Node (No LLM)](#example-1-silly-random-node-no-llm)
- [Example 2: Real LLM Chatbot](#example-2-real-llm-chatbot)
- [Key Takeaways](#key-takeaways)

---

## The LangChain Ecosystem

LangChain (the company) offers three distinct products. Understanding how they relate is critical before diving into LangGraph.

### LangChain

LangChain is the **original abstraction framework** for building LLM applications. It's been around for years and was one of the earliest abstraction layers. Its core value proposition:

- **Abstraction over LLM providers** — switch from GPT to Claude without rewriting integration code. Its initial raison d'être was eliminating the pain of bespoke API integrations.
- **Chaining** — compose sequential LLM calls into pipelines (the "chain" in LangChain).
- **RAG support** — retrieval-augmented generation with vector stores.
- **Prompt templates** — higher-level constructs for prompt engineering with good practices baked in.
- **Memory models** — in-RAM or persisted-to-database conversation memory, with various memory strategies (similar to what CrewAI offers, but with more abstractions).
- **Tool calling abstractions** — unified interface for function/tool use across providers.
- **LCEL (LangChain Expression Language)** — a declarative language for composing chains.

**Trade-offs:** By adopting LangChain's abstractions, you gain rapid development speed (e.g., a RAG pipeline in ~4 lines of code) but lose visibility into the actual prompts and responses flowing to/from the LLM. Over time, LLM APIs have converged — most follow OpenAI's endpoint structure (Anthropic being somewhat of an odd one out). This makes direct API calls increasingly simple. Memory is fundamentally just a JSON blob of conversation history — you can manage it yourself, persist it however you want, combine it in different ways.

LangChain *can* build agentic workflows (it has tool-calling abstractions), but it **predates the modern agent explosion**. It's more of a **glue layer for any LLM application** rather than a dedicated agent orchestration platform. It's not their main agent offering.

### LangGraph

LangGraph is a **separate product** from the same company, focused specifically on **agentic AI workflows**. It is **independent from LangChain** — you can use LangChain's LLM wrappers with it, but it's entirely optional. You can call LLMs directly or use any framework you prefer.

**Core thesis:** Stability, resiliency, and repeatability for complex interconnected processes (agent systems).

LangGraph represents all workflows as **graphs** — tree-like structures of nodes connected by edges. By abstracting workflows this way and adding checkpointing/monitoring hooks at each point, it brings:

- Human-in-the-loop patterns
- Multi-agent collaboration
- Conversation history & memory
- **Time travel** — checkpoint at any point, step backwards, restore prior states
- Fault-tolerant scalability (anything can go down and it keeps running)

**What problem does it solve?** The world of agentic AI is unpredictable. People have resiliency concerns. LangGraph's approach is to put "belts and braces" around each point in the graph, bringing stability and monitoring to an inherently non-deterministic system.

### LangGraph Sub-Products

LangGraph itself is actually **three things**:

| Product | Description | Analogy |
|---------|-------------|---------|
| **LangGraph (Framework)** | The open-source Python framework for defining graphs | CrewAI framework |
| **LangGraph Studio** | Visual builder / UI tool for hooking things up visually | CrewAI Studio |
| **LangGraph Platform** | Hosted solution for deploying & running graphs at scale | CrewAI Enterprise |

The website heavily promotes LangGraph Platform as if it *is* LangGraph — this is likely because it's the core commercial/monetization play. If you've built everything using LangGraph, it's convenient to deploy on their platform. But **we focus exclusively on LangGraph the framework**.

### LangSmith

LangSmith is the **observability/monitoring** product. It provides visibility into your LLM calls and reasoning chains to quickly debug failures. LangGraph does not do monitoring itself — it connects to LangSmith for that. LangSmith works with both LangChain and LangGraph.

### Deep Dive: LangChain vs LangGraph

Although they come from the same company, LangChain and LangGraph solve fundamentally different problems and operate at different levels of abstraction.

#### Philosophical Difference

| | LangChain | LangGraph |
|---|-----------|-----------|
| **Core metaphor** | A **chain** — linear sequence of operations | A **graph** — directed network of stateful operations |
| **Primary concern** | Simplify *calling* LLMs and composing outputs | Orchestrate *complex workflows* with reliability guarantees |
| **Design era** | Pre-agent (2022–2023): "How do I call an LLM and chain results?" | Agent era (2024+): "How do I run multi-step, multi-agent systems reliably?" |
| **Relationship to LLMs** | Tightly coupled — LLM calls are the core primitive | Loosely coupled — nodes are arbitrary Python functions, LLMs optional |

#### Architectural Comparison

| Dimension | LangChain | LangGraph |
|-----------|-----------|-----------|
| **Execution model** | Sequential chain / DAG via LCEL (see below) | Stateful graph with cycles, conditionals, and parallel execution |
| **State management** | Memory classes (ConversationBufferMemory, etc.) — abstracted away | Explicit immutable State object you define and control |
| **Control flow** | Implicit via chain composition or LCEL pipes | Explicit via edges (simple or conditional) between named nodes |
| **Concurrency** | Limited — chains are inherently sequential | First-class — parallel nodes with reducer-based state merging |
| **Checkpointing** | Not built-in | Native — snapshot state at any point, restore, time-travel |
| **Human-in-the-loop** | Requires custom implementation | Built-in pattern — pause execution, wait for human input, resume |
| **Error recovery** | Try/catch at chain level | Graph-level — retry nodes, branch on failure, resume from checkpoint |
| **Cyclic workflows** | Not supported (DAG only) | Supported — feedback loops, iterative refinement, agent loops |

#### What is "DAG via LCEL"?

**DAG** stands for **Directed Acyclic Graph** — a graph where edges have direction (A → B) and there are **no cycles** (you can never loop back to a previous step). This is a fundamental limitation: once a step is done, you cannot revisit it.

**LCEL** (LangChain Expression Language) is LangChain's declarative syntax for composing chains using the pipe operator (`|`). It lets you wire components together in a DAG structure:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LCEL chain using pipe operator (|) — data flows left to right
# prompt formats input → llm generates response → output_parser extracts text
chain = prompt | llm | output_parser

# Parallel branches using RunnableParallel (this is the DAG part)
from langchain_core.runnables import RunnableParallel

# Fan-out: two branches run simultaneously, then results merge into combine_results
# This is a DAG — directed (left to right), acyclic (no loops back)
chain = RunnableParallel(
    summary=prompt_summary | llm | parser,       # Branch 1: summarize
    translation=prompt_translate | llm | parser,  # Branch 2: translate
) | combine_results  # Fan-in: merge both branch outputs
```

With `RunnableParallel`, LCEL can express fan-out/fan-in patterns (multiple branches running in parallel, results merged) — this is a DAG. But it **cannot express cycles**: no "if the output is bad, loop back and try again." That's where LangGraph steps in — it supports cyclic graphs, enabling iterative refinement, retry loops, and agent decision loops that revisit earlier nodes.

#### When to Use Which

**Use LangChain when:**
- You need a quick abstraction over multiple LLM providers
- Building a straightforward RAG pipeline or prompt chain
- You want pre-built integrations (vector stores, document loaders, output parsers)
- The workflow is linear and deterministic

**Use LangGraph when:**
- Building multi-agent systems with complex interaction patterns
- You need human-in-the-loop, checkpointing, or fault tolerance
- Workflows have cycles (e.g., evaluator-optimizer loops, retry logic)
- You need parallel execution with safe state merging
- Observability and reproducibility are requirements (production agent systems)
- You want full control over state and execution flow

**Use both together when:**
- LangGraph orchestrates the workflow (nodes, edges, state)
- LangChain provides convenient LLM wrappers inside nodes (e.g., `ChatOpenAI`)

This is the most common pattern in practice — LangGraph for structure, LangChain for LLM calls:

```python
from langchain_openai import ChatOpenAI          # LangChain: LLM wrapper
from langgraph.graph import StateGraph, START, END  # LangGraph: orchestration

# LangChain handles the LLM API call details
llm = ChatOpenAI(model="gpt-4o-mini")

# LangGraph node — a plain function that uses LangChain's llm inside
def my_node(state: State) -> State:
    response = llm.invoke(state.messages)  # LangChain invoke inside LangGraph node
    return State(messages=[response])      # Return new state for the reducer

# LangGraph handles the orchestration: build → compile → invoke
graph_builder = StateGraph(State)
graph_builder.add_node("agent", my_node)   # Register node
graph_builder.add_edge(START, "agent")     # Wire: start → agent
graph_builder.add_edge("agent", END)       # Wire: agent → end
graph = graph_builder.compile()            # Finalize graph
```

#### Conceptual Evolution

Think of it as a maturity progression:

```
Direct API calls → LangChain (abstraction) → LangGraph (orchestration)
     ↑                    ↑                         ↑
 Full control      Convenience layer         Production-grade agent systems
 No guardrails     Linear workflows          Stateful, resilient, observable
```

LangChain answers: *"How do I talk to an LLM cleanly?"*
LangGraph answers: *"How do I build a reliable system where multiple LLMs, tools, and humans collaborate?"*

They're complementary, not competing. But you can use LangGraph without LangChain — just call LLMs directly in your nodes.

---

## Anthropic's Perspective: Building Effective Agents

The references Anthropic's blog post [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) are essential for reading. It provides a counterpoint to the framework-heavy approach and contains the design patterns referenced throughout the course. Here's a comprehensive summary:

### Agents vs Workflows

Anthropic draws an important architectural distinction within "agentic systems":

- **Workflows** — systems where LLMs and tools are orchestrated through **predefined code paths**. The developer controls the flow.
- **Agents** — systems where LLMs **dynamically direct their own processes** and tool usage, maintaining control over how they accomplish tasks.

### When (and When Not) to Use Agents

Anthropic's key recommendation: **find the simplest solution possible, and only increase complexity when needed.**

- Agentic systems trade **latency and cost** for better task performance. Consider when this tradeoff makes sense.
- **Workflows** offer predictability and consistency for well-defined tasks.
- **Agents** are better when flexibility and model-driven decision-making are needed at scale.
- For many applications, **optimizing single LLM calls with retrieval and in-context examples is usually enough** — no agent needed.

### On Frameworks

> "These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice."

> **"We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error."**

From Anthropic's perspective: they have an API, it's relatively simple, memory is JSON objects, LLMs can be called directly. Building heavy abstraction layers that take you further from the LLM itself doesn't necessarily resonate with their philosophy. This is a different school of thought from LangGraph's — keep it in mind.

### The Augmented LLM (Building Block)

The foundational building block of all agentic systems is an **LLM enhanced with augmentations**:
- **Retrieval** — the model generates its own search queries
- **Tools** — the model selects appropriate tools to call
- **Memory** — the model determines what information to retain

Modern models can actively use all these capabilities. Anthropic's [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) is their approach to integrating with third-party tools via a standard protocol (covered in section 6).

### Workflow Patterns

Anthropic identifies **5 workflow patterns** that represent the most common production architectures:

#### 1. Prompt Chaining
Decompose a task into a **sequence of steps**, where each LLM call processes the output of the previous one. Add programmatic checks ("gates") on intermediate steps.

**Use when:** Task can be cleanly decomposed into fixed subtasks. Trade latency for accuracy by making each LLM call easier.

**Examples:**
- Generate marketing copy → translate to another language
- Write document outline → check criteria → write document from outline

#### 2. Routing
Classify an input and **direct it to a specialized followup task**. Enables separation of concerns and specialized prompts.

**Use when:** Distinct categories exist that are better handled separately, and classification can be done accurately.

**Examples:**
- Customer service: general questions vs. refund requests vs. technical support → different processes/prompts/tools
- Easy questions → small cheap model (Haiku); hard questions → capable model (Sonnet)

#### 3. Parallelization
LLMs work **simultaneously** on a task, outputs aggregated programmatically. Two variations:
- **Sectioning** — break task into independent subtasks run in parallel
- **Voting** — run same task multiple times for diverse outputs

**Use when:** Subtasks can be parallelized for speed, or multiple perspectives needed for confidence.

**Examples:**
- *Sectioning:* One model handles user queries while another screens for inappropriate content. Automating evals where each call evaluates a different aspect.
- *Voting:* Multiple prompts review code for vulnerabilities. Multiple prompts evaluate content appropriateness with vote thresholds.

#### 4. Orchestrator-Workers
A **central LLM dynamically breaks down tasks**, delegates to worker LLMs, and synthesizes results. Unlike parallelization, subtasks aren't pre-defined — the orchestrator determines them based on input.

**Use when:** Can't predict subtasks needed (e.g., in coding, the number and nature of file changes depends on the task).

**Examples:**
- Coding products making complex changes to multiple files
- Search tasks gathering/analyzing information from multiple sources

#### 5. Evaluator-Optimizer
One LLM generates a response, another **provides evaluation and feedback in a loop**.

**Use when:** Clear evaluation criteria exist, iterative refinement provides measurable value, and LLM responses demonstrably improve when feedback is articulated.

**Examples:**
- Literary translation with nuance refinement
- Complex search requiring multiple rounds of searching/analysis

### Autonomous Agents

Full agents emerge as LLMs mature in: understanding complex inputs, reasoning/planning, reliable tool use, and error recovery.

Agent execution pattern:
1. Receive command or have interactive discussion with human
2. Plan and operate independently
3. Gain "ground truth" from environment at each step (tool results, code execution)
4. Pause for human feedback at checkpoints or blockers
5. Terminate on completion or hitting stopping conditions (e.g., max iterations)

**Implementation is often straightforward** — typically just LLMs using tools based on environmental feedback in a loop. The key is designing toolsets and their documentation clearly.

**Trade-offs:** Higher costs, potential for compounding errors. Requires extensive testing in sandboxed environments with appropriate guardrails.

### Agent-Computer Interface (ACI)

Anthropic emphasizes investing as much effort in the **agent-computer interface** as in human-computer interfaces:

- Put yourself in the model's shoes — is it obvious how to use this tool from the description?
- Include example usage, edge cases, input format requirements, boundaries from other tools
- Optimize parameter names and descriptions (like writing a great docstring for a junior dev)
- Test extensively — run many inputs, observe mistakes, iterate
- **Poka-yoke** (error-proof) your tools — change arguments so mistakes are harder to make

Example: For SWE-bench, they found the model made mistakes with relative filepaths. Changing to always require absolute filepaths fixed it completely. They spent **more time optimizing tools than the overall prompt**.

### Core Principles

Anthropic's three principles for implementing agents:
1. Maintain **simplicity** in your agent's design
2. Prioritize **transparency** by explicitly showing planning steps
3. Carefully craft your ACI through thorough tool **documentation and testing**

> "Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production."

---

## LangGraph Terminology

### Graph

An agent workflow is represented as a **graph** — a tree-like structure of interconnected components. It's a directed graph where execution flows from one component to the next based on defined connections. This is the core abstraction of LangGraph (as the name gives away).

### State

The **State** is an object representing the **current snapshot** of your entire application at any point in time. It is:

- Shared across the whole application
- Passed into and returned from every node
- **Immutable** — you never mutate it; you create new instances
- **Information, not a function** — it's data, not logic

### Nodes

Nodes are **Python functions**. This can be confusing at first — when you think of graphs, you might think of nodes as data points. But in LangGraph, nodes are operations. Each node:

1. Receives the current state as input
2. Does something (call an LLM, write to a file, perform computation — **anything**)
3. Returns a **new** state (never mutates the old one)

**Critical insight:** Nodes don't need to involve LLMs at all. They're just Python functions. Any computation works.

### Edges

Edges are the **connections between nodes**. They determine execution flow:

- **Simple edges** — unconditional: "after node A, always run node B"
- **Conditional edges** — Python functions that examine the state and decide which node runs next

**Summary:** Nodes do the work. Edges decide what happens next.

### Reducers

A **reducer** is a function associated with a field in your State that tells LangGraph **how to combine** that field when multiple state updates occur.

**Why reducers exist:** LangGraph can run multiple nodes in parallel. If two nodes both return state updates to the same field simultaneously, the reducer defines how to merge them without one overwriting the other. This is the clever trick that enables safe parallel execution.

LangGraph provides a built-in reducer called `add_messages` that:
- Concatenates message lists together
- Packages raw dicts into `HumanMessage`/`AIMessage` objects automatically

---

## The Annotated Type Hint

Python's `Annotated` type hint lets you attach metadata to a type. Python itself completely ignores the metadata, but other frameworks (like LangGraph) can read it.

```python
from typing import Annotated

# Basic type hint — tells Python (and IDEs) this is a list
my_list: list

# Annotated type hint — adds metadata ("these are a few of mine")
# Python completely ignores the second argument; it's for other frameworks to read
my_list: Annotated[list, "these are a few of mine"]

# Function with annotated parameter
# The Annotated[str, "..."] is invisible to Python — function works identically without it
def shout(text: Annotated[str, "something to be shouted"]) -> str:
    print(text.upper())
    return text.upper()

shout("hello")  # prints: HELLO — annotation has zero effect on execution
```

LangGraph uses `Annotated` to specify **which reducer** to use for each state field:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(BaseModel):
    # Annotated[list, add_messages] means:
    #   - "messages" is a list (the type)
    #   - add_messages is the reducer (the metadata LangGraph reads)
    # LangGraph will call add_messages() to merge old + new messages automatically
    messages: Annotated[list, add_messages]
```

Here, `add_messages` is the reducer function. LangGraph reads this annotation and knows: "whenever a node returns a new state with `messages`, use `add_messages` to combine it with the existing messages."

---

## Immutable State

State in LangGraph is **immutable** — once created, you never change its contents. This is fundamental to LangGraph's ability to:

- Maintain snapshots for time-travel/checkpointing
- Run nodes in parallel safely
- Reason about state transitions predictably
- Always be able to go back to any prior snapshot

**Pattern:** A node receives an old state, creates a **new** state object with updated values, and returns it.

```python
# CORRECT — create and return a new state
def my_counting_node(old_state: State) -> State:
    count = old_state.count  # Read from old state
    count += 1               # Compute new value
    new_state = State(count=count)  # Create a NEW state object with updated value
    return new_state                # Return the new state (old_state unchanged)

# WRONG — never mutate the old state
def my_counting_node(old_state: State) -> State:
    old_state.count += 1  # ❌ NEVER DO THIS — breaks checkpointing & parallelism
    return old_state      # ❌ Returning the same mutated object
```

---

## The 5 Steps to Build a Graph

When you run a LangGraph application, there are **two phases**:

1. **Graph building phase** — you define the structure (steps 1–5 below). This is a "meta phase" where you describe what you want to do.
2. **Execution phase** — you invoke the compiled graph and it actually runs.

This is unusual compared to normal programming. You don't normally have a phase where you're describing what you want to do and then a separate phase where it does it. But that's how LangGraph works — both phases happen at runtime when you start your application.

| Step | Action | What happens |
|------|--------|--------------|
| 1 | Define the State class | Describe what information will be maintained (includes reducer specification) |
| 2 | Start the Graph Builder | Initialize `StateGraph` with your State **class** (not an instance) |
| 3 | Create Node(s) | Write Python functions, register them with the builder via `add_node()` |
| 4 | Create Edge(s) | Define connections between nodes (and START/END) via `add_edge()` |
| 5 | Compile the Graph | Call `.compile()` — graph is now ready to execute |

After compilation, you **invoke** the graph with an initial state to run it. Steps 3 and 4 may be repeated many times to lay out complex workflows — you're "laying out the story" of what you want your agent system to do before it's actually live.

---

## Example 1: Silly Random Node (No LLM)

This example demonstrates that **LangGraph nodes are just Python functions** — no LLM required.

### Step 1: Define the State

```python
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages

# State class defines the shape of data flowing through the graph
# Using Pydantic BaseModel for validation (TypedDict also works)
class State(BaseModel):
    # messages: a list of conversation messages
    # add_messages: the reducer that concatenates new messages onto existing ones
    messages: Annotated[list, add_messages]
```

- Uses Pydantic `BaseModel` (could also use `TypedDict` — both are common)
- Single field `messages` — a list with the `add_messages` reducer
- State objects can be any Python object, but Pydantic and TypedDict are most common

#### Pydantic BaseModel vs TypedDict — When to Use Which

Both are valid choices for defining State in LangGraph. The difference matters:

| | `Pydantic BaseModel` | `TypedDict` |
|---|---|---|
| **Validation** | Runtime type validation — raises errors if you pass wrong types | No validation — just type hints for static checkers |
| **Defaults** | Supports default values for fields | No defaults (all fields required unless `Optional`) |
| **Access style** | Attribute access: `state.messages` | Dict access: `state["messages"]` |
| **Overhead** | Slight runtime cost for validation | Zero overhead — it's just a dict |
| **Serialization** | Built-in `.model_dump()`, `.model_json_schema()` | Manual — it's a plain dict already |
| **When to use** | When you want safety, validation, complex state with many fields | When you want simplicity, speed, or minimal boilerplate |

```python
# Pydantic approach — validates types, attribute access
from pydantic import BaseModel

class State(BaseModel):
    messages: Annotated[list, add_messages]

# Usage in node: state.messages

# TypedDict approach — no validation, dict access
from typing import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Usage in node: state["messages"]
```

In practice: **TypedDict is more common** in LangGraph examples because it's lighter and nodes often return plain dicts (`{"messages": [...]}`) which map naturally. Pydantic is better when your state grows complex and you want validation guarantees.

### Step 2: Start the Graph Builder

```python
from langgraph.graph import StateGraph

# Pass the State CLASS (not an instance) — tells the builder what shape state takes
# This begins the graph building process; nothing executes yet
graph_builder = StateGraph(State)
```

Note: we pass the **class** `State`, not an instance. We're not creating a state with messages — we're telling the builder what *shape* state will take. This begins the graph building process (nothing is running yet).

### Step 3: Create a Node

```python
import random

nouns = ["Cabbages", "Unicorns", "Toasters", "Penguins", "Bananas",
         "Zombies", "Rainbows", "Eels", "Pickles", "Muffins"]
adjectives = ["outrageous", "smelly", "pedantic", "existential", "moody",
              "sparkly", "untrustworthy", "sarcastic", "squishy", "haunted"]

# A node is just a Python function: takes old state → returns new state
def our_first_node(old_state: State) -> State:
    # Generate a silly random reply (no LLM involved!)
    reply = f"{random.choice(nouns)} are {random.choice(adjectives)}"
    # Wrap in standard OpenAI message format (reducer will convert to AIMessage)
    messages = [{"role": "assistant", "content": reply}]
    # Create a NEW state object (never mutate old_state)
    new_state = State(messages=messages)
    return new_state

# Register the function as a named node in the graph builder
graph_builder.add_node("first_node", our_first_node)
```

Key observations:
- The node is a plain Python function — nothing special
- It receives `old_state` but doesn't even use it (grayed out in the IDE) — proving nodes are just functions that don't need to touch the LLM or even the prior state
- It creates a **new** `State` object (immutability respected)
- The message uses standard OpenAI dict format — the `add_messages` reducer will package it into an `AIMessage` object
- `add_node("first_node", our_first_node)` registers it with a name in the graph builder

### Step 4: Create Edges

```python
from langgraph.graph import START, END

# START and END are special LangGraph constants representing workflow boundaries
graph_builder.add_edge(START, "first_node")   # When graph starts → run first_node
graph_builder.add_edge("first_node", END)     # After first_node → workflow ends
```

- `START` and `END` are special constants imported from LangGraph — they signify the beginning and end of the workflow
- This defines: START → first_node → END (linear flow)

### Step 5: Compile the Graph

```python
# Step 5: Compile — finalizes the graph, making it ready to invoke
graph = graph_builder.compile()
```

Optionally visualize:

```python
from IPython.display import Image, display
# Renders the graph as a visual diagram (uses Mermaid under the hood)
display(Image(graph.get_graph().draw_mermaid_png()))
```

This produces: `__start__` → `first_node` → `__end__`

### Running It

```python
import gradio as gr

# Gradio chat function: receives user's current input + conversation history
def chat(user_input: str, history):
    # Format user input as standard OpenAI message dict
    message = {"role": "user", "content": user_input}
    messages = [message]
    # Create initial state with the user's message
    state = State(messages=messages)
    # Execute the graph — this runs all nodes/edges and returns final state
    result = graph.invoke(state)
    print(result)  # Debug: see the full state structure returned
    # Extract the last message's text content (the assistant's reply)
    # result['messages'] = list of HumanMessage/AIMessage objects
    # [-1] = last message (the AI response)
    # .content = the actual text string
    return result["messages"][-1].content

# Launch Gradio chat UI, wiring it to our chat function
gr.ChatInterface(chat, type="messages").launch()
```

**`graph.invoke(state)`** — this is the key LangGraph method. You invoke a graph with a state to execute it. `invoke` is also the standard LangChain word for calling things.

**Output structure:** The result from `invoke` contains messages packaged as `HumanMessage` and `AIMessage` objects (LangGraph/LangChain constructs). The `add_messages` reducer does this packaging automatically — it doesn't just concatenate, it also wraps raw dicts into proper message objects:

```python
{'messages': [
    HumanMessage(content='hello', id='ab9e31e0-...'),
    AIMessage(content='Eels are pedantic', id='e09fe6c5-...')
]}
```

**The point:** LangGraph is all about Python functions. Nodes don't need LLMs. The graph machinery (state, nodes, edges, reducers) works independently of what the nodes actually do.

---

## Example 2: Real LLM Chatbot

Now we add an actual LLM call. Same 5 steps, but the node calls an LLM instead of picking random words.

### Full Implementation

```python
import os
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (OPENROUTER_API_KEY, OPENROUTER_BASE_URL)
# override=True ensures .env values take precedence over existing env vars
load_dotenv(override=True)

# Step 1: Define the State object — same as before
class State(BaseModel):
    messages: Annotated[list, add_messages]

# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)

# Step 3: Create a Node — this time with a real LLM
# ChatOpenAI is a LangChain wrapper; we point it at OpenRouter
llm = ChatOpenAI(
    model="gpt-4o-mini",                          # Model to use (via OpenRouter)
    base_url=os.environ["OPENROUTER_BASE_URL"],   # OpenRouter API endpoint
    api_key=os.environ["OPENROUTER_API_KEY"]      # OpenRouter API key
)

def chatbot_node(old_state: State) -> State:
    # Pass the full message history to the LLM — it sees the conversation context
    response = llm.invoke(old_state.messages)
    # response is an AIMessage object; wrap in list for the reducer
    new_state = State(messages=[response])
    return new_state

# Register the node with name "chatbot"
graph_builder.add_node("chatbot", chatbot_node)

# Step 4: Create Edges — simple linear flow
graph_builder.add_edge(START, "chatbot")  # Start → chatbot
graph_builder.add_edge("chatbot", END)    # chatbot → End

# Step 5: Compile the Graph — ready to invoke
graph = graph_builder.compile()
```

### The LLM Integration

```python
# ChatOpenAI: LangChain's wrapper for OpenAI-compatible APIs
# Works with any provider that exposes an OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="gpt-4o-mini",                          # Which model to request
    base_url=os.environ["OPENROUTER_BASE_URL"],   # API endpoint (OpenRouter here)
    api_key=os.environ["OPENROUTER_API_KEY"]      # Auth key from .env
)
```

- `ChatOpenAI` is from **LangChain** (`langchain_openai` package) — the sibling product
- You **don't need** LangChain for this — you could call any LLM directly, use OpenAI SDK, etc.
- LangChain is optional but convenient, and most community examples use it
- Here we route through OpenRouter (any OpenAI-compatible endpoint works)

### The Node Function

```python
def chatbot_node(old_state: State) -> State:
    # invoke() sends messages to the LLM and returns an AIMessage object
    response = llm.invoke(old_state.messages)
    # Wrap response in a list — the add_messages reducer expects a list
    # The reducer will append this to the existing messages in state
    new_state = State(messages=[response])
    return new_state
```

- `llm.invoke(old_state.messages)` — passes the full message history to the LLM (again, `invoke` is the LangChain/LangGraph word)
- `response` is an `AIMessage` object (LangChain's wrapper)
- We wrap it in a list `[response]` because our state's `messages` field expects a list
- The `add_messages` reducer will concatenate this with existing messages

### Key Differences from Example 1

| Aspect | Example 1 (Random) | Example 2 (LLM) |
|--------|-------------------|------------------|
| Node logic | Random string generation | `llm.invoke(old_state.messages)` |
| Uses old_state? | No (ignored) | Yes — passes `old_state.messages` to LLM |
| Response type | Dict `{"role": "assistant", "content": ...}` | LangChain `AIMessage` object directly |

### Running with Gradio

```python
import gradio as gr

def chat(user_input: str, history):
    # Create a fresh state with ONLY the current user message
    # (no history carried over — this is the limitation we'll fix in Part 2)
    initial_state = State(messages=[{"role": "user", "content": user_input}])
    # Invoke the compiled graph — executes START → chatbot → END
    result = graph.invoke(initial_state)
    print(result)  # Debug: shows HumanMessage + AIMessage objects
    # result['messages'][-1] = the last message = AI's response
    # .content = extract the text string from the AIMessage object
    return result['messages'][-1].content

# Launch the Gradio chat UI
gr.ChatInterface(chat, type="messages").launch()
```

### Current Limitation: No Memory

This implementation has **no conversation history**. Each invocation creates a fresh state with only the current user message:

```
User: My name is Mo
Bot: Nice to meet you, Mo! How can I assist you today?
User: What's my name?
Bot: I'm sorry, but I don't have access to your personal data.
```

The graph is invoked fresh each time — there's nothing persisting the conversation across calls. The state only contains the single message from the current turn. This is addressed in Part 2 (next session) along with:
- Conversation memory/history
- Tool integration
- Conditional edges

---

## Key Takeaways

1. **LangGraph ≠ LangChain** — LangGraph is an independent framework for orchestrating agent workflows as graphs. LangChain is optional glue code for LLM calls.

2. **Everything is a Python function** — Nodes are functions. Conditional edges are functions. There's no magic.

3. **State is immutable** — Always create new state objects; never mutate the old one. This enables time-travel, checkpointing, and safe parallelism.

4. **Reducers handle concurrency** — They define how to merge state updates when multiple nodes run in parallel. `add_messages` simply concatenates lists and packages dicts into message objects.

5. **Two-phase execution** — First you *build* the graph (define state, nodes, edges, compile). Then you *invoke* it. Both phases happen at runtime.

6. **`graph.invoke(state)`** — The single method that executes your entire workflow.

7. **Nodes don't require LLMs** — Any Python function works. LLMs are just one possible thing a node can do.

8. **Understand the abstractions** — Per Anthropic's advice, know what's happening under the hood. LangGraph adds structure but also adds layers between you and the LLM. Start simple, add complexity only when it demonstrably improves outcomes.

9. **Anthropic's design patterns** (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) map naturally onto LangGraph's node/edge architecture — but can also be implemented in a few lines of code without any framework.

10. **`load_dotenv()`** — CrewAI did this automatically; with LangGraph you call it explicitly to load your `.env` file.

---

# LangGraph — Part 2: Tools, Conditional Edges & Checkpointing

## Table of Contents (Part 2)
- [Super Steps](#super-steps)
- [LangSmith Setup & Tracing](#langsmith-setup--tracing)
- [Tools in LangGraph](#tools-in-langgraph)
  - [Off-the-Shelf Tool: Google Search](#off-the-shelf-tool-google-search)
  - [Custom Tool: Push Notification](#custom-tool-push-notification)
  - [The LangChain Tool Wrapper](#the-langchain-tool-wrapper)
- [Building a Graph with Tools](#building-a-graph-with-tools)
  - [bind_tools: Providing Tools to the LLM](#bind_tools-providing-tools-to-the-llm)
  - [ToolNode: Handling Tool Execution](#toolnode-handling-tool-execution)
  - [Conditional Edges: tools_condition](#conditional-edges-tools_condition)
  - [The Tool Loop: tools → chatbot](#the-tool-loop-tools--chatbot)
- [Checkpointing & Memory](#checkpointing--memory)
  - [Why State Alone Isn't Enough](#why-state-alone-isnt-enough)
  - [MemorySaver: In-Memory Checkpointing](#memorysaver-in-memory-checkpointing)
  - [Thread IDs & Config](#thread-ids--config)
  - [get_state & get_state_history](#get_state--get_state_history)
  - [Time Travel](#time-travel)
  - [SQLite Persistence](#sqlite-persistence)

---

## Super Steps

A **super step** is a single complete invocation of the graph — one call to `graph.invoke()`.

This is a crucial concept that's easy to misunderstand:

- Every user interaction = a fresh `graph.invoke()` call = one super step
- The graph describes what happens within ONE super step (agents calling tools, multiple nodes running)
- The **reducer** handles state within a single super step (combining outputs from parallel nodes)
- The reducer does **NOT** handle state between super steps — that's what checkpointing does

```
Define Graph → [Super Step 1: user asks] → [Super Step 2: user follows up] → [Super Step 3: ...]
                     ↑                            ↑                              ↑
              graph.invoke()               graph.invoke()                  graph.invoke()
```

Each super step runs the entire graph from START to END (or until a conditional edge routes to END).

---

## LangSmith Setup & Tracing

LangSmith provides observability into every graph invocation. Setup:

1. Create a free account at https://langsmith.com
2. Generate an API key via "Setup Tracing"
3. Add to your `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=langgraph-course
LANGSMITH_API_KEY=lsv2_pt_your_key_here
```

You'll also need `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `SERPER_API_KEY` in the same `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
SERPER_API_KEY=your_serper_key_here
```

4. Run `load_dotenv(override=True)` — tracing activates automatically

**What you see in LangSmith:**
- Every `invoke()` call logged with input/output
- Latency per call
- Cost per call (fractions of a cent for gpt-4o-mini)
- Token counts
- Full trace of node execution (chatbot → tools_condition → tools → chatbot → ...)
- Errors highlighted in red

---

## Tools in LangGraph

When implementing tools, you always need to handle **two concerns**:

1. **Providing tools to the LLM** — building the JSON schema so the model knows what it can call
2. **Handling tool results** — detecting `finish_reason == "tool_calls"`, executing the function, feeding results back

LangGraph/LangChain abstracts both of these.

### Off-the-Shelf Tool: Google Search

```python
from langchain_community.utilities import GoogleSerperAPIWrapper

# Create the search wrapper (uses SERPER_API_KEY from .env)
serper = GoogleSerperAPIWrapper()
serper.run("What is the capital of France?")  # Returns: "Paris"
```

This is a LangChain community utility — a convenient wrapper around the Serper API (same one used in prior sections). Free tier includes several thousand calls.

### Custom Tool: Push Notification

```python
def push(text: str):
    """Send a push notification to the user"""
    # In production: requests.post(pushover_url, data={...})
    print(f'push notification has been sent with text : {text}')
```

### The LangChain Tool Wrapper

LangChain's `Tool` class wraps any function into a tool object that handles all the JSON schema generation:

```python
from langchain.agents import Tool

# Off-the-shelf tool: wraps serper.run
tool_search = Tool(
    name="search",                    # Name the LLM will see
    func=serper.run,                  # Function to execute
    description="Useful for when you need more information from an online search"
)

# Custom tool: wraps our push function
tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="useful for when you want to send a push notification"
)

# Test via LangChain's invoke
tool_search.invoke("What is the capital of France?")  # "Paris"
tool_push.invoke("Hello, me")  # prints notification

# Combine into a list for the graph
tools = [tool_search, tool_push]
```

---

## Building a Graph with Tools

### Full Implementation

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import os

# Step 1: State — using TypedDict this time (alternative to Pydantic BaseModel)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Step 2: Graph Builder
graph_builder = StateGraph(State)

# Step 3: Create LLM and bind tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)
# bind_tools creates a version of the LLM that automatically includes
# tool JSON schemas in every request — handles concern #1
llm_with_tools = llm.bind_tools(tools)

# Chatbot node — uses llm_with_tools instead of plain llm
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Register nodes
graph_builder.add_node("chatbot", chatbot)
# ToolNode is a pre-built node that handles concern #2:
# detects tool_calls in the response, executes the matching function, returns results
graph_builder.add_node("tools", ToolNode(tools=tools))

# Step 4: Edges
# Conditional edge: only go to "tools" if the LLM requested a tool call
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
# After tools execute, ALWAYS return to chatbot (it needs to process the results)
graph_builder.add_edge("tools", "chatbot")
# Entry point
graph_builder.add_edge(START, "chatbot")

# Step 5: Compile
graph = graph_builder.compile()
```

### bind_tools: Providing Tools to the LLM

```python
# This is LangChain magic — creates a wrapped LLM that automatically:
# 1. Inspects each tool's name, description, and function signature
# 2. Builds the JSON schema for each tool
# 3. Includes it in every API request to the LLM
llm_with_tools = llm.bind_tools(tools)
```

The flip side: it hides the implementation, making debugging harder. But it eliminates all the manual JSON construction from [Agentic AI Systems Foundation](../01-Agentic%20AI%20Systems%20Foundation/).

### ToolNode: Handling Tool Execution

```python
# ToolNode is a pre-built LangGraph node that:
# 1. Reads the AIMessage from state
# 2. Checks if it contains tool_calls
# 3. Executes the matching tool function with the provided arguments
# 4. Returns a ToolMessage with the result
graph_builder.add_node("tools", ToolNode(tools=tools))
```

This replaces the manual `handle_tool_call` logic from [Agentic AI Systems Foundation](../01-Agentic%20AI%20Systems%20Foundation/).

### Conditional Edges: tools_condition

```python
# tools_condition is a pre-built function that checks:
# "Did the LLM's response have finish_reason == 'tool_calls'?"
# If yes → route to "tools" node
# If no → route to END (LangGraph adds this automatically)
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
```

This is the **if statement** — the same `if finish_reason == "tool_calls"` from [Agentic AI Systems Foundation](../01-Agentic%20AI%20Systems%20Foundation/), but abstracted into a reusable conditional edge.

**Why no explicit `add_edge("chatbot", END)`?** When you use `add_conditional_edges`, LangGraph automatically adds an END route for any unresolved condition. Since `tools_condition` only routes to `"tools"` when there's a tool call, the implicit "else" case routes to END. LangGraph handles this — you don't need to declare it manually.

> **Part 1 vs Part 2:** In Part 1 (simple edges only), you *must* explicitly add `add_edge("chatbot", END)` because there's no conditional logic — LangGraph has no way to infer when the graph should terminate. The automatic END fallback only exists with `add_conditional_edges`.

### The Tool Loop: tools → chatbot

```python
# After tools execute, results must go BACK to the chatbot
# The chatbot needs to see the tool output and decide what to do next
# (maybe call another tool, or give a final answer)
graph_builder.add_edge("tools", "chatbot")
```

This creates a **cycle** in the graph: chatbot → tools → chatbot → tools → ... until the LLM stops requesting tools. This is exactly why LangGraph supports cyclic graphs (unlike LCEL).

The resulting graph visualization:
- `__start__` → `chatbot` → (conditional: tools_condition) → `tools` → `chatbot` → ... → `__end__`
- Solid line from tools → chatbot (always)
- Dotted line from chatbot → end (only when no tool call)

---

## Checkpointing & Memory

### Why State Alone Isn't Enough

Despite having reducers and state management, the graph has **no memory between super steps**:

```
User: My name is Mo → Bot: Nice to meet you, Mo!
User: What's my name? → Bot: I don't have access to your personal information.
```

Each `graph.invoke()` is a fresh invocation. The reducer manages state *within* one super step (parallel nodes, tool loops), but not *across* super steps. That's what **checkpointing** solves.

### MemorySaver: In-Memory Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

# Create a checkpointer (stores state in RAM)
memory = MemorySaver()

# Same graph code as before, but compile with checkpointer
graph = graph_builder.compile(checkpointer=memory)
```

That's it. One argument to `.compile()`. The graph now automatically saves state after each super step and restores it on the next invocation.

### Thread IDs & Config

```python
# Config identifies WHICH conversation thread to checkpoint
config = {"configurable": {"thread_id": "1"}}

def chat(user_input: str, history):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config  # Pass config to associate with this thread
    )
    return result["messages"][-1].content
```

- `thread_id` = a conversation thread (not a technical thread)
- Different thread IDs = separate memory slots
- Same thread ID = continuous conversation with full history

### get_state & get_state_history

```python
# Get the current state snapshot for a thread
graph.get_state(config)
# Returns: StateSnapshot with full message history, config, metadata, created_at

# Get ALL historical snapshots (most recent first)
list(graph.get_state_history(config))
# Returns: list of StateSnapshot objects — one per step, going back in time
```

### Time Travel

You can rewind to any prior checkpoint and replay from there:

```python
# Pick a checkpoint_id from get_state_history
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1f1462f5-7d8d-6a1c-8001-166eed0d778a"  # A prior moment
    }
}
# Invoke from that point — graph resumes as if time-traveled
graph.invoke(None, config=config)
```

This enables:
- **Recovery** — if something fails, restart from any prior checkpoint
- **Reproducibility** — replay exact state at any point in time
- **Branching** — fork a conversation from a prior state

### SQLite Persistence

Switch from in-memory to persistent storage by changing one import:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Connect to SQLite (persists to disk)
sql_memory = SqliteSaver(sqlite3.connect("memory.db", check_same_thread=False))

# Rebuild graph with SQL checkpointer — everything else identical
graph = graph_builder.compile(checkpointer=sql_memory)
```

Now memory survives kernel restarts. The SQLite database files appear in your working directory. Changing `MemorySaver` → `SqliteSaver` is literally the only code change needed.

---

## Part 2 Key Takeaways

1. **Super steps** — each `graph.invoke()` is one super step. Reducers work within a super step; checkpointing works across them.

2. **Two concerns with tools** — (1) providing tool schemas to the LLM (`bind_tools`), (2) executing tool calls (`ToolNode`). LangGraph abstracts both.

3. **Conditional edges** — `tools_condition` is the "if statement" that routes to tools only when the LLM requests them. This creates the agent loop.

4. **The tool loop** — `tools → chatbot` edge creates a cycle. The chatbot keeps running until it stops requesting tools.

5. **Checkpointing is elegant** — one argument to `.compile()` adds full conversation memory, time travel, and state history.

6. **Thread IDs** — separate conversations via config. Same code, different memory slots.

7. **Persistence is trivial** — swap `MemorySaver` for `SqliteSaver` to survive restarts. One line change.

8. **LangSmith** — automatic tracing shows the full execution chain, costs, latency, and errors. Essential for debugging tool loops.
