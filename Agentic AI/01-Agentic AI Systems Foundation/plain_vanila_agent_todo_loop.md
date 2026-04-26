# The Unreasonable Effectiveness of the Agent Loop

## What is an Agent?

The definition of "agent" has evolved over time:

1. **Sam Altman / OpenAI (early days):** AI systems that can do work for you independently.
2. **Anthropic / Hugging Face (early 2025):** A system in which an LLM controls the workflow.
3. **Emerging consensus (2026):** An LLM that runs tools in a loop to achieve a goal.

The third definition is what we'll build from scratch in this tutorial. It sounds sophisticated, but the implementation is surprisingly simple — a `while` loop that repeatedly calls an LLM, lets it use tools, and feeds the results back until it's done.

What it *feels* like: you kick off the agent and it goes off planning, executing, and tracking progress autonomously.

What's *actually* happening: your code is calling an LLM repeatedly in a loop. The LLM just generates tokens. You interpret those tokens as tool calls, execute them, and give the LLM the illusion of agency.

## Setup

```python
from rich.console import Console
from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv(override=True)
```

We use the `rich` library for colored, formatted terminal output. `Console().print()` renders Rich markup like `[green]`, `[strike]`, etc.

A small utility to safely display output:

```python
def show(text):
    try:
        Console().print(text)
    except Exception:
        print(text)
```

Initialize the OpenAI client:

```python
openai = OpenAI()
```

## Step 1: Build a Simple Todo List (Plain Python)

Before any AI, we build two vanilla Python functions to manage a todo list. This is the "tool" the LLM will eventually use.

Two lists track state:

```python
todos = []
completed = []
```

A reporting function that prints the current state with Rich markup — completed items get a green strikethrough:

```python
def get_todo_report() -> str:
    result = ""
    for index, todo in enumerate(todos):
        if completed[index]:
            result += f"Todo #{index + 1}: [green][strike]{todo}[/strike][/green]\n"
        else:
            result += f"Todo #{index + 1}: {todo}\n"
    show(result)
    return result
```

A function to add new todos:

```python
def create_todos(descriptions: list[str]) -> str:
    todos.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_todo_report()
```

A function to mark a todo as complete (1-based index):

```python
def mark_complete(index: int, completion_notes: str) -> str:
    if 1 <= index <= len(todos):
        completed[index - 1] = True
    else:
        return "No todo at this index."
    Console().print(completion_notes)
    return get_todo_report()
```

### Testing it manually

```python
todos, completed = [], []
create_todos(["Buy groceries", "Finish extra lab", "Eat banana"])
```

Output:

```
Todo #1: Buy groceries
Todo #2: Finish extra lab
Todo #3: Eat banana
```

```python
mark_complete(1, "bought")
```

Output:

```
bought
Todo #1: ~~Buy groceries~~     (green, struck through)
Todo #2: Finish extra lab
Todo #3: Eat banana
```

Nothing about AI here — just vanilla Python managing a list. But we're about to hand these functions to an LLM as tools.

## Step 2: Define Tool Schemas (JSON)

The OpenAI API needs JSON schemas describing each tool so the LLM knows what's available and what arguments to pass.

```python
create_todos_json = {
    "name": "create_todos",
    "description": "Add new todos from a list of descriptions and return the full list",
    "parameters": {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Descriptions"
            }
        },
        "required": ["descriptions"],
        "additionalProperties": False
    }
}
```

```python
mark_complete_json = {
    "name": "mark_complete",
    "description": "Mark complete the todo at the given position (starting from 1) and return the full list",
    "parameters": {
        "properties": {
            "index": {
                "description": "The 1-based index of the todo to mark as complete",
                "title": "Index",
                "type": "integer"
            },
            "completion_notes": {
                "description": "Notes about how you completed the todo in rich console markup",
                "title": "Completion Notes",
                "type": "string"
            }
        },
        "required": ["index", "completion_notes"],
        "type": "object",
        "additionalProperties": False
    }
}
```

Bundle them into the tools list:

```python
tools = [
    {"type": "function", "function": create_todos_json},
    {"type": "function", "function": mark_complete_json}
]
```

## Step 3: The Tool Executor

This function takes the LLM's tool call requests, looks up the actual Python function by name, calls it, and packages the result back into the message format the API expects:

```python
def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results
```

The `globals().get(tool_name)` trick avoids a big if/elif chain — it dynamically looks up the function by name from the global scope.

## Step 4: The Agent Loop

This is the core of the entire agent. It's a `while` loop that keeps calling the LLM until it stops requesting tools:

```python
def loop(messages):
    done = False
    while not done:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    show(response.choices[0].message.content)
```

How it works:

1. Call the LLM with the current messages and available tools.
2. Check `finish_reason`:
   - `"tool_calls"` → the LLM wants to use a tool. Execute it, append the assistant's message and tool results to the conversation, and loop again.
   - `"stop"` → the LLM is done. Print the final answer and exit.
3. Each iteration, the LLM sees the full conversation history including all previous tool results, so it can plan its next step.

## Step 5: Run It

Set up the system prompt that instructs the LLM to plan with todos, then execute each step:

```python
system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps,
then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Rich console markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""

user_message = """
A train leaves Boston at 2:00 pm traveling 60 mph.
Another train leaves New York at 3:00 pm traveling 80 mph toward Boston.
When do they meet?
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
]
```

Launch the agent:

```python
todos, completed = [], []
loop(messages)
```

### Output

The LLM autonomously creates a plan, then works through each step, crossing items off as it goes:

```
Todo #1: Define the variables for speed and time for both trains.
Todo #2: Calculate the time difference between the train leaving Boston and the train leaving New York.
Todo #3: Set up the distance equations for both trains to determine when they meet.
Todo #4: Solve the equations to find the time they meet.
```

Then it starts executing — each step gets marked complete with a green strikethrough:

```
Defined variables: Train A (Boston) speed = 60 mph, Train B (New York) speed = 80 mph.
Let t be the time in hours after 2:00 pm when they meet.

Todo #1: ~~Define the variables for speed and time for both trains.~~
Todo #2: Calculate the time difference...
...
```

It continues creating sub-steps, solving equations, and crossing off items until it arrives at the final answer:

```
The two trains will meet approximately at 4:17 pm.
```

All of this happened autonomously — the LLM decided what todos to create, what order to work through them, and when it was done.

## Key Takeaways

- **An agent loop is just a `while` loop.** Call the LLM, check if it wants to use tools, execute them, feed results back, repeat.
- **The LLM is the decision maker, your code is the executor.** The LLM decides *which* tool to call and *with what arguments*. Your code actually runs the function.
- **It's shockingly simple to build.** The entire agent is ~10 lines of loop code plus a tool executor. No frameworks needed.
- **Better outcomes through iteration.** Forcing the LLM to plan steps and work through them one by one produces better results than asking for a single-shot answer — the LLM reasons its way through the problem.
- **Tools can be anything.** Here we used a simple todo list, but tools could be file creation, API calls, database queries, calculations — anything you can write a Python function for.

## Exercise

Build your own agent loop from scratch in a new notebook. Refer back to this tutorial as needed, but type the core loop yourself. Try different problems and add extra tools. It's one of the most satisfying things to see come to life.
