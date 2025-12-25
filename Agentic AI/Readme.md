# Agentic AI Systems Introduction

  

## Table of Contents

1.  [Core Concepts & Definitions](#core-concepts--definitions)

2.  [Agentic System Architecture](#agentic-system-architecture)

3.  [Workflow Design Patterns](#workflow-design-patterns)

4.  [Agent Design Patterns](#agent-design-patterns)

5.  [Multi-Model API Integration](#multi-model-api-integration)

6.  [Production Considerations](#production-considerations)

7.  [Code Examples & Implementation](#code-examples--implementation)

  

---

  

## Core Concepts & Definitions

  

### What is an AI Agent?

  

**Primary Definition (Hugging Face)**: AI agents are programs where LLM outputs control the workflow.

  

### Five Hallmarks of Agentic AI

  

1.  **Multiple LLM Calls**: Solutions involving sequential or parallel LLM interactions

2.  **Tool Use**: LLMs with ability to interact with external tools and APIs

3.  **Multi-Agent Coordination**: Environment allowing different LLMs to exchange information

4.  **Planning & Orchestration**: LLM-based coordination of activities and task sequencing

5.  **Autonomy**: LLMs controlling execution order and decision-making processes

  

### Anthropic's Classification Framework

  

**Agentic Systems** (umbrella term) contain two categories:

  

-  **Workflows**: Systems where models and tools are orchestrated through predefined paths

-  **Agents**: Models dynamically direct their own processes and tools, maintaining control over task accomplishment

  

---

  

## Agentic System Architecture

  

### Workflow vs Agent Distinction

  

| Aspect | Workflows | Agents |

|--------|-----------|---------|

|  **Path**  | Predefined, fixed sequence | Dynamic, self-directed |

|  **Control**  | External orchestration | Internal model control |

|  **Predictability**  | High | Variable |

|  **Flexibility**  | Limited | High |

|  **Complexity**  | Lower | Higher |

  

---

  

## Workflow Design Patterns

  

### 1. Prompt Chaining

![Prompt Chaining](./img/prompt_chaining.jpg)

  

**Architecture**: Sequential LLM calls with optional code processing between steps

  

**Use Cases**:

- Multi-step reasoning tasks

- Content generation pipelines

- Business analysis workflows

  

**Implementation Pattern**:

```python

# Step 1: Generate business sector

messages_1 =  [{"role":  "user",  "content":  "Pick a business sector for analysis"}]

response_1 = openai.chat.completions.create(model="gpt-4.1-mini",  messages=messages_1)

sector = response_1.choices[0].message.content

  

# Step 2: Identify pain points

messages_2 =  [{"role":  "user",  "content":  f"Identify pain points in {sector}"}]

response_2 = openai.chat.completions.create(model="gpt-4.1-mini",  messages=messages_2)

pain_points = response_2.choices[0].message.content

  

# Step 3: Propose solutions

messages_3 =  [{"role":  "user",  "content":  f"Propose AI solutions for {pain_points}"}]

response_3 = openai.chat.completions.create(model="gpt-4.1-mini",  messages=messages_3)

```

  

**Benefits**:

- Precise prompt framing for each step

- Maintained workflow guardrails

- Decomposed complex tasks into manageable subtasks

  

### 2. Routing

![Routing](./img/Routing.jpg)

  

**Architecture**: Router LLM classifies tasks and directs to specialist models

  

**Implementation Concept**:

```python

def  route_request(user_input):

router_prompt =  f"Classify this request and route to appropriate specialist: {user_input}"

routing_decision = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": router_prompt}]

)

specialist = routing_decision.choices[0].message.content

return specialist_models[specialist].process(user_input)

```

  

**Benefits**:

- Separation of concerns

- Specialized model expertise

- Efficient resource utilization

  

### 3. Parallelization

![Parallelization](./img/Parallelization.jpg)

  

**Architecture**: Code-driven task decomposition with concurrent LLM processing

  

**Implementation Pattern**:

```python

import asyncio

  

async  def  parallel_processing(task):

# Code decomposes task

subtasks = decompose_task(task)

# Parallel LLM calls

tasks =  [

openai.chat.completions.create(model="gpt-4.1-mini",  messages=[{"role":  "user",  "content": subtask}])

for subtask in subtasks

]

results =  await asyncio.gather(*tasks)

# Code aggregates results

return aggregate_results([r.choices[0].message.content for r in results])

```

  

### 4. Orchestrator-Worker

![Orchestrator Worker](./img/Orchestrator-worker.jpg)

  

**Architecture**: LLM orchestrator dynamically manages task breakdown and result synthesis

  

**Key Difference from Parallelization**: LLM (not code) handles orchestration decisions

  

**Implementation Concept**:

```python

def  orchestrator_worker_pattern(complex_task):

# LLM orchestrator breaks down task

orchestrator_prompt =  f"Break down this complex task into subtasks: {complex_task}"

breakdown = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": orchestrator_prompt}]

)

subtasks = parse_subtasks(breakdown.choices[0].message.content)

# Worker LLMs process subtasks

worker_results =  []

for subtask in subtasks:

result = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": subtask}]

)

worker_results.append(result.choices[0].message.content)

# LLM orchestrator synthesizes results

synthesis_prompt =  f"Synthesize these results: {worker_results}"

final_result = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": synthesis_prompt}]

)

return final_result.choices[0].message.content

```

  

### 5. Evaluator-Optimizer

![Evaluator Optimizer](./img/Evaluator-optimazor.jpg)

  

**Architecture**: Generator-Evaluator feedback loop with iterative improvement

  

**Implementation Pattern**:

```python

def  evaluator_optimizer_pattern(task,  max_iterations=3):

for iteration in  range(max_iterations):

# Generator LLM creates solution

generator_response = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": task}]

)

solution = generator_response.choices[0].message.content

# Evaluator LLM checks quality

evaluator_prompt =  f"Evaluate this solution and provide feedback: {solution}"

evaluator_response = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": evaluator_prompt}]

)

evaluation = evaluator_response.choices[0].message.content

if  "acceptable"  in evaluation.lower():

return solution

else:

# Incorporate feedback for next iteration

task =  f"{task}\n\nPrevious attempt: {solution}\nFeedback: {evaluation}\nImprove the solution."

return solution # Return best attempt after max iterations

```

  

**Benefits**:

- Improved accuracy and robustness

- Quality assurance through validation

- Iterative refinement capabilities

  

---

  

## Agent Design Patterns

  

### Core Characteristics

![Agent Interaction](./img/Agents-Interaction.jpg)

  

**Key Features**:

-  **Open-ended processes**: No fixed termination point

-  **Feedback loops**: Continuous information exchange

-  **Dynamic paths**: No predetermined execution sequence

-  **Environment interaction**: Bidirectional communication with external systems

  

**Generic Agent Loop**:

```python

def  agent_loop(initial_request,  environment):

current_state = initial_request

while  not should_terminate(current_state):

# LLM decides next action

action_decision = openai.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content":  f"Given state: {current_state}, what action should I take?"}]

)

action = action_decision.choices[0].message.content

# Execute action on environment

result = environment.execute(action)

# Update state based on result

current_state = update_state(current_state, result)

return current_state

```

  

### Challenges with Agent Patterns

  

1.  **Unpredictable execution paths**

2.  **Variable output quality**

3.  **Unknown completion times**

4.  **Unpredictable costs**

5.  **Potential infinite loops**

  

### Mitigation Strategies

  

**Monitoring**: Comprehensive visibility into agent interactions and decision-making processes

  

**Guardrails**: Software protections ensuring agents operate within defined boundaries

  

---

  

## Multi-Model API Integration

  

### Supported Models and APIs

  

#### OpenAI Integration

```python

from openai import OpenAI

import os

from dotenv import load_dotenv

  

load_dotenv(override=True)

openai = OpenAI()

  

response = openai.chat.completions.create(

model="gpt-4.1-mini",  # or "gpt-5-nano", "gpt-5-mini"

messages=[{"role":  "user",  "content":  "Your prompt here"}]

)

```

  

#### Anthropic Claude Integration

```python

from anthropic import Anthropic

  

claude = Anthropic()

response = claude.messages.create(

model="claude-sonnet-4-5",

messages=[{"role":  "user",  "content":  "Your prompt here"}],

max_tokens=1000  # Required parameter

)

answer = response.content[0].text

```

  

#### Google Gemini Integration

```python

gemini = OpenAI(

api_key=os.getenv('GOOGLE_API_KEY'),

base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)

response = gemini.chat.completions.create(

model="gemini-2.5-flash",

messages=[{"role":  "user",  "content":  "Your prompt here"}]

)

```

  

#### DeepSeek Integration

```python

deepseek = OpenAI(

api_key=os.getenv('DEEPSEEK_API_KEY'),

base_url="https://api.deepseek.com/v1"

)

response = deepseek.chat.completions.create(

model="deepseek-chat",

messages=[{"role":  "user",  "content":  "Your prompt here"}]

)

```

  

#### Groq Integration

```python

groq = OpenAI(

api_key=os.getenv('GROQ_API_KEY'),

base_url="https://api.groq.com/openai/v1"

)

response = groq.chat.completions.create(

model="openai/gpt-oss-120b",

messages=[{"role":  "user",  "content":  "Your prompt here"}]

)

```

  

#### Local Ollama Integration

```python

# Requires Ollama running locally on http://localhost:11434

ollama = OpenAI(

base_url="http://localhost:11434/v1",

api_key="ollama"  # Required but not used

)

response = ollama.chat.completions.create(

model="llama2",  # or any locally installed model

messages=[{"role":  "user",  "content":  "Your prompt here"}]

)

```

  

### Multi-Model Comparison Framework

  

```python

def  compare_models(question,  models_config):

competitors =  []

answers =  []

messages =  [{"role":  "user",  "content": question}]

for model_name, client in models_config.items():

try:

if  "claude"  in model_name:

response = client.messages.create(

model=model_name,

messages=messages,

max_tokens=1000

)

answer = response.content[0].text

else:

response = client.chat.completions.create(

model=model_name,

messages=messages

)

answer = response.choices[0].message.content

competitors.append(model_name)

answers.append(answer)

except  Exception  as e:

print(f"Error with {model_name}: {e}")

return competitors, answers

```

  

---

  

## Production Considerations

  

### Environment Setup Best Practices

  

```python

# Essential imports for production systems

import os

import json

from dotenv import load_dotenv

from openai import OpenAI

from anthropic import Anthropic

from IPython.display import Markdown, display

  

# Always load environment variables

load_dotenv(override=True)

  

# Validate API keys

def  validate_api_keys():

keys =  {

'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),

'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),

'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),

'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),

'GROQ_API_KEY': os.getenv('GROQ_API_KEY')

}

for key_name, key_value in keys.items():

if key_value:

print(f"{key_name} exists and begins {key_value[:8]}")

else:

print(f"{key_name} not set")

```

  

### Error Handling and Robustness

  

```python

def  robust_llm_call(client,  model,  messages,  max_retries=3):

for attempt in  range(max_retries):

try:

response = client.chat.completions.create(

model=model,

messages=messages

)

return response.choices[0].message.content

except  Exception  as e:

if attempt == max_retries -  1:

raise e

print(f"Attempt {attempt +  1} failed: {e}. Retrying...")

time.sleep(2  ** attempt)  # Exponential backoff

```

  

### Cost Management

  

```python

def  estimate_token_cost(text,  model="gpt-4.1-mini"):

# Rough estimation: ~4 characters per token

estimated_tokens =  len(text)  /  4

# Model pricing (example rates)

pricing =  {

"gpt-4.1-mini":  {"input":  0.00015,  "output":  0.0006},  # per 1K tokens

"gpt-5-nano":  {"input":  0.0001,  "output":  0.0004},

"claude-sonnet-4-5":  {"input":  0.003,  "output":  0.015}

}

if model in pricing:

cost =  (estimated_tokens /  1000)  * pricing[model]["input"]

return cost, estimated_tokens

return  None, estimated_tokens

```

  

---

  

## Code Examples & Implementation

  

### Complete Agentic Workflow Example

  

```python

class  AgenticWorkflow:

def  __init__(self,  openai_client):

self.client = openai_client

self.conversation_history =  []

def  prompt_chaining_example(self):

"""Demonstrates the commercial application exercise from Lab 1"""

# Step 1: Business sector selection

sector_prompt =  "Please propose a business area that might be worth exploring for an Agentic AI opportunity. Respond only with the business area."

sector_response =  self.client.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": sector_prompt}]

)

business_sector = sector_response.choices[0].message.content

# Step 2: Pain point identification

pain_point_prompt =  f"Present a specific pain-point in the {business_sector} industry - something challenging that might be ripe for an Agentic AI solution. Respond only with the pain point description."

pain_response =  self.client.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": pain_point_prompt}]

)

pain_point = pain_response.choices[0].message.content

# Step 3: Solution proposal

solution_prompt =  f"Given this pain point in {business_sector}: '{pain_point}', propose a specific Agentic AI solution that could address this challenge."

solution_response =  self.client.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": solution_prompt}]

)

solution = solution_response.choices[0].message.content

return  {

"business_sector": business_sector,

"pain_point": pain_point,

"solution": solution

}

def  evaluator_optimizer_example(self,  initial_task):

"""Implements the evaluator-optimizer pattern"""

max_iterations =  3

current_solution =  None

for iteration in  range(max_iterations):

# Generator phase

if iteration ==  0:

generator_prompt = initial_task

else:

generator_prompt =  f"{initial_task}\n\nPrevious solution: {current_solution}\nFeedback: {feedback}\nPlease improve the solution based on this feedback."

generator_response =  self.client.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": generator_prompt}]

)

current_solution = generator_response.choices[0].message.content

# Evaluator phase

evaluator_prompt =  f"Evaluate this solution for the task '{initial_task}':\n\nSolution: {current_solution}\n\nProvide specific feedback on quality, completeness, and areas for improvement. If the solution is satisfactory, start your response with 'ACCEPTABLE:'."

evaluator_response =  self.client.chat.completions.create(

model="gpt-4.1-mini",

messages=[{"role":  "user",  "content": evaluator_prompt}]

)

feedback = evaluator_response.choices[0].message.content

if feedback.startswith("ACCEPTABLE:"):

break

return  {

"final_solution": current_solution,

"iterations": iteration +  1,

"final_feedback": feedback

}

  

# Usage example

def  main():

load_dotenv(override=True)

openai_client = OpenAI()

workflow = AgenticWorkflow(openai_client)

# Run prompt chaining example

business_analysis = workflow.prompt_chaining_example()

print("Business Analysis Results:")

for key, value in business_analysis.items():

print(f"{key.title()}: {value}\n")

# Run evaluator-optimizer example

task =  "Write a Python function that efficiently finds the longest common subsequence between two strings."

optimized_result = workflow.evaluator_optimizer_example(task)

print("Optimized Solution:")

print(f"Final Solution: {optimized_result['final_solution']}")

print(f"Iterations: {optimized_result['iterations']}")

  

if __name__ ==  "__main__":

main()

```

  

### Advanced Multi-Model Orchestration

  

```python

class  MultiModelOrchestrator:

def  __init__(self):

self.models =  self._initialize_models()

def  _initialize_models(self):

load_dotenv(override=True)

models =  {}

# OpenAI

if os.getenv('OPENAI_API_KEY'):

models['openai']  = OpenAI()

# Anthropic

if os.getenv('ANTHROPIC_API_KEY'):

models['anthropic']  = Anthropic()

# Gemini

if os.getenv('GOOGLE_API_KEY'):

models['gemini']  = OpenAI(

api_key=os.getenv('GOOGLE_API_KEY'),

base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)

return models

def  route_to_best_model(self,  task_type,  content):

"""Route tasks to most appropriate model based on task type"""

routing_map =  {

"creative_writing":  "openai",

"analysis":  "anthropic",

"coding":  "openai",

"reasoning":  "anthropic",

"general":  "gemini"

}

preferred_model = routing_map.get(task_type,  "openai")

if preferred_model in  self.models:

return  self._call_model(preferred_model, content)

else:

# Fallback to first available model

return  self._call_model(list(self.models.keys())[0], content)

def  _call_model(self,  model_name,  content):

client =  self.models[model_name]

messages =  [{"role":  "user",  "content": content}]

try:

if model_name ==  "anthropic":

response = client.messages.create(

model="claude-sonnet-4-5",

messages=messages,

max_tokens=1000

)

return response.content[0].text

else:

response = client.chat.completions.create(

model="gpt-4.1-mini"  if model_name ==  "openai"  else  "gemini-2.5-flash",

messages=messages

)

return response.choices[0].message.content

except  Exception  as e:

print(f"Error calling {model_name}: {e}")

return  None

```

  

---

  

## Key Takeaways for LLM Engineers

  

1.  **Architecture Matters**: Choose between workflow and agent patterns based on predictability requirements

2.  **Pattern Selection**: Match design patterns to specific use cases and complexity levels

3.  **Multi-Model Strategy**: Leverage different models' strengths through intelligent routing

4.  **Production Readiness**: Implement robust error handling, monitoring, and cost management

5.  **Iterative Improvement**: Use evaluator-optimizer patterns for quality assurance

6.  **Guardrails**: Essential for agent patterns to prevent unpredictable behavior

  

This guide provides the foundational knowledge and practical implementation patterns needed to build robust, production-ready agentic AI systems.