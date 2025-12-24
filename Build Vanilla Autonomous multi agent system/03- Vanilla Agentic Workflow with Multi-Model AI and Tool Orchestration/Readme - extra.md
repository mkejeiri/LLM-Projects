# Vanilla Agentic Workflow

 

## Table of Contents

1.  [Scanner Agent & Messaging Agent](#bm-1)

2.  [Autonomous Planning Agent](#bm-2)

3.  [Complete Framework & UI](#bm-3)

4.  [Technical Deep Dive](#technical-deep-dive)

5.  [Production Patterns](#production-patterns)


***Complete Agent Hierarchy***

```
AutonomousPlanningAgent (Orchestrator)
├── ScannerAgent (Finds deals from RSS)
├── EnsembleAgent (Estimates prices)
│   ├── Preprocessor (Rewrites text with Llama 3.2)
│   ├── SpecialistAgent (Fine-tuned model on Modal)
│   ├── FrontierAgent (GPT-5 with RAG on ChromaDB)
│   └── NeuralNetworkAgent (Deep neural network)
└── MessagingAgent (Sends push notifications with Claude)
```  

---

  

#  Scanner Agent & Messaging Agent {#bm-1}

  

## Overview

Build two critical agents:

-  **ScannerAgent**: Scrapes RSS feeds and uses structured outputs to extract promising deals

-  **MessagingAgent**: Sends push notifications using Pushover and Claude for message crafting

  

---

  

## Structured Outputs with Pydantic

  

### The Core Innovation

  

**Strunctured output:**

```python

from pydantic import BaseModel

  

class  Deal(BaseModel):

product_description:  str

price:  float

url:  str

  

# Tell OpenAI to respond with this structure

response = openai.chat.completions.parse(

model="gpt-5-mini",

messages=messages,

response_format=DealSelection

)

  

# Get back a populated Python object

results = response.choices[0].message.parsed

```

  

**What Actually Happens:**

  

1.  **Schema Generation**: Pydantic converts your class to JSON schema

2.  **Prompt Injection**: Schema added to system prompt automatically

3.  **Constrained Decoding**: During inference, OpenAI zeros out probability for any token that would violate the schema

4.  **Guaranteed Structure**: Only valid JSON tokens can be selected

  

### Why This Matters Commercially

  

**Before LLMs:**

- Parsing unstructured data was nearly impossible

- Traditional parsers were brittle and inflexible

- Couldn't handle variations like "$100 off $300" vs "$250 off this week only"

- Resume parsing, deal extraction required expensive custom solutions

  

**With LLMs + Structured Outputs:**

- Intelligent parsing with human-like understanding

- Handles complex variations and edge cases

- Guaranteed output format

- Trade-offs: cost, latency, some unpredictability

  

### Implementation Pattern

  

```python

from pydantic import BaseModel, Field

from openai import OpenAI

  

class  Deal(BaseModel):

product_description:  str  = Field(

description="Clearly expressed summary of the product in 3-4 sentences"

)

price:  float  = Field(

description="The actual price. If $100 off $300, respond with $200"

)

url:  str  = Field(

description="The URL of the deal as provided in input"

)

  

class  DealSelection(BaseModel):

deals: list[Deal]  = Field(

description="Your selection of the 5 most detailed deals with clear prices"

)

  

# View the generated schema

schema = DealSelection.model_json_schema()

print(schema)

```

  

---

  

## Scanner Agent Implementation

  

### RSS Feed Scraping

  

```python

from agents.deals import ScrapedDeal, DealSelection

import feedparser

  

# Define RSS feeds to monitor

RSS_FEEDS =  [

"https://www.dealnews.com/rss/c142/",  # Electronics

"https://www.dealnews.com/rss/c39/",  # Computers

"https://www.dealnews.com/rss/f1912/"  # Smart Home

]

  

# Fetch deals from all feeds

deals = ScrapedDeal.fetch(show_progress=True)

# Returns ~30 deals (10 from each feed)

  

print(len(deals))  # 30

print(deals[10].describe())

```

  

### Prompt Engineering for Deal Selection

  

**Critical Insight**: The prompt must handle edge cases like "$XXX off" which isn't the actual price.

  

```python

SYSTEM_PROMPT =  """You identify and summarize the 5 most detailed deals from a list,

by selecting deals that have the most detailed, high quality description and the most clear price.

Respond strictly in JSON with no explanation, using this format.

You should provide the price as a number derived from the description.

If the price of a deal isn't clear, do not include that deal in your response.

Most important is that you respond with the 5 deals that have the most detailed product description with price.

It's not important to mention the terms of the deal; most important is a thorough description of the product.

Be careful with products that are described as "$XXX off" or "reduced by $XXX" -

this isn't the actual price of the product. Only respond with products when you are highly confident about the price.

"""

  

USER_PROMPT_PREFIX =  """Respond with the most promising 5 deals from this list,

selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.

You should rephrase the description to be a summary of the product itself, not the terms of the deal.

Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.

Be careful with products that are described as "$XXX off" or "reduced by $XXX" -

this isn't the actual price of the product. Only respond with products when you are highly confident about the price.

  

Deals:

  

"""

  

USER_PROMPT_SUFFIX =  "\n\nInclude exactly 5 deals, no more."

```

  

### Creating the User Prompt

  

```python

def  make_user_prompt(scraped):

"""Construct user prompt from scraped deals"""

user_prompt = USER_PROMPT_PREFIX

user_prompt +=  '\n\n'.join([scrape.describe()  for scrape in scraped])

user_prompt += USER_PROMPT_SUFFIX

return user_prompt

  

# Create messages

user_prompt = make_user_prompt(deals)

messages =  [

{"role":  "system",  "content": SYSTEM_PROMPT},

{"role":  "user",  "content": user_prompt}

]

  

# View first 2000 chars

print(user_prompt[:2000])

```

  

### Structured Output Call

  

```python

# Call OpenAI with structured outputs

response = openai.chat.completions.parse(

model="gpt-5-mini",

messages=messages,

response_format=DealSelection,

reasoning_effort="minimal"

)

  

# Access parsed results

results = response.choices[0].message.parsed

  

# Iterate through deals

for deal in results.deals:

print(deal.product_description)

print(deal.price)

print(deal.url)

print()

```

  

### Complete Scanner Agent Class

  

```python

import logging

from openai import OpenAI

from agents.deals import ScrapedDeal, DealSelection

  

class  ScannerAgent:

COLOR =  "\033[36m"  # Cyan

RESET =  "\033[0m"

def  __init__(self):

self.log("Scanner Agent is initializing")

self.openai = OpenAI()

self.model =  "gpt-5-mini"

self.log("Scanner Agent is ready")

def  log(self,  message:  str):

"""Log with color coding"""

formatted =  f"\033[40m{self.COLOR}[Scanner Agent] {message}{self.RESET}"

logging.info(formatted)

def  make_user_prompt(self,  scraped):

"""Construct user prompt from scraped deals"""

user_prompt = USER_PROMPT_PREFIX

user_prompt +=  '\n\n'.join([scrape.describe()  for scrape in scraped])

user_prompt += USER_PROMPT_SUFFIX

return user_prompt

def  scan(self)  -> DealSelection:

"""Fetch deals from RSS feed and select top 5 using structured outputs"""

self.log("Scanner Agent is about to fetch deals from RSS feed")

deals = ScrapedDeal.fetch()

self.log(f"Scanner Agent received {len(deals)} deals not already scraped")

self.log("Scanner Agent is calling OpenAI using Structured Outputs")

user_prompt =  self.make_user_prompt(deals)

messages =  [

{"role":  "system",  "content": SYSTEM_PROMPT},

{"role":  "user",  "content": user_prompt}

]

response =  self.openai.chat.completions.parse(

model=self.model,

messages=messages,

response_format=DealSelection,

reasoning_effort="minimal"

)

results = response.choices[0].message.parsed

self.log(f"Scanner Agent received {len(results.deals)} selected deals with price>0 from OpenAI")

return results

def  test_scan(self)  -> DealSelection:

"""Return hardcoded test data for testing"""

# Returns predefined DealSelection for testing

pass

  

# Usage

agent = ScannerAgent()

result = agent.scan()

```

  

---

  

## Messaging Agent with Pushover

  

### Why Pushover?

  

**Comparison with Alternatives:**

-  **Twilio**: Requires paperwork for anti-spam compliance, complex setup

-  **Email**: Gets lost in inbox, no immediate notification

-  **Pushover**: Simple, instant, perfect for self-messaging

  

**Setup Steps:**

  

1. Visit https://pushover.net/

2. Click "Login or Signup" (top right)

3. Sign up for free account

4. Get USER token from dashboard (starts with 'u')

5. Click "Create an Application/API Token"

6. Name it (e.g., "AIEngineer")

7. Get TOKEN (starts with 'a')

8. Add to `.env`:

```

PUSHOVER_USER=u...

PUSHOVER_TOKEN=a...

```

9. Install Pushover app on phone

10. Test notification

  

### Basic Push Function

  

```python

import requests

import os

from dotenv import load_dotenv

  

load_dotenv(override=True)

  

pushover_user = os.getenv('PUSHOVER_USER')

pushover_token = os.getenv('PUSHOVER_TOKEN')

pushover_url =  "https://api.pushover.net/1/messages.json"

  

# Verify credentials loaded

if pushover_user:

print(f"Pushover user found and starts with {pushover_user[0]}")

else:

print("Pushover user not found")

  

if pushover_token:

print(f"Pushover token found and starts with {pushover_token[0]}")

else:

print("Pushover token not found")

  

def  push(message):

"""Send push notification"""

print(f"Push: {message}")

payload =  {

"user": pushover_user,

"token": pushover_token,

"message": message

}

requests.post(pushover_url,  data=payload)

  

# Test it

push("MASSIVE DEAL!!")

```

  

### Messaging Agent with LLM-Generated Hype

  

```python

from openai import OpenAI

import requests

import os

  

class  MessagingAgent:

COLOR =  "\033[37m"  # White

RESET =  "\033[0m"

def  __init__(self):

self.log("Messaging Agent is initializing")

self.openai = OpenAI()

self.pushover_user = os.getenv('PUSHOVER_USER')

self.pushover_token = os.getenv('PUSHOVER_TOKEN')

self.pushover_url =  "https://api.pushover.net/1/messages.json"

self.log("Messaging Agent has initialized Pushover and Chatgpt")

def  log(self,  message:  str):

"""Log with color coding"""

formatted =  f"\033[40m{self.COLOR}[Messaging Agent] {message}{self.RESET}"

logging.info(formatted)

def  push(self,  message:  str):

"""Send push notification"""

self.log("Messaging Agent is sending a push notification")

payload =  {

"user":  self.pushover_user,

"token":  self.pushover_token,

"message": message,

"sound":  "cashregister"  # Ka-ching!

}

requests.post(self.pushover_url,  data=payload)

def  craft_message(self,  description:  str,  price:  float,

estimate:  float,  url:  str)  ->  str:

"""Use GPT-4 to craft exciting message"""

self.log("Messaging Agent is using chatgpt to craft the message")

prompt =  f"""Write an exciting 2-sentence push notification about this deal:

Product: {description}

Deal Price: ${price}

Estimated Value: ${estimate}

URL: {url}

  

Make it exciting and urgent! Focus on the value and savings."""

response =  self.openai.chat.completions.create(

model="gpt-4",

messages=[{"role":  "user",  "content": prompt}],

max_tokens=100

)

return response.choices[0].message.content

def  notify(self,  description:  str,  deal_price:  float,

estimated_true_value:  float,  url:  str):

"""Craft hype message and send notification"""

message =  self.craft_message(description, deal_price,

estimated_true_value, url)

self.push(message)

self.log("Messaging Agent has completed")

  

# Usage

agent = MessagingAgent()

agent.push("SUCH A MASSIVE DEAL!!")

agent.notify(

"A special deal on Samsung 60 inch LED TV going at a great bargain",

300,

1000,

"www.samsung.com"

)

```

  

---

  

## Key Takeaways

  

### Structured Outputs

- Pydantic models define output structure

- Constrained decoding guarantees format

- Perfect for intelligent parsing of unstructured data

- Commercial applications: resume parsing, deal extraction, document processing

  

### Scanner Agent

- RSS feed monitoring

- Structured outputs for reliable extraction

- Prompt engineering to handle edge cases

- Color-coded logging for debugging

  

### Messaging Agent

- Push notifications via Pushover

- LLM-generated hype messages

- Simple integration with external APIs

- Real-time user engagement

  

### Production Considerations

- Environment variable management

- Error handling for API calls

- Logging with visual distinction

- Testing with mock data

  
  
  

---

  

#  Autonomous Planning Agent {#bm-2}

  

## Overview

Build the orchestrator that coordinates all agents using tool calling and autonomous planning loops.

  

---

  

## Understanding Agentic AI

  

### Definitions Evolution

  

**Sam Altman (OpenAI, 2024):**

> "AI systems that can work independently to achieve goals"

  

**2025 Industry Consensus:**

> "Systems where the LLM controls the workflow"

  

**Emerging Definition:**

> "LLM in a loop with tools to autonomously achieve a goal"

  

### Hallmarks of Agentic Solutions

  

1.  **Problem Decomposition**: Breaking bigger problems into smaller LLM-powered steps

2.  **Tool Use**: Function calling to interact with external systems

3.  **Structured Outputs**: For reliable orchestration

4.  **Agent Environment**: Communication between specialized agents

5.  **Planning Agent**: Coordinates the workflow

6.  **Autonomy**: Operates beyond single chat sessions

7.  **Memory**: Persists state across runs

  

---

  

## Tool Calling Fundamentals

  

### The Core Pattern

  

**How Tool Calling Works:**

  

1. Define Python functions with docstrings

2. Create JSON descriptions of functions

3. Pass tools array to LLM

4. LLM responds with `finish_reason="tool_calls"`

5. Execute the functions

6. Return results to LLM

7. Loop until LLM responds without tool calls

  

### Test Data Setup

  

```python

from agents.scanner_agent import ScannerAgent

  

# Get test data

test_results = ScannerAgent().test_scan()

print(test_results)

# Returns DealSelection with 4 hardcoded deals

```

  

### Fake Functions for Testing

  

```python

def  scan_the_internet_for_bargains()  ->  str:

"""This tool scans the internet for great deals and gets a curated list of promising deals"""

print("Fake function to scan the internet - this returns a hardcoded set of deals")

return test_results.model_dump_json()

  

def  estimate_true_value(description:  str)  ->  str:

"""

This tool estimates the true value of a product based on a text description of it

"""

print(f"Fake function to estimating true value of {description[:20]}... - this always returns $300")

return  f"Product {description} has an estimated true value of $300"

  

def  notify_user_of_deal(description:  str,  deal_price:  float,

estimated_true_value:  float,  url:  str)  ->  str:

"""

This tool notifies the user of a great deal, given a description of it,

the price of the deal, and the estimated true value

"""

print(f"Fake function to notify user of {description} which costs {deal_price} and estimate is {estimated_true_value}")

return  "notification sent ok"

  

# Test them

notify_user_of_deal("a new iphone",  100,  1000,  "https://www.apple.com/iphone")

```

  

---

  

## Tool JSON Definitions

  

### The Big Block of JSON

  

**Critical Understanding**: This JSON describes the function signature to the LLM.

  

```python

scan_function =  {

"name":  "scan_the_internet_for_bargains",

"description":  "Returns top bargains scraped from the internet along with the price each item is being offered for",

"parameters":  {

"type":  "object",

"properties":  {},

"required":  [],

"additionalProperties":  False

}

}

  

estimate_function =  {

"name":  "estimate_true_value",

"description":  "Given the description of an item, estimate how much it is actually worth",

"parameters":  {

"type":  "object",

"properties":  {

"description":  {

"type":  "string",

"description":  "The description of the item to be estimated"

},

},

"required":  ["description"],

"additionalProperties":  False

}

}

  

notify_function =  {

"name":  "notify_user_of_deal",

"description":  "Send the user a push notification about the single most compelling deal; only call this one time",

"parameters":  {

"type":  "object",

"properties":  {

"description":  {

"type":  "string",

"description":  "The description of the item itself scraped from the internet"

},

"deal_price":  {

"type":  "number",

"description":  "The price offered by this deal scraped from the internet"

},

"estimated_true_value":  {

"type":  "number",

"description":  "The estimated actual value that this is worth"

},

"url":  {

"type":  "string",

"description":  "The URL of this deal as scraped from the internet"

}

},

"required":  ["description",  "deal_price",  "estimated_true_value",  "url"],

"additionalProperties":  False

}

}

  

# Combine into tools array

tools =  [

{"type":  "function",  "function": scan_function},

{"type":  "function",  "function": estimate_function},

{"type":  "function",  "function": notify_function}

]

```

  

---

  

## Tool Call Handler

  

### Executing Tool Calls

  

```python

import json

  

def  handle_tool_call(message):

"""

Actually call the tools associated with this message

"""

results =  []

for tool_call in message.tool_calls:

tool_name = tool_call.function.name

arguments = json.loads(tool_call.function.arguments)

# Dynamic function lookup (use mapping in production!)

tool =  globals().get(tool_name)

result = tool(**arguments)  if tool else  {}

results.append({

"role":  "tool",

"content": json.dumps(result),

"tool_call_id": tool_call.id

})

return results

```

  

**Production Note**: Never use `globals()` in production. Use a mapping dictionary:

  

```python

def  handle_tool_call(message):

mapping =  {

"scan_the_internet_for_bargains": scan_the_internet_for_bargains,

"estimate_true_value": estimate_true_value,

"notify_user_of_deal": notify_user_of_deal

}

results =  []

for tool_call in message.tool_calls:

tool_name = tool_call.function.name

arguments = json.loads(tool_call.function.arguments)

tool = mapping.get(tool_name)

result = tool(**arguments)  if tool else  ""

results.append({

"role":  "tool",

"content": json.dumps(result),

"tool_call_id": tool_call.id

})

return results

```

  

---

  

## The Agent Loop - The Critical Pattern

  

### Setting Up Messages

  

```python

from openai import OpenAI

  

openai = OpenAI()

MODEL =  "gpt-5.1"

  

system_message =  "You find great deals on bargain products using your tools, and notify the user of the best bargain."

  

user_message =  """

First, use your tool to scan the internet for bargain deals.

Then for each deal, use your tool to estimate its true value.

Then pick the single most compelling deal where the price is much lower than the estimated true value,

and use your tool to notify the user.

Then just reply OK to indicate success.

"""

  

messages =  [

{"role":  "system",  "content": system_message},

{"role":  "user",  "content": user_message}

]

```

  

### The Agent Loop

  

**This is what makes it agentic!**

  

```python

done =  False

while  not done:

response = openai.chat.completions.create(

model=MODEL,

messages=messages,

tools=tools

)

if response.choices[0].finish_reason ==  "tool_calls":

message = response.choices[0].message

results = handle_tool_call(message)

messages.append(message)

messages.extend(results)

else:

done =  True

  

final_response = response.choices[0].message.content

print(final_response)  # "OK"

```

  

### What Happens in the Loop

  

**Iteration 1:**

- LLM calls `scan_the_internet_for_bargains()`

- Returns 4 deals as JSON

- Appended to messages

  

**Iteration 2:**

- LLM calls `estimate_true_value()` 4 times (once per deal)

- Each returns "$300"

- All appended to messages

  

**Iteration 3:**

- LLM calls `notify_user_of_deal()` with best deal

- Returns "notification sent ok"

- Appended to messages

  

**Iteration 4:**

- LLM responds with "OK"

-  `finish_reason != "tool_calls"`

- Loop exits

  

---

  

## Autonomous Planning Agent Implementation

  

### Complete Agent Class

  

```python

import json

import logging

from openai import OpenAI

from agents.scanner_agent import ScannerAgent

from agents.ensemble_agent import EnsembleAgent

from agents.messaging_agent import MessagingAgent

  

class  AutonomousPlanningAgent:

COLOR =  "\033[32m"  # Green

RESET =  "\033[0m"

MODEL =  "gpt-5.1"

def  __init__(self,  collection):

self.log("Autonomous Planning Agent is initializing")

self.openai = OpenAI()

self.scanner = ScannerAgent()

self.ensemble = EnsembleAgent(collection)

self.messenger = MessagingAgent()

self.tools =  self._build_tools()

self.log("Autonomous Planning Agent is ready")

def  log(self,  message:  str):

"""Log with color coding"""

formatted =  f"\033[40m{self.COLOR}[Autonomous Planning Agent] {message}{self.RESET}"

logging.info(formatted)

def  scan_the_internet_for_bargains(self)  ->  str:

"""Tool: Scan internet for deals"""

self.log("Autonomous Planning agent is calling scanner")

results =  self.scanner.scan()

return results.model_dump_json()

def  estimate_true_value(self,  description:  str)  ->  str:

"""Tool: Estimate true value via Ensemble Agent"""

self.log("Autonomous Planning agent is estimating value via Ensemble Agent")

estimate =  self.ensemble.price(description)

return  f"The estimated true value of {description} is {estimate}"

def  notify_user_of_deal(self,  description:  str,  deal_price:  float,

estimated_true_value:  float,  url:  str)  ->  str:

"""Tool: Notify user of deal"""

self.log("Autonomous Planning agent is notifying user")

self.messenger.notify(description, deal_price, estimated_true_value, url)

return  "notification sent"

def  _build_tools(self):

"""Build tools array for OpenAI"""

scan_function =  {

"name":  "scan_the_internet_for_bargains",

"description":  "Returns top bargains scraped from the internet",

"parameters":  {

"type":  "object",

"properties":  {},

"required":  [],

"additionalProperties":  False

}

}

estimate_function =  {

"name":  "estimate_true_value",

"description":  "Given description of item, estimate its worth",

"parameters":  {

"type":  "object",

"properties":  {

"description":  {

"type":  "string",

"description":  "The description of the item to be estimated"

}

},

"required":  ["description"],

"additionalProperties":  False

}

}

notify_function =  {

"name":  "notify_user_of_deal",

"description":  "Send user a push notification about the most compelling deal",

"parameters":  {

"type":  "object",

"properties":  {

"description":  {"type":  "string",  "description":  "Item description"},

"deal_price":  {"type":  "number",  "description":  "Deal price"},

"estimated_true_value":  {"type":  "number",  "description":  "Estimated value"},

"url":  {"type":  "string",  "description":  "Deal URL"}

},

"required":  ["description",  "deal_price",  "estimated_true_value",  "url"],

"additionalProperties":  False

}

}

return  [

{"type":  "function",  "function": scan_function},

{"type":  "function",  "function": estimate_function},

{"type":  "function",  "function": notify_function}

]

def  handle_tool_call(self,  message):

"""Execute tool calls"""

mapping =  {

"scan_the_internet_for_bargains":  self.scan_the_internet_for_bargains,

"estimate_true_value":  self.estimate_true_value,

"notify_user_of_deal":  self.notify_user_of_deal

}

results =  []

for tool_call in message.tool_calls:

tool_name = tool_call.function.name

arguments = json.loads(tool_call.function.arguments)

tool = mapping.get(tool_name)

result = tool(**arguments)  if tool else  ""

results.append({

"role":  "tool",

"content": result,

"tool_call_id": tool_call.id

})

return results

def  plan(self,  memory=None)  ->  str:

"""Execute autonomous planning loop"""

self.log("Autonomous Planning Agent is kicking off a run")

system_message =  "You find great deals on bargain products using your tools, and notify the user of the best bargain."

user_message =  """

First, use your tool to scan the internet for bargain deals.

Then for each deal, use your tool to estimate its true value.

Then pick the single most compelling deal where the price is much lower than the estimated true value,

and use your tool to notify the user.

Then just reply OK to indicate success.

"""

messages =  [

{"role":  "system",  "content": system_message},

{"role":  "user",  "content": user_message}

]

# THE AGENT LOOP

done =  False

while  not done:

response =  self.openai.chat.completions.create(

model=self.MODEL,

messages=messages,

tools=self.tools

)

if response.choices[0].finish_reason ==  "tool_calls":

message = response.choices[0].message

results =  self.handle_tool_call(message)

messages.append(message)

messages.extend(results)

else:

done =  True

reply = response.choices[0].message.content

self.log(f"Autonomous Planning Agent completed with reply: {reply}")

return reply

  

# Usage

import chromadb

  

DB =  "products_vectorstore"

client = chromadb.PersistentClient(path=DB)

collection = client.get_or_create_collection('products')

  

agent = AutonomousPlanningAgent(collection)

agent.plan()

```

  

---

  

## Key Takeaways

  

### Agentic AI Definition

- LLM in a loop with tools

- Autonomous goal achievement

- Workflow controlled by LLM

  

### Tool Calling Pattern

```python

done =  False

while  not done:

response = llm_call_with_tools()

if needs_tool_call:

execute_tools()

continue

else:

done =  True

```

  

### Critical Components

1.  **Tool Definitions**: JSON schemas describing functions

2.  **Tool Handler**: Executes functions and formats results

3.  **Agent Loop**: Continues until task completion

4.  **Message History**: Accumulates context across iterations

  

### Production Patterns

- Use mapping dictionaries, not `globals()`

- Add comprehensive error handling

- Implement retry logic

- Log every step with colors

- Monitor costs and latency

  

### Multi-Agent Coordination

- Planning agent orchestrates specialists

- Each agent has specific responsibility

- Tools abstract implementation details

- Structured outputs ensure reliability

  
  
  

---

  

#  Complete Framework & UI {#bm-3}

  

## Overview

Build the complete system with Gradio UI, memory persistence, and autonomous operation.

  

---

  

##  Gradio UI Basics

  

### Simple UI

  

```python

import gradio as gr

  

with gr.Blocks(title="The Price is Right",  fill_width=True)  as ui:

with gr.Row():

gr.Markdown('<div style="text-align: center;font-size:24px">The Price is Right - Deal Hunting Agentic AI</div>')

with gr.Row():

gr.Markdown('<div style="text-align: center;font-size:14px">Autonomous agent framework that finds online deals</div>')

  

ui.launch(inbrowser=True)

```

  

### Adding Data Table

  

```python

from agents.deals import Opportunity, Deal

  

with gr.Blocks(title="The Price is Right",  fill_width=True)  as ui:

# Create initial test data

initial_deal = Deal(

product_description="Example description",

price=100.0,

url="https://cnn.com"

)

initial_opportunity = Opportunity(

deal=initial_deal,

estimate=200.0,

discount=100.0

)

opportunities = gr.State([initial_opportunity])

def  get_table(opps):

return  [

[

opp.deal.product_description,

opp.deal.price,

opp.estimate,

opp.discount,

opp.deal.url

]

for opp in opps

]

with gr.Row():

gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')

with gr.Row():

gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')

with gr.Row():

opportunities_dataframe = gr.Dataframe(

headers=["Description",  "Price",  "Estimate",  "Discount",  "URL"],

wrap=True,

column_widths=[4,  1,  1,  1,  2],

row_count=10,

col_count=5,

max_height=400,

)

# Load data on startup

ui.load(get_table,  inputs=[opportunities],  outputs=[opportunities_dataframe])

  

ui.launch(inbrowser=True)

```

  

### Adding Click Handler

  

```python

from deal_agent_framework import DealAgentFramework

  

agent_framework = DealAgentFramework()

agent_framework.init_agents_as_needed()

  

with gr.Blocks(title="The Price is Right",  fill_width=True)  as ui:

initial_deal = Deal(product_description="Example description",  price=100.0,  url="https://cnn.com")

initial_opportunity = Opportunity(deal=initial_deal,  estimate=200.0,  discount=100.0)

opportunities = gr.State([initial_opportunity])

def  get_table(opps):

return  [[opp.deal.product_description, opp.deal.price, opp.estimate, opp.discount, opp.deal.url]  for opp in opps]

def  do_select(opportunities,  selected_index: gr.SelectData):

"""Handle row click - send notification for selected deal"""

row = selected_index.index[0]

opportunity = opportunities[row]

agent_framework.planner.messenger.alert(opportunity)

with gr.Row():

gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')

with gr.Row():

gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')

with gr.Row():

opportunities_dataframe = gr.Dataframe(

headers=["Description",  "Price",  "Estimate",  "Discount",  "URL"],

wrap=True,

column_widths=[4,  1,  1,  1,  2],

row_count=10,

col_count=5,

max_height=400,

)

ui.load(get_table,  inputs=[opportunities],  outputs=[opportunities_dataframe])

# Add click handler

opportunities_dataframe.select(do_select,  inputs=[opportunities],  outputs=[])

  

ui.launch(inbrowser=True)

```

  

---

  

## Deal Agent Framework

  

### Memory Management

  

```python

import json

from agents.deals import Opportunity, Deal

  

class  DealAgentFramework:

MEMORY_FILE =  "memory.json"

def  __init__(self):

self.memory =  self.read_memory()

self.planner =  None

self.collection =  None

@staticmethod

def  read_memory()  -> list[Opportunity]:

"""Load memory from JSON file"""

try:

with  open(DealAgentFramework.MEMORY_FILE,  "r")  as f:

data = json.load(f)

return  [Opportunity(**item)  for item in data]

except  FileNotFoundError:

return  []

@staticmethod

def  write_memory(opportunities: list[Opportunity]):

"""Save memory to JSON file"""

with  open(DealAgentFramework.MEMORY_FILE,  "w")  as f:

data =  [opp.model_dump()  for opp in opportunities]

json.dump(data, f,  indent=2)

@staticmethod

def  reset_memory():

"""Reset to initial 2 deals"""

initial_deals =  [

Opportunity(

deal=Deal(

product_description="Beats Solo 4 Wireless Headphones...",

price=79.0,

url="https://www.dealnews.com/..."

),

estimate=150.0,

discount=71.0

),

Opportunity(

deal=Deal(

product_description="HumsiENK 12V 310Ah LiFePO4...",

price=262.0,

url="https://www.dealnews.com/..."

),

estimate=400.0,

discount=138.0

)

]

DealAgentFramework.write_memory(initial_deals)

def  init_agents_as_needed(self):

"""Initialize ChromaDB and Planning Agent"""

if  not  self.collection:

import chromadb

client = chromadb.PersistentClient(path="products_vectorstore")

self.collection = client.get_or_create_collection('products')

if  not  self.planner:

from agents.autonomous_planning_agent import AutonomousPlanningAgent

self.planner = AutonomousPlanningAgent(self.collection)

def  run(self)  -> list[Opportunity]:

"""Execute one cycle of deal hunting"""

self.init_agents_as_needed()

# Run autonomous planning agent

self.planner.plan(memory=self.memory)

# Extract new opportunities from planner results

# (Implementation depends on how planner stores results)

new_opportunities =  []  # Extract from planner

# Update memory

self.memory.extend(new_opportunities)

self.write_memory(self.memory)

return new_opportunities

@staticmethod

def  get_plot_data(max_datapoints=800):

"""Get vector DB data for 3D visualization"""

import chromadb

import numpy as np

from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="products_vectorstore")

collection = client.get_collection('products')

# Get all documents

results = collection.get(limit=max_datapoints,  include=["documents",  "embeddings"])

documents = results["documents"]

embeddings = results["embeddings"]

# Reduce to 3D using PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

vectors = pca.fit_transform(embeddings)

# Color by cluster

colors =  ["blue"]  *  len(vectors)

return documents, vectors, colors

```

  

---

  

## Complete Production UI

  

### Logging with Queue

  

```python

import logging

import queue

import threading

import time

  

class  QueueHandler(logging.Handler):

def  __init__(self,  log_queue):

super().__init__()

self.log_queue = log_queue

def  emit(self,  record):

self.log_queue.put(self.format(record))

  

def  setup_logging(log_queue):

handler = QueueHandler(log_queue)

formatter = logging.Formatter(

"[%(asctime)s] %(message)s",

datefmt="%Y-%m-%d %H:%M:%S %z",

)

handler.setFormatter(formatter)

logger = logging.getLogger()

logger.addHandler(handler)

logger.setLevel(logging.INFO)

  

def  html_for(log_data):

"""Format log data as HTML"""

output =  "<br>".join(log_data[-18:])

return  f"""

<div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">

{output}

</div>

"""

```

  

### Complete App Class

  

```python

import gradio as gr

import plotly.graph_objects as go

from deal_agent_framework import DealAgentFramework

from log_utils import reformat

  

class  App:

def  __init__(self):

self.agent_framework =  None

def  get_agent_framework(self):

if  not  self.agent_framework:

self.agent_framework = DealAgentFramework()

return  self.agent_framework

def  run(self):

with gr.Blocks(title="The Price is Right",  fill_width=True)  as ui:

log_data = gr.State([])

def  table_for(opps):

return  [

[

opp.deal.product_description,

f"${opp.deal.price:.2f}",

f"${opp.estimate:.2f}",

f"${opp.discount:.2f}",

opp.deal.url,

]

for opp in opps

]

def  update_output(log_data,  log_queue,  result_queue):

"""Stream logs and results"""

initial_result = table_for(self.get_agent_framework().memory)

final_result =  None

while  True:

try:

message = log_queue.get_nowait()

log_data.append(reformat(message))

yield log_data, html_for(log_data), final_result or initial_result

except queue.Empty:

try:

final_result = result_queue.get_nowait()

yield log_data, html_for(log_data), final_result or initial_result

except queue.Empty:

if final_result is  not  None:

break

time.sleep(0.1)

def  get_plot():

"""Generate 3D scatter plot of vector DB"""

documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)

fig = go.Figure(

data=[

go.Scatter3d(

x=vectors[:,  0],

y=vectors[:,  1],

z=vectors[:,  2],

mode="markers",

marker=dict(size=2,  color=colors,  opacity=0.7),

)

]

)

fig.update_layout(

scene=dict(

xaxis_title="x",

yaxis_title="y",

zaxis_title="z",

aspectmode="manual",

aspectratio=dict(x=2.2,  y=2.2,  z=1),

camera=dict(eye=dict(x=1.6,  y=1.6,  z=0.8))

),

height=400,

margin=dict(r=5,  b=1,  l=5,  t=2),

)

return fig

def  do_run():

"""Execute agent framework"""

new_opportunities =  self.get_agent_framework().run()

table = table_for(new_opportunities)

return table

def  run_with_logging(initial_log_data):

"""Run with threaded logging"""

log_queue = queue.Queue()

result_queue = queue.Queue()

setup_logging(log_queue)

def  worker():

result = do_run()

result_queue.put(result)

thread = threading.Thread(target=worker)

thread.start()

for log_data, output, final_result in update_output(

initial_log_data, log_queue, result_queue

):

yield log_data, output, final_result

def  do_select(selected_index: gr.SelectData):

"""Handle row click"""

opportunities =  self.get_agent_framework().memory

row = selected_index.index[0]

opportunity = opportunities[row]

self.get_agent_framework().planner.messenger.alert(opportunity)

# UI Layout

with gr.Row():

gr.Markdown(

'<div style="text-align: center;font-size:24px">'

'<strong>The Price is Right</strong> - '

'Autonomous Agent Framework that hunts for deals</div>'

)

with gr.Row():

gr.Markdown(

'<div style="text-align: center;font-size:14px">'

'A proprietary fine-tuned LLM deployed on Modal and '

'a RAG pipeline with a frontier model collaborate to '

'send push notifications with great online deals.</div>'

)

with gr.Row():

opportunities_dataframe = gr.Dataframe(

headers=["Deals found so far",  "Price",  "Estimate",  "Discount",  "URL"],

wrap=True,

column_widths=[6,  1,  1,  1,  3],

row_count=10,

col_count=5,

max_height=400,

)

with gr.Row():

with gr.Column(scale=1):

logs = gr.HTML()

with gr.Column(scale=1):

plot = gr.Plot(value=get_plot(),  show_label=False)

# Auto-run on load

ui.load(

run_with_logging,

inputs=[log_data],

outputs=[log_data, logs, opportunities_dataframe],

)

# Auto-run every 5 minutes (300 seconds)

timer = gr.Timer(value=300,  active=True)

timer.tick(

run_with_logging,

inputs=[log_data],

outputs=[log_data, logs, opportunities_dataframe],

)

# Click handler

opportunities_dataframe.select(do_select)

ui.launch(share=False,  inbrowser=True)

  

if __name__ ==  "__main__":

App().run()

```

  

---

  

## Running the Complete System

  

### Reset Memory

  

```python

from deal_agent_framework import DealAgentFramework

  

# Reset to initial 2 deals

DealAgentFramework.reset_memory()

```

  

### Enable Logging

  

```python

import logging

  

root = logging.getLogger()

root.setLevel(logging.INFO)

```

  

### Launch UI

  

```bash

# From command line

uv  run  assess_notify_best_deal.py

  

# Or from notebook

!uv  run  assess_notify_best_deal.py

```

  

### System Behavior

  

**On Startup:**

1. Loads memory from `memory.json`

2. Initializes ChromaDB vector store

3. Creates all agents (Scanner, Ensemble, Messenger, Planner)

4. Displays existing deals in table

5. Shows 3D visualization of vector DB

6. Automatically runs first scan

  

**Every 5 Minutes:**

1. Scanner fetches new deals from RSS feeds

2. Ensemble estimates prices using 3 models

3. Planner identifies best deal

4. Messenger sends push notification

5. UI updates with new deals

6. Memory persisted to JSON

  

**On Click:**

- User clicks any deal row

- Messenger sends push notification for that deal

- Notification includes LLM-generated hype message

  

---

  

## Key Takeaways

  

### Gradio UI

- Blocks for complex layouts

- State for persistent data

- Timers for periodic execution

- Event handlers for interactivity

- HTML components for custom styling

  

### Memory Persistence

- JSON file for simple state storage

- Pydantic models for serialization

- Read/write on every run

- Reset capability for testing

  

### Production Architecture

- Threaded execution for non-blocking UI

- Queue-based logging

- Real-time log streaming

- Error handling and recovery

  

### Complete System Integration

- 7 specialized agents

- 6 different models

- 34 model calls per cycle

- Autonomous operation

- Real-time notifications

  

---

  

# Technical Deep Dive {#technical-deep-dive}

  

## Multi-Agent Architecture

  

### Complete Agent Hierarchy

  

```

DealAgentFramework

└── AutonomousPlanningAgent (GPT-5.1)

├── ScannerAgent (GPT-5-mini)

├── EnsembleAgent

│ ├── Preprocessor (Llama 3.2 via Ollama)

│ ├── SpecialistAgent (Fine-tuned Llama 3.2 3B on Modal)

│ ├── FrontierAgent (GPT-5.1 + RAG on ChromaDB)

│ │ └── all-MiniLM-L6-v2 (embeddings)

│ └── NeuralNetworkAgent (Deep neural network)

└── MessagingAgent (Claude Sonnet 4-5)

```

  

### Model Call Statistics

  

**Per Cycle: 34 Model Calls**

- 29 LLM calls

- 5 Neural network calls

  

**Models Used:**

  

**Frontier Models (3):**

1. GPT-5-mini: Deal scanning and selection

2. GPT-5.1: Autonomous planning and RAG

3. Claude Sonnet 4-5: Message crafting

  

**Open Source Models (3):**

1. Fine-tuned Llama 3.2 3B: Specialist pricing (Modal)

2. all-MiniLM-L6-v2: Embeddings for RAG

3. Llama 3.2: Text preprocessing (Ollama)

  

### Data Flow

  

```

RSS Feeds

↓

ScannerAgent (GPT-5-mini + Structured Outputs)

↓

5 Selected Deals

↓

AutonomousPlanningAgent (GPT-5.1 + Tool Calling)

↓

For each deal:

↓

EnsembleAgent

├→ Preprocessor (Llama 3.2) → rewritten text

├→ SpecialistAgent (Fine-tuned) → price estimate

├→ FrontierAgent (GPT-5.1 + RAG) → price estimate

└→ NeuralNetworkAgent → price estimate

↓

Average of 3 estimates

↓

Best deal selected

↓

MessagingAgent (Claude Sonnet 4-5)

↓

Push Notification (Pushover)

```

  

---

  

# Production Patterns {#production-patterns}

  

## Error Handling

  

```python

def  safe_tool_call(tool,  arguments):

"""Execute tool with error handling"""

try:

result = tool(**arguments)

return  {"success":  True,  "result": result}

except  Exception  as e:

logging.error(f"Tool call failed: {e}")

return  {"success":  False,  "error":  str(e)}

  

def  handle_tool_call_with_retry(message,  max_retries=3):

"""Execute tool calls with retry logic"""

results =  []

for tool_call in message.tool_calls:

tool_name = tool_call.function.name

arguments = json.loads(tool_call.function.arguments)

tool = mapping.get(tool_name)

for attempt in  range(max_retries):

result = safe_tool_call(tool, arguments)

if result["success"]:

break

time.sleep(2  ** attempt)  # Exponential backoff

results.append({

"role":  "tool",

"content": json.dumps(result),

"tool_call_id": tool_call.id

})

return results

```

  

## Cost Monitoring

  

```python

class  CostTracker:

COSTS =  {

"gpt-5-mini":  {"input":  0.15,  "output":  0.60},  # per 1M tokens

"gpt-5.1":  {"input":  2.50,  "output":  10.00},

"claude-sonnet-4-5":  {"input":  3.00,  "output":  15.00}

}

def  __init__(self):

self.total_cost =  0.0

self.calls =  []

def  track(self,  model,  input_tokens,  output_tokens):

"""Track cost of a model call"""

costs =  self.COSTS.get(model,  {"input":  0,  "output":  0})

cost =  (

(input_tokens /  1_000_000)  * costs["input"]  +

(output_tokens /  1_000_000)  * costs["output"]

)

self.total_cost += cost

self.calls.append({

"model": model,

"input_tokens": input_tokens,

"output_tokens": output_tokens,

"cost": cost

})

return cost

def  report(self):

"""Generate cost report"""

print(f"Total Cost: ${self.total_cost:.4f}")

print(f"Total Calls: {len(self.calls)}")

for call in  self.calls:

print(f" {call['model']}: ${call['cost']:.4f}")

```

  

## Latency Optimization

  

```python

import asyncio

from concurrent.futures import ThreadPoolExecutor

  

async  def  parallel_estimates(ensemble,  descriptions):

"""Run estimates in parallel"""

with ThreadPoolExecutor(max_workers=5)  as executor:

loop = asyncio.get_event_loop()

tasks =  [

loop.run_in_executor(executor, ensemble.price, desc)

for desc in descriptions

]

results =  await asyncio.gather(*tasks)

return results

  

# Usage

descriptions =  [deal.product_description for deal in deals]

estimates = asyncio.run(parallel_estimates(ensemble, descriptions))

```

  

## Monitoring and Observability

  

```python

import time

from functools import wraps

  

def  monitor_agent(func):

"""Decorator to monitor agent performance"""

@wraps(func)

def  wrapper(*args,  **kwargs):

start = time.time()

try:

result = func(*args,  **kwargs)

duration = time.time()  - start

logging.info(f"{func.__name__} completed in {duration:.2f}s")

return result

except  Exception  as e:

duration = time.time()  - start

logging.error(f"{func.__name__} failed after {duration:.2f}s: {e}")

raise

return wrapper

  

class  ScannerAgent:

@monitor_agent

def  scan(self):

# Implementation

pass

```

  

---

  

## Key Takeaways for LLM passionate

  

### 1. Structured Outputs

- Pydantic models guarantee output format

- Constrained decoding ensures reliability

- Perfect for intelligent parsing

- Commercial applications: resumes, deals, documents

  

### 2. Tool Calling Pattern

```python

done =  False

while  not done:

response = llm_call_with_tools()

if needs_tool_call:

execute_tools()

continue

else:

done =  True

```

  

### 3. Multi-Agent Orchestration

- Planning agent coordinates specialists

- Each agent has specific responsibility

- Tools abstract implementation details

- Structured outputs ensure reliability

  

### 4. Memory and Persistence

- Simple JSON for state storage

- Enables autonomy beyond single sessions

- Foundation for continuous operation

  

### 5. Production Considerations

- Error handling and retry logic

- Cost tracking and optimization

- Latency optimization with parallelization

- Comprehensive logging and monitoring

- Use mapping dictionaries, not `globals()`

  

### 6. Commercial Applications

- Intelligent parsing (resumes, deals, documents)

- Autonomous monitoring and alerting

- Multi-model ensemble for accuracy

- RAG for domain-specific knowledge

- Fine-tuned models for specialized tasks

  

---

  

## Complete Project Structure

  

```

project/

├── agents/

│ ├── __init__.py

│ ├── scanner_agent.py # RSS scraping + structured outputs

│ ├── ensemble_agent.py # Coordinates pricing models

│ ├── specialist_agent.py # Fine-tuned model on Modal

│ ├── frontier_agent.py # GPT-5 with RAG

│ ├── neural_network_agent.py # Deep neural network

│ ├── messaging_agent.py # Pushover + Claude

│ ├── autonomous_planning_agent.py # Orchestrator with tools

│ ├── preprocessor.py # Text preprocessing

│ ├── evaluator.py # Model evaluation

│ └── deals.py # Pydantic models

├── deal_agent_framework.py # Main framework

├── assess_notify_best_deal.py # Gradio UI

├── log_utils.py # Logging utilities

├── memory.json # Persistent state

├── products_vectorstore/ # ChromaDB

├── .env # API keys

├── requirements.txt # Dependencies

└── README.md # Documentation

```

---

## Advanced Extensions

  

### Using Agent Frameworks

-  **OpenAI Agents SDK**: Official framework with built-in tools

-  **LangGraph**: Graph-based agent orchestration

-  **CrewAI**: Role-based multi-agent systems

-  **AutoGen**: Microsoft's multi-agent framework

  

### MCP (Model Context Protocol)

- Standardized tool interfaces

- Server-based tool execution

- Cross-platform compatibility

- Reusable tool definitions

  

### Production Deployment

-  **Containerization**: Docker for consistent environments

-  **Serverless**: AWS Lambda, Modal for scalability

-  **Monitoring**: Prometheus, Grafana for observability

-  **Cost Optimization**: Caching, batching, model selection

---

  

