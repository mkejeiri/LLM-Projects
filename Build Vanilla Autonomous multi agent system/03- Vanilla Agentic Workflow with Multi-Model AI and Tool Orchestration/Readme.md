# Vanilla Agentic Workflow with Multi-Model AI and Tool Orchestration

## Table of Contents
1. [Structured Outputs with Pydantic](#structured-outputs)
2. [Scanner Agent Implementation](#scanner-agent)
3. [Messaging Agent with Pushover](#messaging-agent)
4. [Autonomous Planning Agent](#planning-agent)
5. [Multi-Agent System Architecture](#multi-agent-system)
6. [Complete Framework Integration](#framework-integration)

---

## 1. Structured Outputs with Pydantic {#structured-outputs}

### Core Concept: Constrained Decoding

**What It Feels Like:**
- Define a Python class (subclass of `BaseModel`)
- Tell OpenAI to respond with a populated Python object instead of natural language
- Get back a structured object with fields filled in

**What Actually Happens:**
1. Pydantic class generates a JSON schema
2. Schema is added to system prompt
3. Model generates JSON tokens (models are trained on tons of JSON)
4. OpenAI client library converts JSON to Python object instance

**The Magic: Inference-Time Constrained Decoding**
- Model generates probability distribution for next tokens
- OpenAI code zeros out probabilities for tokens that would break the JSON spec
- Only valid tokens can be selected
- Guarantees output conforms to schema (except edge cases like infinite loops)

### Implementation Pattern

```python
from pydantic import BaseModel
from openai import OpenAI

# Define Pydantic models
class Deal(BaseModel):
    product_description: str  # "Clearly expressed summary of the product..."
    price: float  # "The actual price, e.g., $100 off $300 = $200"
    url: str  # "The URL as provided"

class DealSelection(BaseModel):
    deals: list[Deal]  # "Your selection of the 5 most detailed deals"

# Use structured outputs
response = openai.chat.completions.parse(
    model="gpt-5-mini",
    messages=messages,
    response_format=DealSelection,
    reasoning_effort="minimal"
)

# Access parsed result
results = response.choices[0].message.parsed
for deal in results.deals:
    print(deal.product_description, deal.price, deal.url)
```

### JSON Schema Generation

```python
# View the schema that gets sent to OpenAI
schema = DealSelection.model_json_schema()
# Returns detailed JSON describing required structure
```

### Commercial Impact

**Problem Solved:** Parsing unstructured data was nearly impossible before LLMs
- Traditional parsers were brittle, inflexible
- Couldn't handle variations like "$100 off $300" vs "$250 off this week only"
- Resume parsing, deal parsing, etc. required heavyweight, expensive solutions

**LLM Solution:**
- Intelligent parsing with human-like understanding
- Handles complex variations and exceptions
- Structured outputs guarantee format
- Trade-offs: cost, latency, unpredictability

---

## 2. Scanner Agent Implementation {#scanner-agent}

### Purpose
Scrapes RSS feeds, parses unstructured deal data, and uses structured outputs to extract the 5 most promising deals with clear prices.

### RSS Feed Scraping

```python
from agents.deals import ScrapedDeal, DealSelection
import feedparser

# Define RSS feeds
RSS_FEEDS = [
    "https://www.dealnews.com/rss/c142/",  # Electronics
    "https://www.dealnews.com/rss/c39/",   # Computers
    "https://www.dealnews.com/rss/f1912/"  # Smart Home
]

# Fetch deals
deals = ScrapedDeal.fetch(show_progress=True)
# Returns ~30 deals (10 from each feed)
```

### Prompt Engineering

```python
SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, 
by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation.
Be careful with products described as "$XXX off" - this isn't the actual price.
Only respond with products when you are highly confident about the price."""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, 
selecting those which have the most detailed product description and a clear price > 0.
Rephrase the description to be a summary of the product itself, not the terms of the deal.
Be careful with "$XXX off" - only respond when highly confident about the price.

Deals:
"""

USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."
```

### Creating User Prompt

```python
def make_user_prompt(scraped):
    user_prompt = USER_PROMPT_PREFIX
    user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
    user_prompt += USER_PROMPT_SUFFIX
    return user_prompt

user_prompt = make_user_prompt(deals)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt}
]
```

### Structured Output Call

```python
response = openai.chat.completions.parse(
    model="gpt-5-mini",
    messages=messages,
    response_format=DealSelection,
    reasoning_effort="minimal"
)

results = response.choices[0].message.parsed
# results is a DealSelection object with 5 Deal objects
```

### Scanner Agent Class

```python
class ScannerAgent:
    COLOR = "\033[36m"  # Cyan
    
    def __init__(self):
        self.log("Scanner Agent is initializing")
        self.log("Scanner Agent is ready")
    
    def scan(self) -> DealSelection:
        """Fetch deals from RSS feed and select top 5 using structured outputs"""
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        deals = ScrapedDeal.fetch()
        self.log(f"Scanner Agent received {len(deals)} deals not already scraped")
        
        self.log("Scanner Agent is calling OpenAI using Structured Outputs")
        user_prompt = self.make_user_prompt(deals)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response = openai.chat.completions.parse(
            model="gpt-5-mini",
            messages=messages,
            response_format=DealSelection
        )
        
        results = response.choices[0].message.parsed
        self.log(f"Scanner Agent received {len(results.deals)} selected deals")
        return results
```

---

## 3. Messaging Agent with Pushover {#messaging-agent}

### Pushover Setup

**Why Pushover?**
- Simple push notifications to your phone
- Free for first month
- Much simpler than Twilio (no paperwork for anti-spam)
- Perfect for self-messaging

**Setup Steps:**
1. Visit https://pushover.net/
2. Sign up for account
3. Get USER token (starts with 'u') from dashboard
4. Create Application/API Token (starts with 'a')
5. Add to `.env`:
   ```
   PUSHOVER_USER=u...
   PUSHOVER_TOKEN=a...
   ```
6. Install Pushover app on phone

### Basic Push Function

```python
import requests
import os

pushover_user = os.getenv('PUSHOVER_USER')
pushover_token = os.getenv('PUSHOVER_TOKEN')
pushover_url = "https://api.pushover.net/1/messages.json"

def push(message):
    payload = {
        "user": pushover_user,
        "token": pushover_token,
        "message": message
    }
    requests.post(pushover_url, data=payload)

# Test it
push("MASSIVE DEAL!!")
```

### Messaging Agent with LLM Hype

```python
from anthropic import Anthropic

class MessagingAgent:
    COLOR = "\033[37m"  # White
    
    def __init__(self):
        self.log("Messaging Agent is initializing")
        self.anthropic = Anthropic()
        self.log("Messaging Agent has initialized Pushover and Chatgpt")
    
    def push(self, message: str):
        """Send push notification"""
        self.log("Messaging Agent is sending a push notification")
        payload = {
            "user": pushover_user,
            "token": pushover_token,
            "message": message,
            "sound": "cashregister"  # Ka-ching!
        }
        requests.post(pushover_url, data=payload)
    
    def craft_juicy_message(self, description: str, price: float, 
                           estimate: float, url: str) -> str:
        """Use Claude to write hype message"""
        self.log("Messaging Agent is using chatgpt to craft the message")
        prompt = f"""Write an exciting 2-sentence push notification about this deal:
        Product: {description}
        Deal Price: ${price}
        Estimated Value: ${estimate}
        Make it exciting and urgent!"""
        
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.content[0].text
    
    def notify(self, description: str, deal_price: float, 
               estimated_true_value: float, url: str):
        """Craft hype message and send notification"""
        message = self.craft_juicy_message(description, deal_price, 
                                           estimated_true_value, url)
        self.push(message)
        self.log("Messaging Agent has completed")
```

---

## 4. Autonomous Planning Agent {#planning-agent}

### Core Concepts

**Agentic AI Definitions:**
1. **Sam Altman (OpenAI):** AI systems that can work independently
2. **2025 Consensus:** LLM controls the workflow
3. **Emerging Definition:** LLM in a loop with tools to achieve a goal

**Hallmarks of Agentic Solutions:**
- Breaking bigger problems into smaller LLM-powered steps
- Using tools (function calling)
- Structured outputs for orchestration
- Agent environment for communication
- Planning agent to coordinate
- Autonomy and memory beyond single chat sessions

### Tool Calling Fundamentals

**How It Works:**
1. Define functions with docstrings
2. Create JSON descriptions of functions
3. Pass tools to LLM in system prompt
4. LLM responds with `finish_reason="tool_calls"`
5. Execute functions and return results
6. Loop until LLM responds without tool calls

### Fake Functions for Testing

```python
def scan_the_internet_for_bargains() -> str:
    """This tool scans the internet for great deals"""
    print("Fake function to scan the internet")
    return test_results.model_dump_json()

def estimate_true_value(description: str) -> str:
    """This tool estimates the true value of a product"""
    print(f"Fake function to estimate value - always returns $300")
    return f"Product {description} has estimated value of $300"

def notify_user_of_deal(description: str, deal_price: float, 
                        estimated_true_value: float, url: str) -> str:
    """This tool notifies the user of a great deal"""
    print(f"Fake function to notify user of {description}")
    return "notification sent ok"
```

### Tool JSON Definitions

```python
scan_function = {
    "name": "scan_the_internet_for_bargains",
    "description": "Returns top bargains scraped from the internet",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}

estimate_function = {
    "name": "estimate_true_value",
    "description": "Given description of item, estimate its worth",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item to be estimated"
            }
        },
        "required": ["description"],
        "additionalProperties": False
    }
}

notify_function = {
    "name": "notify_user_of_deal",
    "description": "Send user a push notification about the most compelling deal",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Item description"},
            "deal_price": {"type": "number", "description": "Deal price"},
            "estimated_true_value": {"type": "number", "description": "Estimated value"},
            "url": {"type": "string", "description": "Deal URL"}
        },
        "required": ["description", "deal_price", "estimated_true_value", "url"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": scan_function},
    {"type": "function", "function": estimate_function},
    {"type": "function", "function": notify_function}
]
```

### Tool Call Handler

```python
def handle_tool_call(message):
    """Execute tools associated with this message"""
    results = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Dynamic function lookup (use mapping in production)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results
```

### Agent Loop - The Critical Pattern

```python
system_message = "You find great deals using your tools and notify the user."
user_message = """
First, use your tool to scan the internet for bargain deals.
Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where price is much lower than estimate,
and use your tool to notify the user.
Then just reply OK to indicate success.
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
]

# THE AGENT LOOP - This is what makes it agentic!
done = False
while not done:
    response = openai.chat.completions.create(
        model="gpt-5.1",
        messages=messages,
        tools=tools
    )
    
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        results = handle_tool_call(message)
        messages.append(message)
        messages.extend(results)
    else:
        done = True

final_response = response.choices[0].message.content
```

---

## 5. Multi-Agent System Architecture {#multi-agent-system}

### Complete Agent Hierarchy

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

### Autonomous Planning Agent Implementation

```python
class AutonomousPlanningAgent:
    COLOR = "\033[32m"  # Green
    MODEL = "gpt-5.1"
    
    def __init__(self, collection):
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Autonomous Planning Agent is ready")
    
    def scan_the_internet_for_bargains(self) -> str:
        """Tool: Scan internet for deals"""
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan()
        return results.model_dump_json()
    
    def estimate_true_value(self, description: str) -> str:
        """Tool: Estimate true value via Ensemble Agent"""
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate}"
    
    def notify_user_of_deal(self, description: str, deal_price: float,
                           estimated_true_value: float, url: str) -> str:
        """Tool: Notify user of deal"""
        self.log("Autonomous Planning agent is notifying user")
        self.messenger.notify(description, deal_price, estimated_true_value, url)
        return "notification sent"
    
    def handle_tool_call(self, message):
        """Execute tool calls"""
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal
        }
        
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })
        return results
    
    def plan(self, memory=None) -> str:
        """Execute autonomous planning loop"""
        self.log("Autonomous Planning Agent is kicking off a run")
        
        system_message = "You find great deals using your tools and notify user."
        user_message = """
        First, use your tool to scan the internet for bargain deals.
        Then for each deal, use your tool to estimate its true value.
        Then pick the single most compelling deal where price is much lower than estimate,
        and use your tool to notify the user.
        Then just reply OK to indicate success.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # AGENT LOOP
        done = False
        while not done:
            response = openai.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.tools
            )
            
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with reply: {reply}")
        return reply
```

### Model Call Statistics

**Total: 34 Model Calls**
- 29 LLM calls
- 5 Neural network calls

**Models Used:**
- **3 Frontier Models:**
  - GPT-5-mini (scanning)
  - GPT-5.1 (planning, RAG)
  - Claude Sonnet 4-5 (messaging)

- **3 Open Source Models:**
  - Fine-tuned Llama 3.2 3B (specialist pricing on Modal)
  - all-MiniLM-L6-v2 (embeddings for RAG)
  - Llama 3.2 via Ollama (preprocessing)

---

## 6. Complete Framework Integration {#framework-integration}

### Deal Agent Framework

```python
class DealAgentFramework:
    def __init__(self):
        self.memory = self.read_memory()
        self.planner = None
        self.collection = None
    
    @staticmethod
    def read_memory() -> list[Opportunity]:
        """Load memory from JSON file"""
        try:
            with open("memory.json", "r") as f:
                data = json.load(f)
                return [Opportunity(**item) for item in data]
        except FileNotFoundError:
            return []
    
    @staticmethod
    def write_memory(opportunities: list[Opportunity]):
        """Save memory to JSON file"""
        with open("memory.json", "w") as f:
            data = [opp.model_dump() for opp in opportunities]
            json.dump(data, f, indent=2)
    
    @staticmethod
    def reset_memory():
        """Reset to initial 2 deals"""
        initial_deals = [...]  # Predefined deals
        DealAgentFramework.write_memory(initial_deals)
    
    def init_agents_as_needed(self):
        """Initialize ChromaDB and Planning Agent"""
        if not self.collection:
            client = chromadb.PersistentClient(path="products_vectorstore")
            self.collection = client.get_or_create_collection('products')
        
        if not self.planner:
            self.planner = AutonomousPlanningAgent(self.collection)
    
    def run(self) -> list[Opportunity]:
        """Execute one cycle of deal hunting"""
        self.init_agents_as_needed()
        
        # Run autonomous planning agent
        self.planner.plan(memory=self.memory)
        
        # Update memory with new opportunities
        new_opportunities = [...]  # Extract from planner results
        self.memory.extend(new_opportunities)
        self.write_memory(self.memory)
        
        return new_opportunities
```

### Gradio UI Implementation

```python
import gradio as gr
import logging
import queue
import threading

class App:
    def __init__(self):
        self.agent_framework = None
    
    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework
    
    def run(self):
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_data = gr.State([])
            
            def table_for(opps):
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]
            
            def do_run():
                new_opportunities = self.get_agent_framework().run()
                return table_for(new_opportunities)
            
            def do_select(selected_index: gr.SelectData):
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.get_agent_framework().planner.messenger.alert(opportunity)
            
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px">'
                           '<strong>The Price is Right</strong> - '
                           'Autonomous Agent Framework that hunts for deals</div>')
            
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">'
                           'A proprietary fine-tuned LLM deployed on Modal and '
                           'a RAG pipeline with a frontier model collaborate to '
                           'send push notifications with great online deals.</div>')
            
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Deals found so far", "Price", "Estimate", 
                            "Discount", "URL"],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
            
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(show_label=False)
            
            # Auto-run on load
            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )
            
            # Auto-run every 5 minutes
            timer = gr.Timer(value=300, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )
            
            # Click to send notification
            opportunities_dataframe.select(do_select)
        
        ui.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    App().run()
```

### Logging with Colors

```python
class ColoredLogger:
    COLORS = {
        "scanner": "\033[36m",    # Cyan
        "ensemble": "\033[33m",   # Yellow
        "specialist": "\033[31m", # Red
        "frontier": "\033[34m",   # Blue
        "neural": "\033[35m",     # Purple
        "messenger": "\033[37m",  # White
        "planner": "\033[32m",    # Green
        "reset": "\033[0m"
    }
    
    def log(self, agent_name: str, message: str):
        color = self.COLORS.get(agent_name, self.COLORS["reset"])
        formatted = f"{color}[{agent_name.title()} Agent] {message}{self.COLORS['reset']}"
        logging.info(formatted)
```

---

## Key Takeaways for LLM Engineers

### 1. Structured Outputs
- Use Pydantic models for guaranteed output format
- Leverage constrained decoding for reliability
- Perfect for parsing unstructured data intelligently

### 2. Tool Calling
- Define tools as JSON schemas
- LLM decides when to call tools autonomously
- Loop until task completion

### 3. Agent Loop Pattern
```python
done = False
while not done:
    response = llm_call_with_tools()
    if needs_tool_call:
        execute_tools()
        continue
    else:
        done = True
```

### 4. Multi-Agent Orchestration
- Planning agent coordinates specialist agents
- Each agent has specific responsibility
- Tools abstract away implementation details

### 5. Memory and Persistence
- Simple JSON file for state persistence
- Enables autonomy beyond single sessions
- Foundation for continuous operation

### 6. Production Considerations
- Use mapping dictionaries instead of `globals()`
- Add proper error handling
- Implement retry logic
- Monitor costs and latency
- Log everything with colors for debugging

### 7. Commercial Applications
- Intelligent parsing (resumes, deals, documents)
- Autonomous monitoring and alerting
- Multi-model ensemble for accuracy
- RAG for domain-specific knowledge
- Fine-tuned models for specialized tasks

---

## Complete Code Structure

```
project/
├── agents/
│   ├── scanner_agent.py          # RSS scraping + structured outputs
│   ├── ensemble_agent.py         # Coordinates pricing models
│   ├── specialist_agent.py       # Fine-tuned model on Modal
│   ├── frontier_agent.py         # GPT-5 with RAG
│   ├── neural_network_agent.py   # Deep neural network
│   ├── messaging_agent.py        # Pushover + Claude
│   ├── autonomous_planning_agent.py  # Orchestrator with tools
│   ├── preprocessor.py           # Text preprocessing
│   └── deals.py                  # Pydantic models
├── deal_agent_framework.py       # Main framework
├── price_is_right.py            # Gradio UI
├── memory.json                   # Persistent state
├── products_vectorstore/         # ChromaDB
└── .env                          # API keys
```

---

## Running the Complete System

```bash
# Reset memory
python -c "from deal_agent_framework import DealAgentFramework; DealAgentFramework.reset_memory()"

# Run the UI
uv run price_is_right.py

# Or in notebook
!uv run price_is_right.py
```

The system will:
1. Auto-run on startup
2. Scan RSS feeds every 5 minutes
3. Estimate prices using ensemble of models
4. Send push notifications for great deals
5. Display all deals in interactive UI
6. Allow manual notification by clicking deals

---

## Advanced Extensions

### Using Agent Frameworks
- OpenAI Agents SDK
- LangGraph
- CrewAI
- AutoGen

### MCP (Model Context Protocol)
- Standardized tool interfaces
- Server-based tool execution
- Cross-platform compatibility

### Production Deployment
- Containerization (Docker)
- Serverless (AWS Lambda, Modal)
- Monitoring (Prometheus, Grafana)
- Cost tracking and optimization

---
