## Overview: Agentic AI & Serverless Deployment

This is the culmination of the LLM Engineering, bringing together all previous concepts into an autonomous agentic AI system called "The Price is Right" - a deal-finding application that monitors RSS feeds, estimates prices, and sends push notifications.

---
## Key Concepts to Remember

### 1. Agentic AI Definitions
**Three Modern Definitions of Agents:**

	1.  **OpenAI Definition (Anthropomorphic)**: AI systems that can do work for you independently - like delegating to a human worker
		- Example: "Find me a Chinese restaurant reservation for Thursday"
  
	2.  **Emerging 2025 Definition (Control-Based)**: An AI system where the LLM controls the workflow

		- The LLM decides which steps to execute and in what order
		- Contrasts with "agentic workflows" where Python orchestrates fixed sequences

  

	3.  **Tool-Loop Definition (Most Common)**: An agent runs tools in a loop to achieve a goal

		- LLM equipped with tools
		- Loops until objective is achieved
		- Example: Claude Code with its to-do list

  

**Important Distinction:**

	-  **Agentic Workflow**: Multiple LLM calls orchestrated by Python (A → B → C)
	-  **True Agent**: LLM decides the execution order dynamically

 

---

  

### 2. Modal.com - Serverless AI Platform

  

**Why Modal?**

-  **Easy Infrastructure-as-Code**: Define hardware requirements in Python
-  **Pay-per-Use**: Only pay for actual compute time
-  **Free Tier**: $30/month free credits
-  **Simple Deployment**: Same code runs locally or remotely

  

**Core Modal Concepts:**
  

```python

import modal

from modal import Image

# Define infrastructure

app = modal.App("app-name")

image = Image.debian_slim().pip_install("package1",  "package2")

# Decorate functions for remote execution
@app.function(image=image,  gpu="T4",  timeout=1800)
def  my_function():
	# Your code here
pass

```

  

**Key Modal Features:**


1.  **Local vs Remote Execution:**

```python

# Run locally
with app.run():
	result = my_function.local()
  

# Run remotely on Modal's cloud
with app.run():
	result = my_function.remote()

```
 

2.  **Region Selection:**

```python

@app.function(image=image,  region="eu")  # Run in Europe for GDPR compliance

```

  

3.  **Persistent Storage with Volumes:**

```python
from modal import Volume
hf_cache_volume = Volume.from_name("hf-hub-cache",  create_if_missing=True)


@app.cls( volumes={"/cache": hf_cache_volume}  # Persist model weights)

class  MyModel:
	pass

```
  

4.  **Deployment:**

```bash

# Deploy to Modal
uv  run  modal  deploy  -m  module_name
# Get handle to deployed function
pricer  =  modal.Function.from_name("service-name",  "function-name")
result  =  pricer.remote(input_data)
```

  

**Modal Configuration:**
1.  **Set API Tokens:**
```bash
uv  run  modal  token  set  --token-id  ak-xxx  --token-secret  as-xxx
```

2.  **Configure Secrets in Modal Dashboard:**

- Name: `huggingface-secret`
- Key: `HF_TOKEN`
- Value: `hf_...`


3.  **Reference Secrets in Code:**

```python
secrets =  [modal.Secret.from_name("huggingface-secret")]
@app.function(secrets=secrets)
	def  my_function():
		# HF_TOKEN now available as environment variable
		pass
```

  

---

  

### 3. Fine-Tuned Model Deployment
**Architecture Pattern:**
  

```python

# pricer_service.py - Production-ready deployment
import modal
from modal import Volume, Image

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
"torch",  "transformers",  "bitsandbytes",  "accelerate",  "peft"
)
secrets =  [modal.Secret.from_name("huggingface-secret")]
GPU =  "T4"
BASE_MODEL =  "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL =  f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR =  "/cache"
hf_cache_volume = Volume.from_name("hf-hub-cache",  create_if_missing=True)
@app.cls(
	image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
	secrets=secrets,
	gpu=GPU,
	timeout=1800,
	volumes={CACHE_DIR: hf_cache_volume},
)

class  Pricer:
@modal.enter()  # Runs once on container startup
	def  setup(self):
		from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
		from peft import PeftModel
		# Load base model with quantization
		quant_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_quant_type="nf4",
		)

	self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
	self.base_model = AutoModelForCausalLM.from_pretrained(	BASE_MODEL,  quantization_config=quant_config)

# Load LoRA adapters
self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_MODEL)

	@modal.method()
	def  price(self,  description:  str)  ->  float:
	# Inference code
	prompt =  f"{QUESTION}\n\n{description}\n\n{PREFIX}"
	inputs =  self.tokenizer.encode(prompt,  return_tensors="pt").to("cuda")
	outputs =  self.fine_tuned_model.generate(inputs,  max_new_tokens=5)
	result =  self.tokenizer.decode(outputs[0])
	# Extract price from result
	return extracted_price

```

**Key Improvements Over Ephemeral:**
-  **Class-based**: Allows persistent state with `@modal.enter()`
-  **Volume Caching**: Model weights cached on disk (~30s startup vs 1+ min)
-  **Warm Containers**: Subsequent calls are near-instant
-  **Production-Ready**: Can be called from any Python code  

---

  
### 4. Data Preprocessing Pattern

**Why Preprocess?**
- Fine-tuned models expect consistent input format
- Training data was preprocessed → inference data must match
- Improves reliability and accuracy

**Preprocessor Implementation:**

```python

from litellm import completion
SYSTEM_PROMPT =  """Create a concise description of a product. Respond only in this format:
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

class  Preprocessor:

	def  __init__(self,  model_name="ollama/llama3.2"):
	self.model_name = model_name
	self.total_cost =  0
	def  preprocess(self,  text:  str)  ->  str:
	messages =  [
	{"role":  "system",  "content": SYSTEM_PROMPT},
	{"role":  "user",  "content": text}
	]
	response = completion(messages=messages,  model=self.model_name)
	return response.choices[0].message.content
```

 

**Usage Pattern:**

```python
preprocessor = Preprocessor()
raw_text =  "Quadcast HyperX condenser mic, connects via usb-c"
formatted_text = preprocessor.preprocess(raw_text)
price = pricer.remote(formatted_text)  # Now in expected format
```

  

**Configuration Options:**

- Default model: `ollama/llama3.2` (local, free)
- Alternative: `groq/openai/gpt-oss-20b` (cloud, faster)
- Set via environment: `PRICER_PREPROCESSOR_MODEL=groq/...`
 

---

  

### 5. Agent Architecture Pattern
 

**Base Agent Class:**
  

```python
import logging
class  Agent:
	"""Abstract superclass for all agents"""
	# ANSI color codes for logging
	RED =  '\033[31m'
	GREEN =  '\033[32m'
	YELLOW =  '\033[33m'
	BLUE =  '\033[34m'
	MAGENTA =  '\033[35m'
	CYAN =  '\033[36m'
	RESET =  '\033[0m'
	name:  str  =  ""
	color:  str  =  '\033[37m'
	def  log(self,  message):
		"""Log with color-coded agent identification"""
		color_code =  '\033[40m'  +  self.color # Black background + color
		logging.info(f"{color_code}[{self.name}] {message}{self.RESET}")
```

  

**Specialist Agent Implementation:**  

```python
	import modal
	from agents.agent import Agent
	class  SpecialistAgent(Agent):
		"""Agent that runs fine-tuned LLM on Modal"""
		name =  "Specialist Agent"
		color = Agent.RED
		def  __init__(self):
			self.log("Initializing - connecting to modal")
			# Get handle to deployed Modal class
			Pricer = modal.Cls.from_name("pricer-service",  "Pricer")
			self.pricer = Pricer()  # Create instance
			def  price(self,  description:  str)  ->  float:
			self.log("Calling remote fine-tuned model")
			result =  self.pricer.price.remote(description)
			self.log(f"Completed - predicting ${result:.2f}")
			return result

```

  

**Agent Design Principles:**

1.  **Start Simple**: Begin with one LLM call, not multiple agents
2.  **Divide by Need**: Split into multiple agents only when it improves results
3.  **Avoid Anthropomorphization**: Don't assign "human roles" - focus on business objectives
4.  **Evaluate First**: Test single-agent approach before building complex architectures

---

 

### 6. The Complete Architecture

  
**Seven Agents Working Together:**
1.  **Specialist Agent** (RED): Calls fine-tuned model on Modal for pricing
2.  **Frontier Agent** (TBD): Uses frontier models with RAG for alternative pricing
3.  **Ensemble Agent** (TBD): Combines multiple model predictions
4.  **Scanner Agent** (TBD): Monitors RSS feeds for deals
5.  **Messenger Agent** (TBD): Sends push notifications
6.  **Planner Agent** (TBD): Orchestrates all other agents
7.  **Framework Agent** (TBD): Handles memory and logging


**System Flow:**

```
RSS Feeds → Scanner Agent → Planner Agent → Ensemble Agent
├─ Specialist Agent (Modal)
└─ Frontier Agent (RAG)
→ Messenger Agent → Push Notification
```
  
---

## Critical Code Patterns

### Pattern 1: Modal Ephemeral vs Deployed

**Ephemeral (Quick Testing):**

```python
from my_module import app, my_function
with app.run():
result = my_function.remote(input_data)
```

  

**Deployed (Production):**

```bash
uv  run  modal  deploy  -m  my_module
```

```python
my_function = modal.Function.from_name("app-name",  "function-name")
result = my_function.remote(input_data)
```

  

### Pattern 2: Container Lifecycle Management

**Cold Start Problem:**
- Modal containers sleep after 2 minutes of inactivity
- Cold start: ~30 seconds to reload model into memory
- Warm start: Near-instant response

 

**Solutions:**

```python
# Option 1: Keep containers alive longer (costs more)
MIN_CONTAINERS =  1  # Always keep 1 container running
# Option 2: Adjust idle timeout
# Via Modal CLI or dashboard settings
```

### Pattern 3: Quantization for Deployment
 

```python

from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
load_in_4bit=True,  # 4-bit quantization
bnb_4bit_use_double_quant=True,  # Double quantization
bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
)

model = AutoModelForCausalLM.from_pretrained(
		model_name,
		quantization_config=quant_config,
		device_map="auto"  # Automatic device placement
		)

```

  

**Benefits:**

- Reduces memory footprint by ~75%
- Enables larger models on smaller GPUs
- Minimal accuracy loss with NF4 quantization

  

### Pattern 4: LoRA Adapter Loading
 

```python

from peft import PeftModel
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
# Load LoRA adapters on top
fine_tuned_model = PeftModel.from_pretrained( base_model,FINETUNED_MODEL,revision=REVISION # Specific commit/version)

```

  

**Key Points:**

- LoRA adapters are small (~few MB vs GB for full model)
- Can swap adapters without reloading base model
- Enables efficient multi-task deployment

---

  

## Best Practices & Lessons
### 1. Agent Development Workflow


**DON'T:**
- Jump straight to multi-agent architecture
- Assign agents based on "human roles"
- Build complex systems without evaluation
 
**DO:**
1. Start with single LLM call
2. Establish evaluation metrics
3. Test if single agent solves problem
4. Split into multiple agents only if it improves results
5. Focus on business objectives, not architecture elegance

  

### 2. Modal Deployment Strategy

**Development:**

- Use ephemeral functions for rapid iteration
- Test locally first with `.local()`
- Use `modal.enable_output()` for debugging
  

**Production:**

- Deploy with `modal deploy`
- Use class-based approach for state management
- Implement volume caching for model weights
- Monitor costs via Modal dashboard
  

### 3. Cost Optimization

**Modal Costs:**
- Free tier: $30/month
- Only pay for actual compute time
- Containers auto-sleep after 2 minutes
- Volume storage is minimal cost

**LLM API Costs:**

- Preprocessing: Use local Ollama when possible
- Fine-tuned model: One-time training cost, then free inference on Modal
- Frontier models: Use cheapest variants (gpt-4o-mini, claude-haiku)
 

### 4. Debugging & Observability

**Logging Pattern:**

```python
import logging
logging.basicConfig(level=logging.INFO,	format='%(asctime)s - %(message)s')

# Each agent logs with color coding
agent.log("Status message")  # Automatically color-coded
```

  

**Modal Monitoring:**

- Dashboard shows real-time execution
- View logs for each function call
- Track costs per deployment
- Monitor container lifecycle

---

  

## Common Pitfalls & Solutions
### Pitfall 1: Unicode Errors on Windows

**Problem:** Modal outputs emojis that Windows can't display
**Solution:**

```python
os.environ["PYTHONIOENCODING"]  =  "utf-8"

```
Or run Modal commands in terminal instead of notebook

### Pitfall 2: Secret Configuration
**Problem:** Modal can't find HuggingFace token
**Solution:** Verify in Modal dashboard:
- Secret name: `huggingface-secret` (matches code)
- Key: `HF_TOKEN` (environment variable name)
- Value: `hf_...` (actual token)

  

### Pitfall 3: Cold Start Delays
**Problem:** First call takes 30+ seconds
**Solution:**

- Accept it for development (free tier)
- Use `MIN_CONTAINERS=1` for production (costs more)
- Implement volume caching for model weights

 

### Pitfall 4: Inconsistent Preprocessing
**Problem:** Model gives poor results on raw text
**Solution:** Always preprocess inference data to match training format

```python

preprocessor = Preprocessor()
formatted = preprocessor.preprocess(raw_text)
result = model.predict(formatted)  # Not raw_text!
```
---

  

## Project: "The Price is Right"


**Business Objective:**
Autonomous system that finds online deals, estimates true value, and notifies user of best bargains

**Technical Components:**

1.  **Data Pipeline:**
- RSS feed monitoring
- Deal extraction and filtering
- Text preprocessing

2.  **Pricing Models:**
- Specialist: Fine-tuned Llama 3.2 on Modal
- Frontier: GPT-4 with RAG
- Ensemble: Combines both predictions


3.  **Orchestration:**
- Planner agent controls workflow
- Memory prevents duplicate notifications
- Logging for observability

  
4.  **User Interface:**
- Push notifications to phone
- Deal details and price estimates
- Action links
  

**Key Learnings:**
- Serverless deployment with Modal
- Multi-agent orchestration
- Production ML system design
- Cost-effective inference strategies


---

  

## Next Steps 


**Step 2:** RAG implementation for frontier agent

**Step 3:** Scanner and messenger agents

**Step 4:** Autonomous planner agent

**Step 5:** Complete system integration  

---


## Essential Commands Reference

```bash
# Modal Setup
uv  run  modal  token  set  --token-id  ak-xxx  --token-secret  as-xxx

# Deploy to Modal
uv  run  modal  deploy  -m  module_name

# Alternative: Deploy from terminal
modal  deploy  module_name.py
# Check Modal status
modal  app  list
modal  app  logs  app-name
```

  

```python
# Python Usage Patterns
# Ephemeral execution
with app.run():
result = function.remote(data)
# Deployed execution
func = modal.Function.from_name("app",  "func")
result = func.remote(data)

# Class-based execution
Cls = modal.Cls.from_name("app",  "ClassName")
instance = Cls()

result = instance.method.remote(data)

```

---

## Key Takeaways


1.  **Serverless AI is Production-Ready**: Modal makes deployment as easy as writing Python
2.  **Start Simple**: One agent → evaluate → split if needed
3.  **Preprocessing Matters**: Match inference format to training format
4.  **Cost Optimization**: Use quantization, caching, and free tiers strategically
5.  **Observability is Critical**: Color-coded logging, Modal dashboard, cost tracking
6.  **Agentic AI ≠ Complex**: Simple architectures often outperform complex ones
 
---
